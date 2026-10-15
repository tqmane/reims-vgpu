//! Frames for the terminal: QEMU's PPM screendump read, scaled to the cells it fills, and
//! written in the terminal's image protocol.
//!
//! - kitty: a PNG, placed over the cells (`c`, `r`), which the terminal scales to them. Frames
//!   take two image ids in turn, and each frame deletes the one before, so a frame replaces its
//!   predecessor without a blank between them and the terminal keeps no old frames.
//! - iTerm2: a PNG, sized in cells.
//! - sixel: the frame at the cells' size in pixels, in 240 colors: a 6×6×6 cube, dithered
//!   (4×4 Bayer), and 24 grays, which take near-gray pixels undithered, so the guest's gray
//!   interface and its text stay clean.
//! - half blocks: two pixels a cell, the upper half block's foreground over its background.
//!
//! Scaling averages the source pixels each output pixel covers, so text shrunk from a HiDPI
//! guest stays legible.

use super::term::Protocol;
use anyhow::{Context, Result, bail};
use std::io::Write;
use std::sync::Arc;

/// A screendump: `w`×`h` pixels of RGB at `data[off..]`.
#[derive(Clone)]
pub struct Frame {
    pub w: u32,
    pub h: u32,
    pub data: Arc<Vec<u8>>,
    pub off: usize,
}

impl Frame {
    pub fn rgb(&self) -> &[u8] {
        &self.data[self.off..self.off + (self.w * self.h * 3) as usize]
    }
}

/// Reads a binary PPM (P6, 8 bits a channel), as QEMU's screendump writes it: each row at its
/// pixman line buffer's stride (ui/ui-qmp-cmds.c, `ppm_save`), which pads a row of a width not
/// a multiple of 4, such as 1366's, to 4 bytes. Padded rows are copied together.
pub fn parse_ppm(data: Arc<Vec<u8>>) -> Result<Frame> {
    let b = &data[..];
    if !b.starts_with(b"P6") {
        bail!("the screendump is not a PPM");
    }
    let mut i = 2;
    let mut nums = [0u32; 3];
    for n in &mut nums {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        *n = std::str::from_utf8(&b[start..i])
            .ok()
            .and_then(|s| s.parse().ok())
            .context("the screendump's PPM header is cut short")?;
    }
    // One whitespace byte ends the header.
    i += 1;
    let [w, h, max] = nums;
    if max != 255 || w == 0 || h == 0 {
        bail!("the screendump is a {w}x{h} PPM of maximum {max}");
    }
    let row = (w * 3) as usize;
    let stride = b.len().saturating_sub(i) / h as usize;
    if stride < row {
        bail!("the screendump's pixels are cut short");
    }
    if stride == row {
        return Ok(Frame { w, h, data, off: i });
    }
    let mut px = Vec::with_capacity(row * h as usize);
    for y in 0..h as usize {
        let at = i + y * stride;
        px.extend_from_slice(&b[at..at + row]);
    }
    Ok(Frame {
        w,
        h,
        data: Arc::new(px),
        off: 0,
    })
}

/// Cells on the screen: `cols` by `rows` from `col`, `row`, counted from 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Place {
    pub col: u16,
    pub row: u16,
    pub cols: u16,
    pub rows: u16,
}

/// The cells a `w`×`h` frame fills inside `area`: as many as fit it at its own aspect, for cells
/// `cell` pixels in size, centered.
pub fn fit(w: u32, h: u32, area: Place, cell: (u16, u16)) -> Place {
    let (cw, ch) = (f64::from(cell.0.max(1)), f64::from(cell.1.max(1)));
    let s =
        (f64::from(area.cols) * cw / f64::from(w)).min(f64::from(area.rows) * ch / f64::from(h));
    let cols = ((f64::from(w) * s / cw).round() as u16).clamp(1, area.cols.max(1));
    let rows = ((f64::from(h) * s / ch).round() as u16).clamp(1, area.rows.max(1));
    Place {
        col: area.col + (area.cols.saturating_sub(cols)) / 2,
        row: area.row + (area.rows.saturating_sub(rows)) / 2,
        cols,
        rows,
    }
}

/// The size in pixels a frame is sent at to fill `p`: kitty's and iTerm2's are scaled by the
/// terminal, so a frame smaller than the cells goes as it is; sixel has the cells' pixels, its
/// height in whole sixels; half blocks two pixels a cell.
pub fn pixels(proto: Protocol, w: u32, h: u32, p: Place, cell: (u16, u16)) -> (u32, u32) {
    let bw = u32::from(p.cols) * u32::from(cell.0.max(1));
    let bh = u32::from(p.rows) * u32::from(cell.1.max(1));
    match proto {
        Protocol::Kitty | Protocol::Iterm if w <= bw && h <= bh => (w, h),
        Protocol::Kitty | Protocol::Iterm => (bw, bh),
        Protocol::Sixel => (bw, (bh / 6 * 6).max(6)),
        Protocol::Blocks => (u32::from(p.cols), u32::from(p.rows) * 2),
    }
}

/// One axis of a scale: for each output pixel, its first source pixel and the share of each
/// source pixel it covers, in 1/16384ths summing to 16384.
struct Axis {
    start: Vec<u32>,
    /// The output pixel's weights are `w[off[i]..off[i + 1]]`.
    off: Vec<usize>,
    w: Vec<u32>,
}

impl Axis {
    fn new(src: u32, dst: u32) -> Axis {
        let scale = f64::from(src) / f64::from(dst);
        let mut a = Axis {
            start: Vec::with_capacity(dst as usize),
            off: vec![0],
            w: Vec::new(),
        };
        for o in 0..dst {
            let lo = f64::from(o) * scale;
            let hi = (f64::from(o + 1) * scale).min(f64::from(src));
            let first = (lo.floor() as u32).min(src - 1);
            let last = ((hi.ceil() as u32).max(first + 1)).min(src);
            let begin = a.w.len();
            for i in first..last {
                let cover = (hi.min(f64::from(i + 1)) - lo.max(f64::from(i))).max(0.0);
                a.w.push((cover / (hi - lo) * 16384.0).round() as u32);
            }
            // Rounding leaves the sum a little off 16384: the largest share takes the difference.
            let sum: u32 = a.w[begin..].iter().sum();
            let big = begin
                + a.w[begin..]
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, w)| **w)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
            a.w[big] = (a.w[big] + 16384).saturating_sub(sum);
            a.start.push(first);
            a.off.push(a.w.len());
        }
        a
    }

    fn taps(&self, o: usize) -> (usize, &[u32]) {
        (
            self.start[o] as usize,
            &self.w[self.off[o]..self.off[o + 1]],
        )
    }
}

/// `src`, `sw`×`sh` pixels of RGB, scaled to `dw`×`dh`, each output pixel the average of the
/// source pixels it covers.
pub fn scale(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    if (sw, sh) == (dw, dh) {
        return src[..(sw * sh * 3) as usize].to_vec();
    }
    let (ax, ay) = (Axis::new(sw, dw), Axis::new(sh, dh));
    let (sw, dw, dh) = (sw as usize, dw as usize, dh as usize);
    // Rows scaled across, each channel 8.8 fixed point.
    let mut mid = vec![0u16; sh as usize * dw * 3];
    for y in 0..sh as usize {
        let row = &src[y * sw * 3..(y + 1) * sw * 3];
        let out = &mut mid[y * dw * 3..(y + 1) * dw * 3];
        for x in 0..dw {
            let (first, w) = ax.taps(x);
            let mut acc = [0u32; 3];
            for (k, &wk) in w.iter().enumerate() {
                let p = &row[(first + k) * 3..(first + k) * 3 + 3];
                acc[0] += u32::from(p[0]) * wk;
                acc[1] += u32::from(p[1]) * wk;
                acc[2] += u32::from(p[2]) * wk;
            }
            for c in 0..3 {
                out[x * 3 + c] = ((acc[c] + 32) >> 6) as u16;
            }
        }
    }
    let mut dst = vec![0u8; dw * dh * 3];
    let mut acc = vec![0u32; dw * 3];
    for y in 0..dh {
        let (first, w) = ay.taps(y);
        acc.fill(0);
        for (k, &wk) in w.iter().enumerate() {
            let row = &mid[(first + k) * dw * 3..(first + k + 1) * dw * 3];
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += u32::from(v) * wk;
            }
        }
        for (d, &a) in dst[y * dw * 3..(y + 1) * dw * 3].iter_mut().zip(&acc) {
            *d = ((a + (1 << 21)) >> 22).min(255) as u8;
        }
    }
    dst
}

/// What the main thread writes or draws for a frame.
pub enum Payload {
    /// Escape sequences for the terminal, starting with the cursor's move to the frame.
    Escapes(Vec<u8>),
    /// Half blocks: `cols`×`2 * rows` pixels of RGB, a cell's two pixels a row apart.
    Blocks(Vec<u8>),
}

pub struct Encoded {
    pub place: Place,
    /// The frame's size, as captured.
    pub frame: (u32, u32),
    /// The size the frame was sent at.
    pub sent: (u32, u32),
    pub payload: Payload,
}

/// The kitty image ids frames take in turn.
pub const KITTY_IDS: [u32; 2] = [0x7276_0001, 0x7276_0002];

/// Deletes the viewer's kitty images and frees their data.
pub fn kitty_delete_all() -> Vec<u8> {
    let mut out = Vec::new();
    for id in KITTY_IDS {
        let _ = write!(out, "\x1b_Ga=d,d=I,i={id},q=2\x1b\\");
    }
    out
}

/// Scales `f` to fill `area` and encodes it for `proto`; `turn` is the kitty id frames are at,
/// which advances.
pub fn encode(
    proto: Protocol,
    f: &Frame,
    area: Place,
    cell: (u16, u16),
    turn: &mut usize,
) -> Result<Encoded> {
    let place = fit(f.w, f.h, area, cell);
    let (pw, ph) = pixels(proto, f.w, f.h, place, cell);
    let rgb = scale(f.rgb(), f.w, f.h, pw, ph);
    let payload = match proto {
        Protocol::Kitty => {
            let id = KITTY_IDS[*turn % 2];
            let old = KITTY_IDS[(*turn + 1) % 2];
            *turn += 1;
            Payload::Escapes(kitty(&png(&rgb, pw, ph)?, place, id, old))
        }
        Protocol::Iterm => Payload::Escapes(iterm(&png(&rgb, pw, ph)?, place)),
        Protocol::Sixel => Payload::Escapes(sixel(&rgb, pw, ph, place)),
        Protocol::Blocks => Payload::Blocks(rgb),
    };
    Ok(Encoded {
        place,
        frame: (f.w, f.h),
        sent: (pw, ph),
        payload,
    })
}

fn png(rgb: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(rgb.len() / 8);
    let mut e = png::Encoder::new(&mut out, w, h);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.set_compression(png::Compression::Fast);
    let mut wr = e.write_header().context("encode a PNG")?;
    wr.write_image_data(rgb).context("encode a PNG")?;
    wr.finish().context("encode a PNG")?;
    Ok(out)
}

fn cursor_to(out: &mut Vec<u8>, p: Place) {
    let _ = write!(out, "\x1b[{};{}H", p.row + 1, p.col + 1);
}

fn kitty(png: &[u8], p: Place, id: u32, old: u32) -> Vec<u8> {
    let b64 = base64(png);
    let mut out = Vec::with_capacity(b64.len() + b64.len() / 4096 * 16 + 128);
    cursor_to(&mut out, p);
    // A chunk carries at most 4096 bytes of base64; q=2 keeps the terminal from answering.
    let chunks: Vec<&[u8]> = b64.chunks(4096).collect();
    for (i, c) in chunks.iter().enumerate() {
        let more = u8::from(i + 1 < chunks.len());
        if i == 0 {
            let _ = write!(
                out,
                "\x1b_Ga=T,f=100,t=d,q=2,i={id},p=1,c={},r={},C=1,m={more};",
                p.cols, p.rows
            );
        } else {
            let _ = write!(out, "\x1b_Gm={more},q=2;");
        }
        out.extend_from_slice(c);
        out.extend_from_slice(b"\x1b\\");
    }
    let _ = write!(out, "\x1b_Ga=d,d=I,i={old},q=2\x1b\\");
    out
}

fn iterm(png: &[u8], p: Place) -> Vec<u8> {
    let b64 = base64(png);
    let mut out = Vec::with_capacity(b64.len() + 128);
    cursor_to(&mut out, p);
    let _ = write!(
        out,
        "\x1b]1337;File=inline=1;size={};width={};height={};preserveAspectRatio=0;doNotMoveCursor=1:",
        png.len(),
        p.cols,
        p.rows
    );
    out.extend_from_slice(&b64);
    out.push(0x07);
    out
}

/// The sixel palette: 216 colors of a 6×6×6 cube, then 24 grays from 8 to 238.
const LEVELS: [u8; 6] = [0, 51, 102, 153, 204, 255];
const COLORS: usize = 240;

fn palette(i: usize) -> (u8, u8, u8) {
    if i < 216 {
        (LEVELS[i / 36], LEVELS[i / 6 % 6], LEVELS[i % 6])
    } else {
        let g = 8 + 10 * (i - 216) as u8;
        (g, g, g)
    }
}

const BAYER: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

/// For each gray level, the nearest gray of the palette: one of the ramp's, or of the cube's
/// diagonal.
fn gray_table() -> [u8; 256] {
    let mut t = [0u8; 256];
    for (v, slot) in t.iter_mut().enumerate() {
        let mut best = (u32::MAX, 0usize);
        let grays = (216..COLORS).chain((0..6).map(|k| k * 43));
        for i in grays {
            let d = (palette(i).0 as i32 - v as i32).unsigned_abs();
            if d < best.0 {
                best = (d, i);
            }
        }
        *slot = best.1 as u8;
    }
    t
}

fn quantize(rgb: &[u8], w: u32, h: u32) -> Vec<u8> {
    let grays = gray_table();
    let mut idx = vec![0u8; (w * h) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * w as usize + x;
            let (r, g, b) = (rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
            let (hi, lo) = (r.max(g).max(b), r.min(g).min(b));
            idx[i] = if hi - lo < 16 {
                grays[((u32::from(r) + u32::from(g) + u32::from(b)) / 3) as usize]
            } else {
                // A threshold from -25 to +23, a level's step being 51.
                let t = i32::from(BAYER[y & 3][x & 3]) * 51 / 16 - 25;
                let q = |c: u8| ((i32::from(c) + t + 25) / 51).clamp(0, 5) as u8;
                q(r) * 36 + q(g) * 6 + q(b)
            };
        }
    }
    idx
}

fn sixel(rgb: &[u8], w: u32, h: u32, p: Place) -> Vec<u8> {
    let idx = quantize(rgb, w, h);
    let (w, h) = (w as usize, h as usize);
    let mut out = Vec::with_capacity(w * h / 2);
    cursor_to(&mut out, p);
    // P2=1: pixels no color sets keep what is under them; raster attributes 1:1, w×h.
    let _ = write!(out, "\x1bP0;1;0q\"1;1;{w};{h}");
    let mut used = [false; COLORS];
    for &i in &idx {
        used[i as usize] = true;
    }
    for (i, _) in used.iter().enumerate().filter(|(_, u)| **u) {
        let (r, g, b) = palette(i);
        let pc = |v: u8| u32::from(v) * 100 / 255;
        let _ = write!(out, "#{i};2;{};{};{}", pc(r), pc(g), pc(b));
    }
    // Each band of six rows: a row of sixels for each color in it.
    let mut rows = vec![0u8; COLORS * w];
    let mut present = [false; COLORS];
    let mut list = Vec::with_capacity(COLORS);
    for band in (0..h).step_by(6) {
        list.clear();
        for k in 0..6.min(h - band) {
            let line = &idx[(band + k) * w..(band + k + 1) * w];
            for (x, &c) in line.iter().enumerate() {
                let c = c as usize;
                if !present[c] {
                    present[c] = true;
                    list.push(c);
                    rows[c * w..(c + 1) * w].fill(0);
                }
                rows[c * w + x] |= 1 << k;
            }
        }
        for (n, &c) in list.iter().enumerate() {
            present[c] = false;
            let _ = write!(out, "#{c}");
            let row = &rows[c * w..(c + 1) * w];
            let end = row.iter().rposition(|&v| v != 0).map_or(0, |e| e + 1);
            let mut x = 0;
            while x < end {
                let v = row[x];
                let run = row[x..end].iter().take_while(|&&u| u == v).count();
                let ch = 63 + v;
                if run >= 4 {
                    let _ = write!(out, "!{run}");
                    out.push(ch);
                } else {
                    out.extend(std::iter::repeat_n(ch, run));
                }
                x += run;
            }
            out.push(if n + 1 < list.len() { b'$' } else { b'-' });
        }
    }
    out.extend_from_slice(b"\x1b\\");
    out
}

/// Standard base64, padded.
pub fn base64(b: &[u8]) -> Vec<u8> {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        out.push(A[(n >> 18) as usize & 63]);
        out.push(A[(n >> 12) as usize & 63]);
        out.push(if c.len() > 1 {
            A[(n >> 6) as usize & 63]
        } else {
            b'='
        });
        out.push(if c.len() > 2 {
            A[n as usize & 63]
        } else {
            b'='
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32, px: impl Fn(u32, u32) -> [u8; 3]) -> Frame {
        let mut data = format!("P6\n{w} {h}\n255\n").into_bytes();
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&px(x, y));
            }
        }
        parse_ppm(Arc::new(data)).unwrap()
    }

    #[test]
    fn ppm_header_and_pixels() {
        let f = frame(3, 2, |x, y| [x as u8, y as u8, 7]);
        assert_eq!((f.w, f.h), (3, 2));
        assert_eq!(&f.rgb()[..6], &[0, 0, 7, 1, 0, 7]);
        assert!(parse_ppm(Arc::new(b"P6\n3 2\n255\n\0\0".to_vec())).is_err());
        assert!(parse_ppm(Arc::new(b"P5\n1 1\n255\n\0".to_vec())).is_err());
        // QEMU pads a row of 3 pixels, 9 bytes, to 12.
        let mut data = b"P6\n3 2\n255\n".to_vec();
        for y in 0..2u8 {
            for x in 0..3u8 {
                data.extend_from_slice(&[x, y, 9]);
            }
            data.extend_from_slice(&[0xee; 3]);
        }
        let f = parse_ppm(Arc::new(data)).unwrap();
        assert_eq!(
            f.rgb(),
            &[0, 0, 9, 1, 0, 9, 2, 0, 9, 0, 1, 9, 1, 1, 9, 2, 1, 9]
        );
    }

    #[test]
    fn scaling_averages_what_each_pixel_covers() {
        // Halving a checkerboard gives its gray; a flat color stays itself.
        let f = frame(
            4,
            4,
            |x, y| if (x + y) % 2 == 0 { [255; 3] } else { [0; 3] },
        );
        let s = scale(f.rgb(), 4, 4, 2, 2);
        assert!(s.iter().all(|&v| v == 127 || v == 128), "{s:?}");
        let f = frame(7, 5, |_, _| [10, 200, 30]);
        let s = scale(f.rgb(), 7, 5, 3, 2);
        assert!(s.chunks(3).all(|p| p == [10, 200, 30]), "{s:?}");
        // Up as well as down.
        let s = scale(f.rgb(), 7, 5, 20, 9);
        assert_eq!(s.len(), 20 * 9 * 3);
        assert!(s.chunks(3).all(|p| p == [10, 200, 30]));
        for (src, dst) in [(1920, 1597), (3840, 1600), (17, 23), (5, 5)] {
            let a = Axis::new(src, dst);
            for o in 0..dst as usize {
                assert_eq!(
                    a.taps(o).1.iter().sum::<u32>(),
                    16384,
                    "{src}->{dst} at {o}"
                );
            }
        }
    }

    #[test]
    fn a_frame_fits_its_aspect_centered() {
        let area = Place {
            col: 0,
            row: 1,
            cols: 200,
            rows: 58,
        };
        // 16:9 into 2000x1160 pixels of 10x20 cells: 1160 tall, 2062 wide does not fit, so
        // 2000 wide and 1125 tall, 56 rows.
        let p = fit(1920, 1080, area, (10, 20));
        assert_eq!(
            p,
            Place {
                col: 0,
                row: 2,
                cols: 200,
                rows: 56
            }
        );
        assert_eq!(
            pixels(Protocol::Kitty, 1920, 1080, p, (10, 20)),
            (1920, 1080)
        );
        assert_eq!(
            pixels(Protocol::Kitty, 3840, 2160, p, (10, 20)),
            (2000, 1120)
        );
        assert_eq!(
            pixels(Protocol::Sixel, 1920, 1080, p, (10, 20)),
            (2000, 1116)
        );
        assert_eq!(
            pixels(Protocol::Blocks, 1920, 1080, p, (10, 20)),
            (200, 112)
        );
    }

    #[test]
    fn kitty_frames_alternate_ids_and_delete_the_one_before() {
        let f = frame(8, 8, |_, _| [1, 2, 3]);
        let area = Place {
            col: 0,
            row: 1,
            cols: 4,
            rows: 2,
        };
        let mut turn = 0;
        let a = encode(Protocol::Kitty, &f, area, (8, 16), &mut turn).unwrap();
        let Payload::Escapes(a) = a.payload else {
            panic!()
        };
        let a = String::from_utf8(a).unwrap();
        assert!(
            a.starts_with("\x1b[2;1H\x1b_Ga=T,f=100,t=d,q=2,i=1920335873,p=1,c=4,r=2,C=1,m=0;"),
            "{a}"
        );
        assert!(a.ends_with("\x1b_Ga=d,d=I,i=1920335874,q=2\x1b\\"));
        let b = encode(Protocol::Kitty, &f, area, (8, 16), &mut turn).unwrap();
        let Payload::Escapes(b) = b.payload else {
            panic!()
        };
        let b = String::from_utf8(b).unwrap();
        assert!(b.contains("i=1920335874,p=1") && b.ends_with("i=1920335873,q=2\x1b\\"));
    }

    #[test]
    fn sixel_bands_and_colors() {
        // 2x7: a white top row of six, then a red row.
        let f = frame(2, 7, |_, y| if y < 6 { [255; 3] } else { [255, 0, 0] });
        let s = sixel(f.rgb(), 2, 7, Place::default());
        let s = String::from_utf8(s).unwrap();
        assert!(s.starts_with("\x1b[1;1H\x1bP0;1;0q\"1;1;2;7"), "{s}");
        // White is the cube's 215, red its 180.
        assert!(
            s.contains("#215;2;100;100;100") && s.contains("#180;2;100;0;0"),
            "{s}"
        );
        assert!(s.ends_with("#215~~-#180@@-\x1b\\"), "{s}");
        // Runs of four or more are counted.
        let f = frame(9, 1, |_, _| [255; 3]);
        let s = String::from_utf8(sixel(f.rgb(), 9, 1, Place::default())).unwrap();
        assert!(s.ends_with("#215!9@-\x1b\\"), "{s}");
    }

    #[test]
    fn grays_go_to_the_ramp_undithered() {
        let px = [128u8, 128, 128, 130, 126, 128, 0, 0, 0, 255, 255, 255];
        let idx = quantize(&px, 4, 1);
        // 128 is the ramp's 128 (8 + 10 * 12); black and white are the cube's corners.
        assert_eq!(idx, vec![228, 228, 0, 215]);
    }

    #[test]
    fn base64_pads() {
        assert_eq!(base64(b""), b"");
        assert_eq!(base64(b"f"), b"Zg==");
        assert_eq!(base64(b"fo"), b"Zm8=");
        assert_eq!(base64(b"foo"), b"Zm9v");
        assert_eq!(base64(b"foobar"), b"Zm9vYmFy");
    }
}
