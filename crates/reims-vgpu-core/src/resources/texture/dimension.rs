//! A texture's guest layout (its dimension record) and its check against the Metal texture,
//! [`validate_texture_dimension`].

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::num::NonZeroU64;

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::guest::records::GuestFeatures;
use crate::num::{self, ToUsize, Truncate};
use crate::platform::Platform;
use crate::platform::resource::{GpuResource, Texture};
use crate::resources::texture::pixel_format::{
    FLAG_TWO_PLANE, KIND_COMPRESSED, KIND_NORMAL, KIND_TWO_PLANE, guest_pixel_format_info,
};

/// `MTLTextureTypeCube`.
const TEXTURE_TYPE_CUBE: u64 = 5;
/// `MTLTextureTypeCubeArray`.
const TEXTURE_TYPE_CUBE_ARRAY: u64 = 6;
/// The mip count's 14 bits.
const MIP_LEVELS_MASK: u16 = 0x3fff;
/// The cube bit, bit 14.
const IS_CUBE: u16 = 0x4000;
/// The compressed bit, bit 15.
const IS_COMPRESSED: u16 = 0x8000;

/// Where a texture's guest pages are: the first virtual page and the length.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C)]
pub struct TexturePagingInfo {
    /// The first virtual page.
    pub page: u32,
    /// Padding.
    pub unknown04: [u8; 4],
    /// The length in bytes.
    pub length: u64,
}

const _: () = assert!(size_of::<TexturePagingInfo>() == 16);

/// One mip's guest layout: packed, 0x24 bytes.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct TextureMipDimension {
    /// Its offset in a slice.
    pub offset: u64,
    /// Its declared size.
    pub size: u64,
    /// Bytes per row.
    pub bytes_per_row: u64,
    /// Width in texels.
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Depth in texels.
    pub depth: u32,
}

const _: () = assert!(size_of::<TextureMipDimension>() == 0x24);

/// One plane's guest layout: packed, 0x18 bytes, followed by its mips. `bits` holds the mip
/// count (14 bits), the cube bit and the compressed bit.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
#[repr(C, packed)]
pub struct TextureDimension {
    /// The mip count, cube and compressed bits.
    pub bits: u16,
    /// The array length.
    pub array_length: u16,
    /// Where the texture starts in the guest pages.
    pub base_offset: u64,
    /// Bytes per slice (a slice is one array element, or one face of a cube).
    pub bytes_per_slice: u64,
    /// Block width in texels.
    pub block_width: u8,
    /// Block height in texels.
    pub block_height: u8,
    /// Block depth in texels.
    pub block_depth: u8,
    /// Bytes per block.
    pub block_bytes: u8,
}

const _: () = assert!(size_of::<TextureDimension>() == 0x18);

impl TextureDimension {
    /// The mip count (bits 0-13).
    pub fn mip_levels(&self) -> u16 {
        self.bits & MIP_LEVELS_MASK
    }

    /// Stores the mip count: `v`'s low 14 bits.
    pub fn set_mip_levels(&mut self, v: u32) {
        let low = v.low_u16() & MIP_LEVELS_MASK;
        self.bits = (self.bits & !MIP_LEVELS_MASK) | low;
    }

    /// The cube bit (bit 14).
    pub fn is_cube(&self) -> bool {
        self.bits & IS_CUBE != 0
    }

    /// The compressed bit (bit 15).
    pub fn is_compressed(&self) -> bool {
        self.bits & IS_COMPRESSED != 0
    }
}

/// A dimension record with its mips, as the resource manager reads it out of guest memory.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TextureDimensionRecord {
    /// The fixed part.
    pub header: TextureDimension,
    /// The mips.
    pub mips: Vec<TextureMipDimension>,
}

/// `a * x + b` into a `u32`, with overflow: the product goes through a `u32` first, then the sum
/// into the `u32` result; either overflowing reports it.
fn mul_add_overflow_u32(a: u64, x: u32, b: u64) -> (u32, bool) {
    // The exact product of a u64 and a u32 fits in u128.
    let product = u128::from(a).wrapping_mul(u128::from(x));
    let [p0, p1, p2, p3, ..] = product.to_le_bytes();
    let tmp = u32::from_le_bytes([p0, p1, p2, p3]);
    let s = u32::try_from(product).is_err();
    let (r, t) = num::add_u64_into_u32(b, u64::from(tmp));
    (r, s | t)
}

/// The next mip's size: half, but at least 1.
fn halve(x: u64) -> u64 {
    (x / const { NonZeroU64::new(2).unwrap() }).max(1)
}

/// `x` texels in whole blocks of `block`, in 64 bits.
fn blocks(x: u64, block: u8) -> u64 {
    let b = u64::from(block);
    num::div0(x.wrapping_add(b).wrapping_sub(1), b)
}

/// The checks both kinds share once a mip's rows and row length are known: its size and end,
/// then its place in the slice. `Some(end)` when the mip passes; every `None` has logged one
/// line.
fn check_mip_extent(
    mip: &TextureMipDimension,
    total_rows: u64,
    bytes_per_row: u64,
    slice_size: u64,
    last_mip_end: u64,
) -> Option<u64> {
    let (mip_size, overflow) = total_rows.overflowing_mul(bytes_per_row);
    if overflow {
        log::warn!(
            "texture dimension refused: a mip of {total_rows} rows of {bytes_per_row} bytes \
             overflows"
        );
        return None;
    }
    let mip_start = mip.offset;
    let (mip_end, overflow) = mip_start.overflowing_add(mip_size);
    if overflow {
        log::warn!(
            "texture dimension refused: a mip of {mip_size:#x} bytes at {mip_start:#x} overflows"
        );
        return None;
    }
    let declared = mip.size;
    if mip_size > declared {
        log::warn!(
            "texture dimension refused: a mip needs {mip_size:#x} bytes, more than its declared \
             {declared:#x}"
        );
        return None;
    }
    // A slice size of 0 bounds nothing.
    let mip_end_valid = mip_end <= slice_size || slice_size == 0;
    if !mip_end_valid {
        log::warn!(
            "texture dimension refused: a mip ends at {mip_end:#x}, past the slice's \
             {slice_size:#x} bytes"
        );
        return None;
    }
    // Mip 0 must end past 0 (the last end starts at 0).
    if last_mip_end >= mip_end {
        log::warn!(
            "texture dimension refused: a mip ends at {mip_end:#x}, not past the previous \
             one's {last_mip_end:#x}"
        );
        return None;
    }
    Some(mip_end)
}

/// Whether a texture's data fits its allocation, counting the last mip's final row as the bytes
/// its width needs (`real_slice_end`), not a full `bytes_per_row` stride. The slices before the
/// last are a full `bytes_per_slice` apart; only the last slice's trailing stride is unused.
///
/// Apple's framework counts every row's full stride (base plus slices times `bytes_per_slice`),
/// so it refuses a strided, buffer-backed texture whose last row ends within a stride of the
/// buffer's end. The bytes the texture reads are all there — its last row never reads past its
/// own width — so this accepts it, keeping SkyLight's zero-copy window tiles (a window wider
/// than two tiles would otherwise lose WindowServer; see docs/qemu.md).
fn slice_footprint_fits(
    base_offset: u64,
    slice_count: u32,
    bytes_per_slice: u64,
    real_slice_end: u64,
    allocation: u64,
) -> bool {
    // A zero slice size bounds nothing, as in [`check_mip_extent`] and Apple's own check (its
    // `base plus slices times zero` is the base alone): the mip extent does not bound it.
    let footprint = if bytes_per_slice == 0 {
        base_offset
    } else {
        base_offset
            .wrapping_add(u64::from(slice_count.saturating_sub(1)).wrapping_mul(bytes_per_slice))
            .wrapping_add(real_slice_end)
    };
    if footprint > allocation {
        log::warn!(
            "texture dimension refused: its data ends at {footprint:#x}, past the allocation's \
             {allocation:#x} bytes"
        );
        return false;
    }
    true
}

/// Logs a mip whose size is not the one its predecessors' halvings give (`want`: width, height,
/// depth).
fn log_mip_size_mismatch(mip: u32, m: &TextureMipDimension, want: [u64; 3]) {
    let (width, height, depth) = (m.width, m.height, m.depth);
    let [want_width, want_height, want_depth] = want;
    log::warn!(
        "texture dimension refused: mip {mip} is {width}x{height}x{depth}, where the texture's is \
         {want_width}x{want_height}x{want_depth}"
    );
}

/// The source mip at `mip`. Every caller's record holds at least `mip_levels` mips per plane,
/// so the default is never taken.
fn src_mip(src: &TextureDimensionRecord, mip: u32) -> TextureMipDimension {
    src.mips.get(mip.to_usize()).copied().unwrap_or_default()
}

/// Copies each plane's dimension and mips from `dimension` into `dest` and checks them against
/// `texture`, asking the texture's getters and the pixel format's description in a fixed order.
/// `N` is the plane count (1 or 2). Writes `is_pvrtc` before any check. `false` has logged
/// exactly one line.
///
/// On `false`, `dest` holds what was written by then: each plane's header up to the failing
/// plane, and the mips copied before the failing one.
#[expect(
    clippy::too_many_arguments,
    reason = "each input comes separately from the guest's command, the texture or the device"
)]
pub fn validate_texture_dimension<P: Platform, const N: usize>(
    dimension: [&TextureDimensionRecord; N],
    dest: [&mut TextureDimensionRecord; N],
    mip_levels: u32,
    generate_mipmaps: bool,
    texture_allocation_length: u64,
    texture: &P::Texture,
    is_pvrtc: &mut bool,
    features: &GuestFeatures,
) -> bool {
    const { assert!(N == 1 || N == 2, "a texture has one plane or two") };
    let plane_count = N.low_u32();
    let device = texture.device();
    let pixel_format = texture.pixel_format();
    let info = guest_pixel_format_info::<P>(device.as_ref(), pixel_format, features);
    *is_pvrtc = info.flags.wrapping_shr(13) & 1 != 0;
    if (plane_count == 2) != (info.flags & FLAG_TWO_PLANE != 0) {
        // The pixel format is asked a second time, for the line, whether or not the line is
        // written.
        let pixel_format = texture.pixel_format().low_u32();
        log::warn!(
            "texture dimension refused: plane count {plane_count} does not match pixel format \
             {pixel_format}"
        );
        return false;
    }
    for (plane, (src, dst)) in dimension.into_iter().zip(dest).enumerate() {
        // The header copied, then its 14-bit mip count set.
        dst.header = src.header;
        dst.mips.clear();
        dst.header.set_mip_levels(mip_levels);
        let pd = dst.header;
        let array_length = pd.array_length;
        let bytes_per_slice = pd.bytes_per_slice;
        let base_offset = pd.base_offset;
        // A cube counts six slices per array element.
        let slice_count =
            (if pd.is_cube() { 6_u32 } else { 1 }).wrapping_mul(u32::from(array_length));
        // The allocation span is computed into a `u32`, so a span over 4 GiB - 1 is refused as
        // an overflow.
        let (_, overflow) = mul_add_overflow_u32(bytes_per_slice, slice_count, base_offset);
        if overflow {
            log::warn!(
                "texture dimension refused: base {base_offset:#x} plus {slice_count} x \
                 {bytes_per_slice:#x} bytes overflows 32 bits"
            );
            return false;
        }
        // Whether the data fits the allocation is checked per kind below, from the last mip's
        // real extent (`slice_footprint_fits`), not the last row's full stride.
        // 64 bits against the u16.
        if texture.array_length() != u64::from(array_length) {
            // Asked a second time, for the line, whether or not the line is written.
            let texture_array_length = texture.array_length();
            log::warn!(
                "texture dimension refused: array length {array_length}, where the texture has \
                 {texture_array_length}"
            );
            return false;
        }
        // The getter first, then `generate_mipmaps`.
        if texture.mipmap_level_count() != u64::from(pd.mip_levels()) && !generate_mipmaps {
            // Asked a second time, for the line, whether or not the line is written.
            let texture_mip_levels = texture.mipmap_level_count();
            log::warn!(
                "texture dimension refused: {} mip levels, where the texture has \
                 {texture_mip_levels}",
                pd.mip_levels()
            );
            return false;
        }
        // The texture type is asked twice unless the first answer is a cube.
        let is_cube = texture.texture_type() == TEXTURE_TYPE_CUBE
            || texture.texture_type() == TEXTURE_TYPE_CUBE_ARRAY;
        if is_cube != pd.is_cube() {
            log::warn!(
                "texture dimension refused: the dimension's cube flag is {}, the texture's \
                 {is_cube}",
                pd.is_cube()
            );
            return false;
        }
        match info.kind {
            KIND_NORMAL | KIND_TWO_PLANE => {
                if pd.is_compressed() {
                    log::warn!(
                        "texture dimension refused: marked compressed, for an uncompressed pixel \
                         format"
                    );
                    return false;
                }
                if pd.block_width != 1 || pd.block_height != 1 || pd.block_depth != 1 {
                    log::warn!(
                        "texture dimension refused: blocks of {}x{}x{} texels, for an \
                         uncompressed pixel format",
                        pd.block_width,
                        pd.block_height,
                        pd.block_depth
                    );
                    return false;
                }
                let pixel_size = pd.block_bytes;
                // The plane's own description (N is 1 or 2).
                let plane_block_bytes = info
                    .planes
                    .get(plane)
                    .map(|p| p.block_bytes)
                    .unwrap_or_default();
                // 64 bits against the byte.
                if plane_block_bytes != u64::from(pixel_size) {
                    log::warn!(
                        "texture dimension refused: {pixel_size} bytes per pixel, where the pixel \
                         format's plane has {plane_block_bytes}"
                    );
                    return false;
                }
                // Sample count, width, height, depth.
                let sample_count = texture.sample_count();
                let bytes_per_block = pd.block_bytes;
                let mut width = texture.width();
                let mut height = texture.height();
                let mut depth = texture.depth();
                // 32 bits, wrapping.
                let bytes_per_pixel =
                    u32::from(bytes_per_block).wrapping_mul(sample_count.low_u32());
                let mut last_mip_end = 0u64;
                let mut real_slice_end = 0u64;
                for mip in 0..mip_levels {
                    let m = src_mip(src, mip);
                    dst.mips.push(m);
                    if width != u64::from(m.width)
                        || height != u64::from(m.height)
                        || depth != u64::from(m.depth)
                    {
                        log_mip_size_mismatch(mip, &m, [width, height, depth]);
                        return false;
                    }
                    // 64 bits, wrapping.
                    let row_length = width.wrapping_mul(u64::from(bytes_per_pixel));
                    let bytes_per_row = m.bytes_per_row;
                    if bytes_per_row < row_length {
                        log::warn!(
                            "texture dimension refused: mip {mip} has {bytes_per_row} bytes per \
                             row, fewer than the {row_length} its width needs"
                        );
                        return false;
                    }
                    // A zero pixel size (never: the block bytes equal Metal's) leaves the row
                    // length as the remainder.
                    if num::rem0(bytes_per_row, u64::from(bytes_per_pixel)) != 0 {
                        log::warn!(
                            "texture dimension refused: mip {mip} has {bytes_per_row} bytes per \
                             row, not a multiple of its {bytes_per_pixel} bytes per pixel"
                        );
                        return false;
                    }
                    // The mip's own height and depth.
                    let total_rows = u64::from(m.height).wrapping_mul(u64::from(m.depth));
                    let Some(mip_end) = check_mip_extent(
                        &m,
                        total_rows,
                        bytes_per_row,
                        bytes_per_slice,
                        last_mip_end,
                    ) else {
                        return false;
                    };
                    // The mip's data ends its last row at its width, not a full stride.
                    real_slice_end = m
                        .offset
                        .wrapping_add(total_rows.saturating_sub(1).wrapping_mul(bytes_per_row))
                        .wrapping_add(row_length);
                    // Each halving clamps to 1.
                    width = halve(width);
                    height = halve(height);
                    depth = halve(depth);
                    last_mip_end = mip_end;
                }
                if !slice_footprint_fits(
                    base_offset,
                    slice_count,
                    bytes_per_slice,
                    real_slice_end,
                    texture_allocation_length,
                ) {
                    return false;
                }
            }
            KIND_COMPRESSED => {
                if !pd.is_compressed() {
                    log::warn!(
                        "texture dimension refused: marked uncompressed, for a compressed pixel \
                         format"
                    );
                    return false;
                }
                let mut width = texture.width();
                let mut height = texture.height();
                let mut depth = texture.depth();
                // Every plane is checked against plane 0's description.
                // Bytes against 64-bit words.
                let p0 = info.planes[0];
                if u64::from(pd.block_width) != p0.block_width
                    || u64::from(pd.block_height) != p0.block_height
                    || u64::from(pd.block_depth) != p0.block_depth
                    || u64::from(pd.block_bytes) != p0.block_bytes
                {
                    log::warn!(
                        "texture dimension refused: blocks of {}x{}x{} texels and {} bytes, \
                         where the pixel format's are {}x{}x{} and {}",
                        pd.block_width,
                        pd.block_height,
                        pd.block_depth,
                        pd.block_bytes,
                        p0.block_width,
                        p0.block_height,
                        p0.block_depth,
                        p0.block_bytes
                    );
                    return false;
                }
                let mut last_mip_end = 0u64;
                let mut real_slice_end = 0u64;
                for mip in 0..mip_levels {
                    let m = src_mip(src, mip);
                    dst.mips.push(m);
                    if width != u64::from(m.width)
                        || height != u64::from(m.height)
                        || depth != u64::from(m.depth)
                    {
                        log_mip_size_mismatch(mip, &m, [width, height, depth]);
                        return false;
                    }
                    let blocks_wide = blocks(width, pd.block_width);
                    let bytes_per_row = m.bytes_per_row;
                    let block_size = pd.block_bytes;
                    let row_length = blocks_wide.wrapping_mul(u64::from(block_size));
                    if bytes_per_row < row_length {
                        log::warn!(
                            "texture dimension refused: mip {mip} has {bytes_per_row} bytes per \
                             row, fewer than the {row_length} its width needs"
                        );
                        return false;
                    }
                    if num::rem0(bytes_per_row, u64::from(block_size)) != 0 {
                        log::warn!(
                            "texture dimension refused: mip {mip} has {bytes_per_row} bytes per \
                             row, not a multiple of its {block_size} bytes per block"
                        );
                        return false;
                    }
                    let blocks_high = blocks(height, pd.block_height);
                    let blocks_deep = blocks(depth, pd.block_depth);
                    let total_rows = blocks_deep.wrapping_mul(blocks_high);
                    let Some(mip_end) = check_mip_extent(
                        &m,
                        total_rows,
                        bytes_per_row,
                        bytes_per_slice,
                        last_mip_end,
                    ) else {
                        return false;
                    };
                    // The mip's data ends its last row of blocks at its width, not a full stride.
                    real_slice_end = m
                        .offset
                        .wrapping_add(total_rows.saturating_sub(1).wrapping_mul(bytes_per_row))
                        .wrapping_add(row_length);
                    // Each halving clamps to 1.
                    width = halve(width);
                    height = halve(height);
                    depth = halve(depth);
                    last_mip_end = mip_end;
                }
                if !slice_footprint_fits(
                    base_offset,
                    slice_count,
                    bytes_per_slice,
                    real_slice_end,
                    texture_allocation_length,
                ) {
                    return false;
                }
            }
            // The kinds are 0, 1 and 2 (pixel_format.rs).
            _ => {}
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::platform::types::PixelFormatInfo;
    use crate::resources::resource::test_support::logged;
    use crate::testkit::impls::new_texture;
    use crate::testkit::objects::TexDesc;
    use crate::testkit::{FakePlatform, Fk, Step, kind};

    type Tex = Fk<kind::Texture>;

    /// A fake process.
    fn plat() -> FakePlatform {
        FakePlatform::new()
    }

    /// A texture of `p`'s device (no call logged).
    fn tex(p: &FakePlatform, d: TexDesc) -> Tex {
        let t = new_texture(p.world(), d, None, 0, None, false);
        t.state().device = Some(p.device().erase());
        t
    }

    fn desc(w: u64, h: u64, d: u64, levels: u64) -> TexDesc {
        TexDesc {
            width: w,
            height: h,
            depth: d,
            mipmap_level_count: levels,
            ..TexDesc::default()
        }
    }

    fn mip(offset: u64, size: u64, bpr: u64, w: u32, h: u32, d: u32) -> TextureMipDimension {
        TextureMipDimension {
            offset,
            size,
            bytes_per_row: bpr,
            width: w,
            height: h,
            depth: d,
        }
    }

    /// A normal (1x1x1-block) dimension of `bb` bytes per pixel, with `mips`.
    fn rec(bps: u64, bb: u8, mips: Vec<TextureMipDimension>) -> TextureDimensionRecord {
        let mut header = TextureDimension {
            array_length: 1,
            bytes_per_slice: bps,
            block_width: 1,
            block_height: 1,
            block_depth: 1,
            block_bytes: bb,
            ..TextureDimension::default()
        };
        header.set_mip_levels(u32::try_from(mips.len()).unwrap());
        TextureDimensionRecord { header, mips }
    }

    fn metal(flags: u64, bb: u64, bw: u64, bh: u64, bd: u64) -> PixelFormatInfo {
        PixelFormatInfo {
            name: None,
            flags,
            cast_class: 0,
            block_bytes: bb,
            block_width: bw,
            block_height: bh,
            block_depth: bd,
        }
    }

    fn script_metal(p: &FakePlatform, info: PixelFormatInfo) {
        p.script()
            .always("MTLPixelFormatGetInfoForDevice()", Step::value(info));
    }

    struct Out {
        ok: bool,
        pvrtc: bool,
        dest: Vec<TextureDimensionRecord>,
        /// The lines the validation logged.
        lines: Vec<String>,
    }

    fn run1(
        src: &TextureDimensionRecord,
        mip_levels: u32,
        generate: bool,
        alloc: u64,
        t: &Tex,
        f: &GuestFeatures,
    ) -> Out {
        let mut d = TextureDimensionRecord::default();
        let mut pvrtc = false;
        let (ok, lines) = logged(|| {
            validate_texture_dimension::<FakePlatform, 1>(
                [src],
                [&mut d],
                mip_levels,
                generate,
                alloc,
                t,
                &mut pvrtc,
                f,
            )
        });
        Out {
            ok,
            pvrtc,
            dest: vec![d],
            lines,
        }
    }

    fn run2(
        src: [&TextureDimensionRecord; 2],
        mip_levels: u32,
        alloc: u64,
        t: &Tex,
        f: &GuestFeatures,
    ) -> Out {
        let mut d0 = TextureDimensionRecord::default();
        let mut d1 = TextureDimensionRecord::default();
        let mut pvrtc = false;
        let (ok, lines) = logged(|| {
            validate_texture_dimension::<FakePlatform, 2>(
                src,
                [&mut d0, &mut d1],
                mip_levels,
                false,
                alloc,
                t,
                &mut pvrtc,
                f,
            )
        });
        Out {
            ok,
            pvrtc,
            dest: vec![d0, d1],
            lines,
        }
    }

    /// The lines of one refused dimension: `msg`'s.
    fn refused(msg: &str) -> Vec<String> {
        vec![format!("texture dimension refused: {msg}")]
    }

    fn feats() -> GuestFeatures {
        GuestFeatures {
            dual_plane_textures: true,
            s8_byte_count_for_d32s8: 1,
            ..GuestFeatures::default()
        }
    }

    /// A 16x8 RGBA8 texture with three tightly packed mips.
    fn good16x8() -> (TexDesc, TextureDimensionRecord) {
        let r = rec(
            0x300,
            4,
            vec![
                mip(0, 0x200, 64, 16, 8, 1),
                mip(0x200, 0x80, 32, 8, 4, 1),
                mip(0x280, 0x20, 16, 4, 2, 1),
            ],
        );
        (desc(16, 8, 1, 3), r)
    }

    /// TextureDimension bit-fields
    #[test]
    fn dimension_bits() {
        let mut h = TextureDimension {
            bits: 0xc000,
            ..TextureDimension::default()
        };
        assert!(h.is_cube() && h.is_compressed());
        h.set_mip_levels(0x1_7fff);
        assert_eq!(h.mip_levels(), 0x3fff);
        assert_eq!({ h.bits }, 0xffff);
        h.set_mip_levels(0x4001);
        assert_eq!(h.mip_levels(), 1);
        assert!(h.is_cube() && h.is_compressed());
        let bytes = TextureDimension {
            bits: 0x0102,
            array_length: 3,
            base_offset: 4,
            bytes_per_slice: 5,
            block_width: 6,
            block_height: 7,
            block_depth: 8,
            block_bytes: 9,
        };
        let b = bytes.as_bytes();
        assert_eq!(b[4], 4); // baseOffset at +4
        assert_eq!(b[0xc], 5); // bytesPerSlice at +0xc
        assert_eq!(&b[0x14..0x18], &[6, 7, 8, 9]);
    }

    /// A multiply-add into a `u32` reports either step's overflow
    #[test]
    fn mul_and_add_into_u32() {
        assert_eq!(
            mul_add_overflow_u32(0xffff_ffff, 1, 0),
            (0xffff_ffff, false)
        );
        assert_eq!(mul_add_overflow_u32(0xffff_ffff, 1, 1), (0, true));
        assert!(mul_add_overflow_u32(0x8000_0000, 2, 0).1);
        assert_eq!(mul_add_overflow_u32(u64::MAX, 0, 7), (7, false));
        assert!(mul_add_overflow_u32(0, 9, 0x1_0000_0000).1);
        assert_eq!(mul_add_overflow_u32(0x10, 6, 0x20), (0x80, false));
    }

    /// Prologue and plane checks
    mod plane {
        use super::*;

        /// A normal texture passes; dest is the source with its mips, and Metal's pixel-format
        /// description is the one ordered call.
        #[test]
        fn success_and_call_order() {
            let p = plat();
            let (d, r) = good16x8();
            let t = tex(&p, d);
            let o = run1(&r, 3, false, 0x300, &t, &feats());
            assert!(o.ok);
            assert!(!o.pvrtc);
            assert_eq!(o.dest[0], r);
            assert!(o.lines.is_empty());
            assert_eq!(
                p.call_log().sels(),
                vec!["MTLPixelFormatGetInfoForDevice()"]
            );
        }

        /// `*isPVRTC` is written before the plane-count check.
        #[test]
        fn pvrtc_then_dual_plane_mismatch() {
            let p = plat();
            script_metal(&p, metal(0x80_2000, 4, 1, 1, 1));
            let (d, r) = good16x8();
            let t = tex(&p, d);
            let o = run1(&r, 3, false, 0x300, &t, &feats());
            assert!(!o.ok);
            assert!(o.pvrtc);
            assert_eq!(o.dest[0], TextureDimensionRecord::default());
            assert_eq!(
                o.lines,
                refused("plane count 1 does not match pixel format 70")
            );
        }

        /// Two planes of a format without bit 23 are refused too, and logged.
        #[test]
        fn two_planes_single_plane_format() {
            let p = FakePlatform::new();
            let (d, r) = good16x8();
            let t = tex(&p, d);
            let o = run2([&r, &r], 3, 0x300, &t, &feats());
            assert!(!o.ok);
            assert!(!o.pvrtc);
            assert_eq!(o.lines.len(), 1);
        }

        /// The span is computed in 32 bits: 0xffffffff passes, one more byte overflows.
        #[test]
        fn allocation_span_u32() {
            let p = plat();
            let t = tex(&p, desc(1, 1, 1, 1));
            let mut r = rec(0xffff_ffff, 4, vec![mip(0, 4, 4, 1, 1, 1)]);
            let o = run1(&r, 1, false, u64::MAX, &t, &feats());
            assert!(o.ok);
            r.header.base_offset = 1;
            let o = run1(&r, 1, false, u64::MAX, &t, &feats());
            assert!(!o.ok);
            // The header was copied (mipLevels set) before the refusal; no mip was.
            assert_eq!(o.dest[0].header, r.header);
            assert!(o.dest[0].mips.is_empty());
            assert_eq!(
                o.lines,
                refused("base 0x1 plus 1 x 0xffffffff bytes overflows 32 bits")
            );
        }

        /// A cube counts six slices per array element (and the u16 array length multiplies in).
        #[test]
        fn cube_six_slices() {
            let p = plat();
            let t = tex(&p, desc(1, 1, 1, 1));
            let mut r = rec(0x3000_0000, 4, vec![mip(0, 4, 4, 1, 1, 1)]);
            r.header.bits |= IS_CUBE;
            let o = run1(&r, 1, false, u64::MAX, &t, &feats());
            assert!(!o.ok);
            assert!(o.lines[0].contains(" plus 6 x "));
            // A cube's six faces are counted: five full strides plus the last face's data.
            let c = tex(
                &p,
                TexDesc {
                    texture_type: TEXTURE_TYPE_CUBE,
                    ..desc(1, 1, 1, 1)
                },
            );
            let mut r = rec(0x10, 4, vec![mip(0, 4, 4, 1, 1, 1)]);
            r.header.bits |= IS_CUBE;
            assert!(run1(&r, 1, false, 0x54, &c, &feats()).ok);
            let o = run1(&r, 1, false, 0x53, &c, &feats());
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("its data ends at 0x54, past the allocation's 0x53 bytes")
            );
        }

        /// The span `baseOffset + bytesPerSlice * sliceCount` must end within the allocation;
        /// equality passes.
        #[test]
        fn larger_than_allocation() {
            let p = plat();
            let (d, mut r) = good16x8();
            r.header.base_offset = 0x10;
            let t = tex(&p, d);
            // The data ends at base 0x10 plus the last mip's end 0x2a0, not the declared slice.
            assert!(run1(&r, 3, false, 0x2b0, &t, &feats()).ok);
            let o = run1(&r, 3, false, 0x2af, &t, &feats());
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("its data ends at 0x2b0, past the allocation's 0x2af bytes")
            );
        }

        /// The texture's 64-bit array length against the u16.
        #[test]
        fn array_length_mismatch() {
            let p = plat();
            let (mut d, r) = good16x8();
            d.array_length = 0x1_0001;
            let t = tex(&p, d);
            let o = run1(&r, 3, false, 0x300, &t, &feats());
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("array length 1, where the texture has 65537")
            );
        }

        /// The mip count must match unless mipmaps are generated; the compared count is the
        /// 14-bit field after `mipLevels` was stored.
        #[test]
        fn mip_count() {
            let p = plat();
            let (d, r) = good16x8();
            let t = tex(
                &p,
                TexDesc {
                    mipmap_level_count: 5,
                    ..d
                },
            );
            let o = run1(&r, 3, false, 0x300, &t, &feats());
            assert!(!o.ok);
            assert_eq!(o.lines, refused("3 mip levels, where the texture has 5"));
            // Type 3 (generateMipmaps) validates level 0 only: its caller passes one level.
            let o = run1(&r, 1, true, 0x300, &t, &feats());
            assert!(o.ok);
            assert_eq!(o.dest[0].header.mip_levels(), 1);
            assert_eq!(o.dest[0].mips.len(), 1);
        }

        /// A 32-bit `mipLevels` over 14 bits: the header keeps its low 14 bits, the loop runs on
        /// the full count (the missing source mip reads as zero here: size mismatch).
        #[test]
        fn mip_levels_width() {
            let p = plat();
            let t = tex(&p, desc(1, 1, 1, 1));
            let r = rec(0, 4, vec![mip(0, 4, 4, 1, 1, 1)]);
            let o = run1(&r, 0x4001, false, 0x10, &t, &feats());
            assert!(!o.ok);
            assert_eq!(o.dest[0].header.mip_levels(), 1);
            assert_eq!(
                o.lines,
                refused("mip 1 is 0x0x0, where the texture's is 1x1x1")
            );
            assert_eq!(o.dest[0].mips.len(), 2);
        }

        /// Cubeness: the dimension's bit must agree with the texture's type.
        #[test]
        fn cubeness() {
            for ty in [5u64, 6, 2] {
                let p = plat();
                let t = tex(
                    &p,
                    TexDesc {
                        texture_type: ty,
                        ..desc(1, 1, 1, 1)
                    },
                );
                let r = rec(0, 4, vec![mip(0, 4, 4, 1, 1, 1)]);
                let o = run1(&r, 1, false, 0x10, &t, &feats());
                assert_eq!(o.ok, ty == 2, "type {ty}");
                if ty != 2 {
                    assert_eq!(
                        o.lines,
                        refused("the dimension's cube flag is false, the texture's true")
                    );
                }
            }
            let p = plat();
            let t = tex(
                &p,
                TexDesc {
                    texture_type: 5,
                    ..desc(1, 1, 1, 1)
                },
            );
            let mut r = rec(0, 4, vec![mip(0, 4, 4, 1, 1, 1)]);
            r.header.bits |= IS_CUBE;
            assert!(run1(&r, 1, false, 0x100, &t, &feats()).ok);
        }
    }

    /// Kinds 0 and 2
    mod normal {
        use super::*;

        fn one(t: &Tex, r: &TextureDimensionRecord) -> Out {
            let n = u32::try_from(r.mips.len()).unwrap();
            run1(r, n, false, u64::MAX, t, &feats())
        }

        #[test]
        fn compressed_flag() {
            let p = plat();
            let (d, mut r) = good16x8();
            r.header.bits |= IS_COMPRESSED;
            let t = tex(&p, d);
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("marked compressed, for an uncompressed pixel format")
            );
        }

        #[test]
        fn non_unity_block() {
            for i in 0..3 {
                let p = plat();
                let (d, mut r) = good16x8();
                let blocks = match i {
                    0 => {
                        r.header.block_width = 2;
                        "2x1x1"
                    }
                    1 => {
                        r.header.block_height = 0;
                        "1x0x1"
                    }
                    _ => {
                        r.header.block_depth = 4;
                        "1x1x4"
                    }
                };
                let t = tex(&p, d);
                let o = one(&t, &r);
                assert!(!o.ok);
                assert_eq!(
                    o.lines,
                    refused(&format!(
                        "blocks of {blocks} texels, for an uncompressed pixel format"
                    ))
                );
            }
        }

        #[test]
        fn pixel_size() {
            let p = plat();
            script_metal(&p, metal(0, 0x108, 1, 1, 1));
            let (d, r) = good16x8();
            let t = tex(&p, d);
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("4 bytes per pixel, where the pixel format's plane has 264")
            );
        }

        /// Width, height and depth each checked against the mip; the failing mip was copied.
        #[test]
        fn size_mismatch() {
            for i in 0..3 {
                let p = plat();
                let (d, mut r) = good16x8();
                let size = match i {
                    0 => {
                        r.mips[1].width = 9;
                        "9x4x1"
                    }
                    1 => {
                        r.mips[1].height = 5;
                        "8x5x1"
                    }
                    _ => {
                        r.mips[1].depth = 2;
                        "8x4x2"
                    }
                };
                let t = tex(&p, d);
                let o = one(&t, &r);
                assert!(!o.ok);
                assert_eq!(
                    o.lines,
                    refused(&format!("mip 1 is {size}, where the texture's is 8x4x1"))
                );
                assert_eq!(o.dest[0].mips, r.mips[..2].to_vec());
            }
        }

        /// The row length is width x blockBytes x sampleCount; the row must be a multiple of
        /// that pixel size.
        #[test]
        fn row_length() {
            let p = plat();
            let (d, mut r) = good16x8();
            r.mips[0].bytes_per_row = 63;
            let t = tex(&p, d);
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("mip 0 has 63 bytes per row, fewer than the 64 its width needs")
            );

            let p = plat();
            let t = tex(
                &p,
                TexDesc {
                    sample_count: 2,
                    ..desc(4, 1, 1, 1)
                },
            );
            let r = rec(0, 4, vec![mip(0, 0x100, 36, 4, 1, 1)]);
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("mip 0 has 36 bytes per row, not a multiple of its 8 bytes per pixel")
            );
            let r = rec(0, 4, vec![mip(0, 0x100, 40, 4, 1, 1)]);
            assert!(one(&t, &r).ok);
        }

        /// The 32-bit pixel size wraps (`blockBytes * (uint32_t)sampleCount`); a zero pixel
        /// size divides as `udiv` does (the remainder is the row length).
        #[test]
        fn pixel_size_width() {
            let p = plat();
            script_metal(&p, metal(0, 0, 1, 1, 1));
            let t = tex(&p, desc(4, 1, 1, 1));
            let r = rec(0, 0, vec![mip(0, 0x100, 5, 4, 1, 1)]);
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("mip 0 has 5 bytes per row, not a multiple of its 0 bytes per pixel")
            );

            let p = plat();
            // sampleCount 0x1_0000_0001 narrows to 1.
            let t = tex(
                &p,
                TexDesc {
                    sample_count: 0x1_0000_0001,
                    ..desc(4, 1, 1, 1)
                },
            );
            let r = rec(0, 4, vec![mip(0, 16, 16, 4, 1, 1)]);
            assert!(one(&t, &r).ok);
        }

        #[test]
        fn total_mip_size_overflow() {
            let p = plat();
            let t = tex(&p, desc(1, 0x1_0000, 0x1_0000, 1));
            let r = rec(
                0,
                4,
                vec![mip(0, u64::MAX, 0x1_0000_0000, 1, 0x1_0000, 0x1_0000)],
            );
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip of 4294967296 rows of 4294967296 bytes overflows")
            );
        }

        #[test]
        fn mip_end_overflow() {
            let p = plat();
            let t = tex(&p, desc(1, 1, 1, 1));
            let r = rec(0, 4, vec![mip(u64::MAX - 3, 4, 4, 1, 1, 1)]);
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip of 0x4 bytes at 0xfffffffffffffffc overflows")
            );
            let r = rec(0, 4, vec![mip(u64::MAX - 4, 4, 4, 1, 1, 1)]);
            assert!(one(&t, &r).ok);
        }

        #[test]
        fn declared_size() {
            let p = plat();
            let (d, mut r) = good16x8();
            r.mips[2].size = 0x1f;
            let t = tex(&p, d);
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip needs 0x20 bytes, more than its declared 0x1f")
            );
        }

        /// The mip must end within the slice, unless the slice size is 0.
        #[test]
        fn slice_size() {
            let p = plat();
            let (d, mut r) = good16x8();
            r.header.bytes_per_slice = 0x29f;
            let t = tex(&p, d);
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip ends at 0x2a0, past the slice's 0x29f bytes")
            );
            r.header.bytes_per_slice = 0;
            assert!(one(&t, &r).ok);
        }

        /// Mip ends must increase, from 0: a mip 0 ending at 0 (a depth-0 texture) fails, as does a
        /// mip ending where the previous did.
        #[test]
        fn non_increasing() {
            let p = plat();
            let t = tex(&p, desc(1, 1, 0, 1));
            let r = rec(0, 4, vec![mip(0, 0, 4, 1, 1, 0)]);
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip ends at 0x0, not past the previous one's 0x0")
            );

            let p = plat();
            let (d, mut r) = good16x8();
            r.mips[2].offset = 0x260;
            let t = tex(&p, d);
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip ends at 0x280, not past the previous one's 0x280")
            );
        }

        /// Each halving clamps to 1 (5 -> 2 -> 1 -> 1).
        #[test]
        fn halving_clamps() {
            let p = plat();
            let t = tex(&p, desc(5, 1, 3, 4));
            let r = rec(
                0,
                4,
                vec![
                    mip(0, 60, 20, 5, 1, 3),
                    mip(60, 8, 8, 2, 1, 1),
                    mip(68, 4, 4, 1, 1, 1),
                    mip(72, 4, 4, 1, 1, 1),
                ],
            );
            assert!(one(&t, &r).ok);
        }

        /// Kind 2 (the two-plane formats described here): each plane against its own plane info. D32S8 as two
        /// planes has a 1-byte stencil plane.
        #[test]
        fn two_plane_kinds() {
            let p = plat();
            let t = tex(
                &p,
                TexDesc {
                    pixel_format: 260,
                    ..desc(2, 1, 1, 1)
                },
            );
            let depth = rec(0, 4, vec![mip(0, 8, 8, 2, 1, 1)]);
            let stencil = rec(0, 1, vec![mip(0, 2, 2, 2, 1, 1)]);
            let o = run2([&depth, &stencil], 1, 0x100, &t, &feats());
            assert!(o.ok);
            assert_eq!(o.dest, vec![depth.clone(), stencil.clone()]);

            // Plane 1 with plane 0's pixel size fails on plane 1's info, after plane 0 passed.
            let p = plat();
            let t = tex(
                &p,
                TexDesc {
                    pixel_format: 260,
                    ..desc(2, 1, 1, 1)
                },
            );
            let o = run2([&depth, &depth], 1, 0x100, &t, &feats());
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("4 bytes per pixel, where the pixel format's plane has 1")
            );
            assert_eq!(o.dest[0], depth);
            assert_eq!(o.dest[1].header, depth.header);

            // 550 on an XR10 device: 4-byte planes, and one plane is a mismatch.
            let p = plat();
            let t = tex(
                &p,
                TexDesc {
                    pixel_format: 550,
                    ..desc(2, 1, 1, 1)
                },
            );
            let x = rec(0, 4, vec![mip(0, 8, 8, 2, 1, 1)]);
            assert!(run2([&x, &x], 1, 0x100, &t, &feats()).ok);
            let o = run1(&x, 1, false, 0x100, &t, &feats());
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("plane count 1 does not match pixel format 550")
            );
        }
    }

    mod compressed {
        use super::*;

        /// A 4x4-block, 16-byte format (BC7-like), 10x6 with two mips.
        fn bc(p: &FakePlatform, flags: u64) -> (Tex, TextureDimensionRecord) {
            script_metal(p, metal(flags, 16, 4, 4, 1));
            let t = tex(p, desc(10, 6, 1, 2));
            let mut r = rec(
                0x100,
                16,
                vec![mip(0, 0x60, 0x30, 10, 6, 1), mip(0x60, 0x20, 0x20, 5, 3, 1)],
            );
            r.header.bits |= IS_COMPRESSED;
            r.header.block_width = 4;
            r.header.block_height = 4;
            (t, r)
        }

        fn one(t: &Tex, r: &TextureDimensionRecord) -> Out {
            run1(r, 2, false, u64::MAX, t, &feats())
        }

        /// Blocks round up.
        #[test]
        fn success() {
            let p = plat();
            let (t, r) = bc(&p, 0x400);
            let o = one(&t, &r);
            assert!(o.ok);
            assert_eq!(o.dest[0], r);
            assert!(o.lines.is_empty());
        }

        #[test]
        fn not_compressed() {
            let p = plat();
            let (t, mut r) = bc(&p, 0x400);
            r.header.bits &= !IS_COMPRESSED;
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("marked uncompressed, for a compressed pixel format")
            );
        }

        /// Each of the four block fields is compared with Metal's 64-bit word.
        #[test]
        fn block_mismatch() {
            for i in 0..4 {
                let p = plat();
                let (t, mut r) = bc(&p, 0x400);
                let blocks = match i {
                    0 => {
                        r.header.block_width = 5;
                        "5x4x1 texels and 16"
                    }
                    1 => {
                        r.header.block_height = 1;
                        "4x1x1 texels and 16"
                    }
                    2 => {
                        r.header.block_depth = 2;
                        "4x4x2 texels and 16"
                    }
                    _ => {
                        r.header.block_bytes = 8;
                        "4x4x1 texels and 8"
                    }
                };
                let o = one(&t, &r);
                assert!(!o.ok);
                assert_eq!(
                    o.lines,
                    refused(&format!(
                        "blocks of {blocks} bytes, where the pixel format's are 4x4x1 and 16"
                    ))
                );
            }
            // A Metal word over 8 bits never matches the byte.
            let p = plat();
            script_metal(&p, metal(0x400, 0x110, 4, 4, 1));
            let t = tex(&p, desc(10, 6, 1, 2));
            let (_, r) = bc(&plat(), 0x400);
            assert!(!one(&t, &r).ok);
        }

        #[test]
        fn row_checks() {
            let p = plat();
            let (t, mut r) = bc(&p, 0x400);
            r.mips[0].bytes_per_row = 0x2f;
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("mip 0 has 47 bytes per row, fewer than the 48 its width needs")
            );
            let p = plat();
            let (t, mut r) = bc(&p, 0x400);
            r.mips[0].bytes_per_row = 0x38;
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("mip 0 has 56 bytes per row, not a multiple of its 16 bytes per block")
            );
        }

        /// The shared tail: rows are blocksDeep x blocksHigh; sizes, ends and order as for
        /// normal textures.
        #[test]
        fn extent_checks() {
            let p = plat();
            let (t, mut r) = bc(&p, 0x400);
            r.mips[0].size = 0x5f;
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip needs 0x60 bytes, more than its declared 0x5f")
            );
            let p = plat();
            let (t, mut r) = bc(&p, 0x400);
            r.mips[1].offset = 0x40;
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip ends at 0x60, not past the previous one's 0x60")
            );
            let p = plat();
            let (t, mut r) = bc(&p, 0x400);
            r.mips[1].width = 4;
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("mip 1 is 4x3x1, where the texture's is 5x3x1")
            );
            let p = plat();
            let (t, mut r) = bc(&p, 0x400);
            r.header.bytes_per_slice = 0x7f;
            let o = one(&t, &r);
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip ends at 0x80, past the slice's 0x7f bytes")
            );
        }

        /// The two overflow checks refuse a compressed texture as they refuse a normal one.
        #[test]
        fn extent_overflows() {
            let compressed = |r: &mut TextureDimensionRecord| {
                r.header.bits |= IS_COMPRESSED;
                r.header.block_width = 4;
                r.header.block_height = 4;
            };
            // 0x10000 x 0x10000 rows of 4x4x1 blocks, times a 4 GiB row: 2^64 overflows.
            let p = plat();
            script_metal(&p, metal(0x400, 16, 4, 4, 1));
            let t = tex(&p, desc(4, 0x4_0000, 0x1_0000, 1));
            let mut r = rec(
                0,
                16,
                vec![mip(0, u64::MAX, 0x1_0000_0000, 4, 0x4_0000, 0x1_0000)],
            );
            compressed(&mut r);
            let o = run1(&r, 1, false, u64::MAX, &t, &feats());
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip of 4294967296 rows of 4294967296 bytes overflows")
            );
            // One 16-byte block starting 8 bytes before the end of the address space.
            let p = plat();
            script_metal(&p, metal(0x400, 16, 4, 4, 1));
            let t = tex(&p, desc(4, 4, 1, 1));
            let mut r = rec(0, 16, vec![mip(u64::MAX - 7, 16, 16, 4, 4, 1)]);
            compressed(&mut r);
            let o = run1(&r, 1, false, u64::MAX, &t, &feats());
            assert!(!o.ok);
            assert_eq!(
                o.lines,
                refused("a mip of 0x10 bytes at 0xfffffffffffffff8 overflows")
            );
        }

        /// A two-plane compressed format checks plane 1 against plane 0's info (plane 1's own is
        /// zero for a Metal format), so it passes.
        #[test]
        fn plane_one_uses_plane_zero_info() {
            let p = plat();
            let (t, r) = bc(&p, 0x80_0400);
            let o = run2([&r, &r], 2, u64::MAX, &t, &feats());
            assert!(o.ok);
            assert_eq!(o.dest, vec![r.clone(), r]);
        }
    }
}
