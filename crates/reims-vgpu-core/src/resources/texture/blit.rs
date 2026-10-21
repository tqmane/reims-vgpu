//! The blit-paged layouts: one plane ([`SinglePlane`], object types 2 and 3) and the two halves
//! of a depth-stencil format ([`DualPlaneBlit`], type 12), and the copy loop that pages them and
//! the shared backings through a blit encoder ([`page`]).

use std::sync::Arc;

use crate::nil_send;
use crate::num::{self, Truncate};
use crate::platform::Platform;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::resource::Texture;
use crate::platform::types::{MtlOrigin, MtlSize};
use crate::raise::{Kept, keep_forever};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::resource::{Resource, ResourceState, TaskRef, features_of, platform_for};
use crate::resources::texture::dimension::{
    TextureDimensionRecord, TexturePagingInfo, validate_texture_dimension,
};
use crate::resources::texture::{
    DIMENSION_SIZE, DeathKind, MIP_DIMENSION_SIZE, PagedTexture, RaiseKind, TextureLayout,
    into_resource, mip_at, raise, record_at,
};
use crate::rt::Hold;

/// `MTLBlitOptionRowLinearPVRTC`: a PVRTC texture's copies.
pub(crate) const BLIT_OPTION_ROW_LINEAR_PVRTC: u64 = 4;

/// `MTLBlitOptionDepthFromDepthStencil` and `MTLBlitOptionStencilFromDepthStencil`: plane 0's and
/// plane 1's copies.
pub(crate) const DEPTH_STENCIL_BLIT_OPTIONS: [u64; 2] = [1, 2];

/// One plane, paged by blits.
#[derive(Debug, Clone, Default)]
pub struct SinglePlane {
    /// The validated dimension, its mips included.
    pub(crate) dimension: TextureDimensionRecord,
    /// Type 3 with more than one level: level 0 is paged up, and the rest generated from it.
    pub(crate) generate_mipmaps: bool,
    /// [`BLIT_OPTION_ROW_LINEAR_PVRTC`] for a PVRTC texture, else 0.
    pub(crate) blit_option: u64,
}

/// Two planes, a depth-stencil format's halves, paged by blits plane by plane.
#[derive(Debug, Clone, Default)]
pub struct DualPlaneBlit {
    /// The two planes' validated dimensions.
    pub(crate) planes: [TextureDimensionRecord; 2],
}

/// The blit copies of one direction for `planes` (each with its blit option), in the order
/// plane, slice, face, level. The cube bit and the array length are plane 0's for every plane.
/// A nil encoder makes every copy a send to nil. An upload row shorter than its mip's width
/// dies (validation makes it unreachable).
pub(crate) fn page<P: Platform>(
    paged: &PagedTexture<P>,
    encoder: Option<&P::BlitEncoder>,
    is_download: bool,
    planes: &[(&TextureDimensionRecord, u64)],
) -> RaiseResult<()> {
    let transfer = paged.transfer_buffer.lock();
    let Some(&(first, _)) = planes.first() else {
        return Ok(());
    };
    let array_length = u32::from(first.header.array_length);
    let is_cube = first.header.is_cube();
    let faces: u8 = if is_cube { 6 } else { 1 };
    for &(dimension, options) in planes {
        let header = dimension.header;
        for slice in 0..array_length {
            for face in 0..faces {
                // 32 bits, wrapping.
                let texture_slice = if is_cube {
                    slice.wrapping_mul(6).wrapping_add(u32::from(face))
                } else {
                    slice
                };
                for level in 0..u64::from(header.mip_levels()) {
                    let mip = mip_at(dimension, level);
                    // 64 bits, wrapping.
                    let offset = header
                        .bytes_per_slice
                        .wrapping_mul(u64::from(texture_slice))
                        .wrapping_add(header.base_offset)
                        .wrapping_add(mip.offset);
                    let bytes_per_row = mip.bytes_per_row;
                    // Rows in whole blocks, 32 bits.
                    let block_height = u32::from(header.block_height);
                    let rows = num::div0(
                        mip.height.wrapping_add(block_height).wrapping_sub(1),
                        block_height,
                    );
                    let bytes_per_image = bytes_per_row.wrapping_mul(u64::from(rows));
                    let size = MtlSize::new(
                        u64::from(mip.width),
                        u64::from(mip.height),
                        u64::from(mip.depth),
                    );
                    if is_download {
                        nil_send!(
                            encoder,
                            encode(
                                OnFault::Return,
                                BlitCmd::CopyTextureToBufferOptions {
                                    source_texture: Some(&paged.texture),
                                    source_slice: u64::from(texture_slice),
                                    source_level: level,
                                    source_origin: MtlOrigin::new(0, 0, 0),
                                    source_size: size,
                                    destination_buffer: transfer.as_ref(),
                                    destination_offset: offset,
                                    destination_bytes_per_row: bytes_per_row,
                                    destination_bytes_per_image: bytes_per_image,
                                    options
                                }
                            )
                        )?;
                    } else {
                        // The upload assert, kept though validation makes it unreachable.
                        crate::fatal_unless!(
                            bytes_per_row >= u64::from(mip.width),
                            DeathKind::ShortUploadRow
                        );
                        nil_send!(
                            encoder,
                            encode(
                                OnFault::Return,
                                BlitCmd::CopyBufferToTextureOptions {
                                    source_buffer: transfer.as_ref(),
                                    source_offset: offset,
                                    source_bytes_per_row: bytes_per_row,
                                    source_bytes_per_image: bytes_per_image,
                                    source_size: size,
                                    destination_texture: Some(&paged.texture),
                                    destination_slice: u64::from(texture_slice),
                                    destination_level: level,
                                    destination_origin: MtlOrigin::new(0, 0, 0),
                                    options
                                }
                            )
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

impl SinglePlane {
    /// A one-plane texture of object type 2, or 3 (level 0 only, the rest generated). `None`
    /// when validation refuses the record, or with no platform scope. `texture` is the caller's
    /// reference (see [`create_texture_resource`](super::create_texture_resource)).
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub(crate) fn new<P: Platform>(
        task: TaskRef<P>,
        object_type: u32,
        paging_info: &TexturePagingInfo,
        dimension: &TextureDimensionRecord,
        dimension_length: u64,
        texture: P::Texture,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let Some(platform) = platform_for(&task) else {
            drop(texture);
            return Ok(None);
        };
        let mut paged = PagedTexture::new(&platform, task, &texture);
        let mut generate_mipmaps = object_type == 3;
        // 16 bits, as the record's field is.
        let mut mip_levels: u16 = dimension.header.mip_levels();
        if generate_mipmaps {
            // Type 3 validates and pages level 0 only.
            mip_levels = 1;
        }
        if dimension_length < DIMENSION_SIZE {
            // The half-made texture is released, and with it its reference, then the caller's.
            // Unreachable: the manager passes at least a header.
            drop(paged);
            drop(texture);
            return raise(RaiseKind::DimensionLength);
        }
        let mut dest = TextureDimensionRecord::default();
        paged.page = paging_info.page;
        paged.length = paging_info.length;
        let features = features_of(&paged.task);
        let valid = validate_texture_dimension::<P, 1>(
            [dimension],
            [&mut dest],
            u32::from(mip_levels),
            generate_mipmaps,
            paging_info.length,
            &texture,
            &mut paged.is_pvrtc,
            &features,
        );
        if !valid {
            drop(paged);
            drop(texture);
            return Ok(None);
        }
        let blit_option = if paged.is_pvrtc {
            BLIT_OPTION_ROW_LINEAR_PVRTC
        } else {
            0
        };
        if generate_mipmaps && paged.texture.mipmap_level_count() <= 1 {
            generate_mipmaps = false;
        }
        drop(texture);
        Ok(Some(into_resource(
            paged,
            TextureLayout::SinglePlane(SinglePlane {
                dimension: dest,
                generate_mipmaps,
                blit_option,
            }),
        )))
    }

    /// Pages every level in one direction.
    pub(crate) fn page<P: Platform>(
        &self,
        paged: &PagedTexture<P>,
        encoder: Option<&P::BlitEncoder>,
        is_download: bool,
    ) -> RaiseResult<()> {
        page(
            paged,
            encoder,
            is_download,
            &[(&self.dimension, self.blit_option)],
        )
    }

    /// Guest to host; a type-3 texture of more than one level then generates its mipmaps.
    pub(crate) fn prepare<P: Platform>(
        &self,
        hold: Hold<'_>,
        paged: &PagedTexture<P>,
        state: &ResourceState,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        paged.prepare_by_blits(
            hold,
            state,
            encoder,
            || {
                self.page(paged, encoder, false)?;
                if self.generate_mipmaps {
                    nil_send!(
                        encoder,
                        encode(
                            OnFault::Return,
                            BlitCmd::GenerateMipmaps {
                                texture: Some(&paged.texture)
                            }
                        )
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Host to guest.
    pub(crate) fn synchronize<P: Platform>(
        &self,
        hold: Hold<'_>,
        paged: &PagedTexture<P>,
        state: &ResourceState,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        paged.synchronize_by_blits(
            hold,
            state,
            encoder,
            || self.page(paged, encoder, true),
        )
    }
}

impl DualPlaneBlit {
    /// A two-plane texture of object type 12 in a format the blits page. `None` when validation
    /// refuses the record, or with no platform scope. `texture` is the caller's reference (see
    /// [`create_texture_resource`](super::create_texture_resource)).
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub(crate) fn new<P: Platform>(
        task: TaskRef<P>,
        paging_info: &TexturePagingInfo,
        dimension: &TextureDimensionRecord,
        dimension_length: u64,
        texture: P::Texture,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let Some(platform) = platform_for(&task) else {
            drop(texture);
            return Ok(None);
        };
        let mut paged = PagedTexture::new(&platform, task, &texture);
        let mip_levels = u64::from(dimension.header.mip_levels());
        // The task's features are read here only when mipped.
        if mip_levels != 1 && !features_of(&paged.task).supports_multiplane_mipped_blit_textures {
            drop(paged);
            // The caller's texture is kept for good: its resource index and GPU VA stay taken,
            // and later placements are numbered after them.
            keep_forever(texture, Kept::MetalNumbering);
            return raise(RaiseKind::MippedMultiPlane);
        }
        if dimension_length < 2_u64.wrapping_mul(DIMENSION_SIZE) {
            // The half-made texture is released, then the caller's texture. Unreachable: the
            // manager sizes the copy.
            drop(paged);
            drop(texture);
            return raise(RaiseKind::DimensionLength);
        }
        // Plane 1 starts after plane 0's mips, and takes plane 0's count.
        let plane1 = record_at(
            dimension,
            DIMENSION_SIZE.wrapping_add(mip_levels.wrapping_mul(MIP_DIMENSION_SIZE)),
            mip_levels,
        );
        let mut dest = [
            TextureDimensionRecord::default(),
            TextureDimensionRecord::default(),
        ];
        paged.page = paging_info.page;
        paged.length = paging_info.length;
        let features = features_of(&paged.task);
        let [d0, d1] = &mut dest;
        let valid = validate_texture_dimension::<P, 2>(
            [dimension, &plane1],
            [d0, d1],
            mip_levels.low_u32(),
            false,
            paging_info.length,
            &texture,
            &mut paged.is_pvrtc,
            &features,
        );
        if !valid {
            drop(paged);
            drop(texture);
            return Ok(None);
        }
        // Unreachable once validated: the two-plane kinds refuse compressed planes.
        if dest[0].header.is_compressed() {
            drop(paged);
            // The caller's texture is released.
            drop(texture);
            return raise(RaiseKind::CompressedMultiPlane);
        }
        drop(texture);
        Ok(Some(into_resource(
            paged,
            TextureLayout::DualPlaneBlit(DualPlaneBlit { planes: dest }),
        )))
    }

    /// Pages plane 0 entirely, then plane 1, each with its blit option.
    pub(crate) fn page<P: Platform>(
        &self,
        paged: &PagedTexture<P>,
        encoder: Option<&P::BlitEncoder>,
        is_download: bool,
    ) -> RaiseResult<()> {
        let [p0, p1] = &self.planes;
        let [o0, o1] = DEPTH_STENCIL_BLIT_OPTIONS;
        page(paged, encoder, is_download, &[(p0, o0), (p1, o1)])
    }

    /// Guest to host.
    pub(crate) fn prepare<P: Platform>(
        &self,
        hold: Hold<'_>,
        paged: &PagedTexture<P>,
        state: &ResourceState,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        paged.prepare_by_blits(
            hold,
            state,
            encoder,
            || self.page(paged, encoder, false),
        )
    }

    /// Host to guest.
    pub(crate) fn synchronize<P: Platform>(
        &self,
        hold: Hold<'_>,
        paged: &PagedTexture<P>,
        state: &ResourceState,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        paged.synchronize_by_blits(
            hold,
            state,
            encoder,
            || self.page(paged, encoder, true),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::texture::test_support::*;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{FakePlatform, Step};

    type P = FakePlatform;

    mod page {
        use super::*;

        /// A paged texture over `t`, at page 3 with 0x400 bytes.
        fn base(e: &Env, t: &Tex) -> PagedTexture<P> {
            let mut b = PagedTexture::new(&e.p, e.task(), t);
            b.page = 3;
            b.length = 0x400;
            b
        }

        /// Offsets wrap in 64 bits, rows and images in 32
        #[test]
        fn widths_wrap() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            b.ensure_physical(Hold::NONE).unwrap();
            let mut h = header(u64::MAX, 4, 1);
            h.base_offset = 2;
            h.block_height = 0; // arm64 udiv by 0 gives 0
            let d = rec1(h, vec![mip(1, 0, u64::MAX, 1, 7, 1)]);
            let enc = e.blit();
            let m = e.p.call_log().mark();
            page(&b, Some(&enc), true, &[(&d, 9)]).unwrap();
            let c = &e.calls(m, "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:")[0];
            // offset = MAX*0 + 2 + 1
            assert_eq!(u(c, 6), 3);
            // rows 0 -> image 0
            assert_eq!(u(c, 8), 0);
            assert_eq!(u(c, 9), 9);
        }

        /// A nil encoder sends nothing, yet the upload assert still runs: a short row dies
        #[test]
        fn nil_encoder_sends_nothing_but_asserts() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            b.ensure_physical(Hold::NONE).unwrap();
            let d = rec1(header(0x40, 4, 1), vec![mip(0, 0x40, 16, 4, 4, 1)]);
            let m = e.p.call_log().mark();
            page::<P>(&b, None, false, &[(&d, 0)]).unwrap();
            assert!(e.sels(m).is_empty());
            let short = rec1(header(0x40, 4, 1), vec![mip(0, 0x40, 3, 4, 4, 1)]);
            let got = catch_death(|| page::<P>(&b, None, false, &[(&short, 0)]));
            assert_eq!(got.err(), Some(died(DeathKind::ShortUploadRow)));
            // A download never asserts.
            page::<P>(&b, None, true, &[(&short, 0)]).unwrap();
        }

        /// A copy's raise stops the loop
        #[test]
        fn raise_stops() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let b = base(&e, &t);
            b.ensure_physical(Hold::NONE).unwrap();
            let mut h = header(0x40, 4, 1);
            h.array_length = 3;
            let d = rec1(h, vec![mip(0, 0x40, 16, 4, 4, 1)]);
            let enc = e.blit();
            let sel = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:";
            e.p.script()
                .after(sel, 1, Step::raise("NSGenericException", "x"));
            let m = e.p.call_log().mark();
            assert!(page(&b, Some(&enc), false, &[(&d, 0)]).is_err());
            assert_eq!(e.calls(m, sel).len(), 2);
        }
    }

    /// One plane.
    mod single {

        use super::*;
        use crate::platform::Handle;
        use crate::platform::types::{MtlOrigin, MtlSize, PixelFormatInfo};
        use crate::resources::resource::ResourceKind;
        use crate::resources::texture::{TextureResource, create_texture_resource};
        use crate::testkit::capture::catch_death;
        use crate::testkit::{Arg, FakePlatform, Step};

        type P = FakePlatform;

        const UP: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:";
        const DOWN: &str = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:";

        /// A 16x8 RGBA8 texture with three tightly packed mips.
        fn good16x8() -> TextureDimensionRecord {
            rec1(
                header(0x300, 4, 3),
                vec![
                    mip(0, 0x200, 64, 16, 8, 1),
                    mip(0x200, 0x80, 32, 8, 4, 1),
                    mip(0x280, 0x20, 16, 4, 2, 1),
                ],
            )
        }

        fn make(
            e: &Env,
            ty: u32,
            t: Tex,
            r: &TextureDimensionRecord,
            len: u64,
        ) -> RaiseResult<Option<Arc<Resource<P>>>> {
            e.run(|| SinglePlane::new(e.task(), ty, &paging(7, 0x300), r, len, t))
        }

        fn parts(r: &Resource<P>) -> (&TextureResource<P>, &SinglePlane) {
            match &r.kind {
                ResourceKind::Texture(t) => match &t.layout {
                    TextureLayout::SinglePlane(s) => (t, s),
                    _ => panic!("not single-plane"),
                },
                _ => panic!("not a texture"),
            }
        }

        mod new {
            use super::*;

            #[test]
            fn type_two_validates_every_level() {
                let e = Env::new();
                let t = e.tex(desc(70, 16, 8, 1, 3), false);
                let m = e.p.call_log().mark();
                let r = make(&e, 2, t.clone_ref(), &good16x8(), 0x18 + 3 * 0x24)
                    .unwrap()
                    .unwrap();
                let (tr, s) = parts(&r);
                assert_eq!(s.dimension, good16x8());
                assert!(!s.generate_mipmaps);
                assert_eq!(s.blit_option, 0);
                assert_eq!(tr.paged.page, 7);
                assert_eq!(tr.paged.length, 0x300);
                assert!(!tr.paged.is_pvrtc);
                assert!(r.state.guest_valid());
                assert!(!r.state.host_valid());
                // `[task features]` before validation's pixel format info.
                let sels = e.sels(m);
                assert_eq!(sels[..2], ["features", "MTLPixelFormatGetInfoForDevice()"]);
                assert_eq!(t.retain_count(), 2);
                drop(r);
                assert_eq!(t.retain_count(), 1);
            }

            /// Type 3 keeps level 0 and generates
            #[test]
            fn type_three_level_zero_and_generate() {
                let e = Env::new();
                // The texture has 3 levels, the record says 3: only level 0 is validated and kept.
                let t = e.tex(desc(70, 16, 8, 1, 3), false);
                let r = make(&e, 3, t, &good16x8(), 0x18 + 3 * 0x24)
                    .unwrap()
                    .unwrap();
                let (_, s) = parts(&r);
                assert!(s.generate_mipmaps);
                assert_eq!(s.dimension.header.mip_levels(), 1);
                assert_eq!(s.dimension.mips.len(), 1);
            }

            /// One level, no generation
            #[test]
            fn type_three_single_level_texture_does_not_generate() {
                let e = Env::new();
                let t = e.tex(desc(70, 16, 8, 1, 1), false);
                let r = make(&e, 3, t, &good16x8(), 0x18 + 3 * 0x24)
                    .unwrap()
                    .unwrap();
                assert!(!parts(&r).1.generate_mipmaps);
            }

            #[test]
            fn pvrtc_sets_blit_option() {
                let e = Env::new();
                e.p.script().always(
                    "MTLPixelFormatGetInfoForDevice()",
                    Step::value(PixelFormatInfo {
                        name: None,
                        flags: 0x2000,
                        cast_class: 0,
                        block_bytes: 4,
                        block_width: 1,
                        block_height: 1,
                        block_depth: 1,
                    }),
                );
                let t = e.tex(desc(70, 16, 8, 1, 3), false);
                let r = make(&e, 2, t, &good16x8(), 0x18 + 3 * 0x24)
                    .unwrap()
                    .unwrap();
                let (tr, s) = parts(&r);
                assert!(tr.paged.is_pvrtc);
                assert_eq!(s.blit_option, 4);
            }

            /// Validation refused: nil, both references released
            #[test]
            fn invalid_is_nil() {
                let e = Env::new();
                // Type 2 with a record of 0 mips fails the mip count's check.
                let t = e.tex(desc(70, 16, 8, 1, 3), false);
                let (r, lines) = logged(|| {
                    make(
                        &e,
                        2,
                        t.clone_ref(),
                        &rec1(header(0x300, 4, 0), vec![]),
                        0x18,
                    )
                });
                assert!(r.unwrap().is_none());
                assert_eq!(t.retain_count(), 1);
                assert_eq!(
                    lines,
                    vec!["texture dimension refused: 0 mip levels, where the texture has 3"]
                );
            }

            /// Short descriptor: raise, the half-made resource and the caller's texture released
            #[test]
            fn short_length_raises_and_releases() {
                let e = Env::new();
                let t = e.tex(desc(70, 16, 8, 1, 3), false);
                let m = e.p.call_log().mark();
                let err = make(&e, 2, t.clone_ref(), &good16x8(), 0x17).err().unwrap();
                assert_eq!(raised(&err), Some(RaiseKind::DimensionLength));
                assert!(e.sels(m).is_empty());
                // Ours alone: the half-made resource's and the caller's are released.
                assert_eq!(t.retain_count(), 1);
            }

            /// Dead task at `[task features]`
            #[test]
            fn dead_task_dies() {
                let e = Env::new();
                let t = e.tex(desc(70, 16, 8, 1, 3), false);
                let dead = {
                    let d: Arc<dyn crate::resources::host::ResourceHost<P>> =
                        crate::testkit::resource_host::FakeResourceHost::new(e.p.clone());
                    Arc::downgrade(&d)
                };
                let got = catch_death(|| {
                    e.run(|| SinglePlane::new(dead, 2, &paging(0, 0x300), &good16x8(), 0x80, t))
                });
                assert_eq!(
                    got.err(),
                    Some(crate::resources::DeathKind::GoneTaskFeatures.site())
                );
            }
        }

        fn made(e: &Env, ty: u32, levels: u64) -> Arc<Resource<P>> {
            let t = e.tex(desc(70, 16, 8, 1, levels), false);
            e.run(|| create_texture_resource(e.task(), ty, &paging(7, 0x300), &good16x8(), 0x84, t))
                .unwrap()
                .unwrap()
        }

        mod paging {
            use super::*;

            #[test]
            fn upload_levels() {
                let e = Env::new();
                let r = made(&e, 2, 3);
                let (tr, s) = parts(&r);
                tr.paged.ensure_physical(Hold::NONE).unwrap();
                let buf = tr.paged.transfer_buffer.lock().as_ref().unwrap().id();
                let enc = e.blit();
                let m = e.p.call_log().mark();
                s.page(&tr.paged, Some(&enc), false).unwrap();
                let c = e.calls(m, UP);
                assert_eq!(c.len(), 3);
                assert_eq!(
                    c[1].args,
                    vec![
                        Arg::Obj(Some(buf)),
                        Arg::U64(0x200),
                        Arg::U64(32),
                        Arg::U64(32 * 4),
                        Arg::val(&MtlSize::new(8, 4, 1)),
                        Arg::Obj(Some(tr.paged.texture.id())),
                        Arg::U64(0),
                        Arg::U64(1),
                        Arg::val(&MtlOrigin::new(0, 0, 0)),
                        Arg::U64(0),
                    ]
                );
                assert_eq!(e.sels(m).len(), 3);
            }

            /// Download of a cube array: slice-major, six faces, level innermost
            #[test]
            fn download_cube_array_order() {
                let e = Env::new();
                let r = made(&e, 2, 3);
                let (tr, _) = parts(&r);
                tr.paged.ensure_physical(Hold::NONE).unwrap();
                let mut h = header(0x1000, 4, 2);
                h.bits |= 0x4000; // isCube
                h.array_length = 2;
                h.base_offset = 0x10;
                let s = SinglePlane {
                    dimension: rec1(h, vec![mip(0, 0, 8, 2, 3, 1), mip(0x40, 0, 4, 1, 1, 1)]),
                    generate_mipmaps: false,
                    blit_option: 4,
                };
                let enc = e.blit();
                let m = e.p.call_log().mark();
                s.page(&tr.paged, Some(&enc), true).unwrap();
                let c = e.calls(m, DOWN);
                assert_eq!(c.len(), 2 * 6 * 2);
                for (i, call) in c.iter().enumerate() {
                    let (slice, face, level) = (i / 12, (i / 2) % 6, i % 2);
                    let ts = (slice * 6 + face) as u64;
                    assert_eq!(u(call, 1), ts);
                    assert_eq!(u(call, 2), level as u64);
                    let off = if level == 0 { 0 } else { 0x40 };
                    assert_eq!(u(call, 6), 0x1000 * ts + 0x10 + off);
                    assert_eq!(u(call, 9), 4);
                }
                // rows = height / blockHeight(1): 3 rows of 8.
                assert_eq!(u(&c[0], 8), 24);
            }

            /// The upload assert
            #[test]
            fn upload_assert_dies() {
                let e = Env::new();
                let r = made(&e, 2, 3);
                let (tr, _) = parts(&r);
                tr.paged.ensure_physical(Hold::NONE).unwrap();
                let s = SinglePlane {
                    dimension: rec1(header(0, 4, 1), vec![mip(0, 0, 3, 4, 1, 1)]),
                    generate_mipmaps: false,
                    blit_option: 0,
                };
                let got = catch_death(|| s.page::<P>(&tr.paged, None, false));
                assert_eq!(got.err(), Some(died(DeathKind::ShortUploadRow)));
                // Download never asserts.
                s.page::<P>(&tr.paged, None, true).unwrap();
            }
        }

        mod prepare {
            use super::*;

            /// Page up, then host-valid
            #[test]
            fn pages_and_validates() {
                let e = Env::new();
                let r = made(&e, 2, 3);
                let enc = e.blit();
                let m = e.p.call_log().mark();
                r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
                assert_eq!(e.sels(m), vec!["new_buffer_for_virtual_page", UP, UP, UP]);
                assert!(r.state.host_valid());
                // Host-valid: nothing more.
                let m = e.p.call_log().mark();
                r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
                assert!(e.sels(m).is_empty());
            }

            /// Generates after level 0
            #[test]
            fn type_three_generates() {
                let e = Env::new();
                let r = made(&e, 3, 3);
                let enc = e.blit();
                let m = e.p.call_log().mark();
                r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
                assert_eq!(
                    e.sels(m),
                    vec![
                        "new_buffer_for_virtual_page",
                        UP,
                        "generateMipmapsForTexture:"
                    ]
                );
            }

            /// Guest-invalid logs; protected: never paged, still host-valid
            #[test]
            fn guest_invalid_logs_protected_not_paged() {
                let e = Env::new();
                let r = made(&e, 3, 3);
                r.state.set_guest_valid(false);
                parts(&r).0.paged.texture.set_protection_options(1);
                let m = e.p.call_log().mark();
                let (prepared, lines) = logged(|| r.prepare_in_encoder(Hold::NONE, None));
                prepared.unwrap();
                assert!(e.sels(m).is_empty());
                assert_eq!(lines, vec![PREPARE_INVALID]);
                assert!(r.state.host_valid());
            }

            /// Devolatilize first may re-page
            #[test]
            fn devolatilize_first() {
                let e = Env::new();
                let r = made(&e, 2, 3);
                r.state.set_host_valid(true);
                r.state.set_is_volatile(true);
                e.p.script().push("setPurgeableState:", Step::value(4u64));
                let m = e.p.call_log().mark();
                r.prepare_in_encoder(Hold::NONE, None).unwrap();
                assert_eq!(e.sels(m)[0], "setPurgeableState:");
                assert!(r.state.host_valid());
            }

            /// A raise mid-page leaves it host-invalid
            #[test]
            fn raise_leaves_host_invalid() {
                let e = Env::new();
                let r = made(&e, 2, 3);
                let (tr, _) = parts(&r);
                let enc = e.blit();
                e.p.script()
                    .after(UP, 1, Step::raise("NSGenericException", "x"));
                let hold = Hold::NONE;
                assert!(r.prepare_in_encoder(hold, Some(&enc)).is_err());
                assert!(!r.state.host_valid());
                // The transfer buffer stays bound.
                assert!(tr.paged.transfer_buffer.lock().is_some());
            }
        }

        mod synchronize {
            use super::*;

            #[test]
            fn guest_valid_nothing() {
                let e = Env::new();
                let r = made(&e, 2, 3);
                let m = e.p.call_log().mark();
                r.synchronize_in_encoder(Hold::NONE, None).unwrap();
                assert!(e.sels(m).is_empty());
            }

            /// Pages down (no mipmaps, no devolatilize); host-invalid logs
            #[test]
            fn pages_down() {
                let e = Env::new();
                let r = made(&e, 3, 3);
                r.state.set_guest_valid(false);
                r.state.set_is_volatile(true);
                let enc = e.blit();
                let m = e.p.call_log().mark();
                let (synchronized, lines) =
                    logged(|| r.synchronize_in_encoder(Hold::NONE, Some(&enc)));
                synchronized.unwrap();
                assert_eq!(e.sels(m), vec!["new_buffer_for_virtual_page", DOWN]);
                assert_eq!(lines, vec![SYNCHRONIZE_INVALID]);
                assert!(r.state.guest_valid());
                assert!(r.state.is_volatile());
            }

            /// Protected: guest-valid without paging
            #[test]
            fn protected() {
                let e = Env::new();
                let r = made(&e, 2, 3);
                r.state.set_guest_valid(false);
                r.state.set_host_valid(true);
                parts(&r).0.paged.texture.set_protection_options(2);
                let m = e.p.call_log().mark();
                r.synchronize_in_encoder(Hold::NONE, None).unwrap();
                assert!(e.sels(m).is_empty());
                assert!(r.state.guest_valid());
            }
        }

        #[test]
        fn the_last_release_releases() {
            let e = Env::new();
            let t = e.tex(desc(70, 16, 8, 1, 3), false);
            let r = make(&e, 2, t.clone_ref(), &good16x8(), 0x84)
                .unwrap()
                .unwrap();
            parts(&r).0.paged.ensure_physical(Hold::NONE).unwrap();
            assert_eq!(t.retain_count(), 2);
            drop(r);
            assert_eq!(t.retain_count(), 1);
        }
    }

    /// Two planes.
    mod dual {

        use super::*;
        use crate::guest::records::GuestFeatures;
        use crate::platform::Handle;
        use crate::resources::resource::ResourceKind;
        use crate::resources::texture::TextureResource;
        use crate::testkit::capture::catch_death;
        use crate::testkit::{FakePlatform, Step};

        type P = FakePlatform;

        const UP: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:";
        const DOWN: &str = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:";

        fn env(mipped: bool) -> Env {
            let e = Env::new();
            e.features(&GuestFeatures {
                dual_plane_textures: true,
                s8_byte_count_for_d32s8: 1,
                supports_multiplane_mipped_blit_textures: mipped,
                ..GuestFeatures::default()
            });
            e
        }

        /// A 4x4 D32S8 record of `levels` mips: plane 0 at 4 bytes per pixel, plane 1 at 1.
        fn d32s8(levels: u32) -> TextureDimensionRecord {
            let (mut m0, mut m1) = (Vec::new(), Vec::new());
            let (mut w, mut off0, mut off1) = (4u32, 0u64, 0x100u64);
            for _ in 0..levels {
                m0.push(mip(off0, (w * w * 4) as u64, (w * 4) as u64, w, w, 1));
                m1.push(mip(off1, (w * w) as u64, w as u64, w, w, 1));
                off0 += (w * w * 4) as u64;
                off1 += (w * w) as u64;
                w = (w / 2).max(1);
            }
            rec2(header(0, 4, levels), m0, header(0, 1, levels), m1)
        }

        fn len(levels: u64) -> u64 {
            2 * (0x18 + levels * 0x24)
        }

        fn make(
            e: &Env,
            t: Tex,
            r: &TextureDimensionRecord,
            l: u64,
        ) -> RaiseResult<Option<Arc<Resource<P>>>> {
            e.run(|| DualPlaneBlit::new(e.task(), &paging(5, 0x200), r, l, t))
        }

        fn parts(r: &Resource<P>) -> (&TextureResource<P>, &DualPlaneBlit) {
            match &r.kind {
                ResourceKind::Texture(t) => match &t.layout {
                    TextureLayout::DualPlaneBlit(s) => (t, s),
                    _ => panic!("not dual-plane blit"),
                },
                _ => panic!("not a texture"),
            }
        }

        mod new {
            use super::*;

            #[test]
            fn one_level() {
                let e = env(false);
                let t = e.tex(desc(260, 4, 4, 1, 1), false);
                let m = e.p.call_log().mark();
                let r = make(&e, t.clone_ref(), &d32s8(1), len(1)).unwrap().unwrap();
                let (tr, s) = parts(&r);
                // One `[task features]`: the mipped check short-circuits.
                assert_eq!(e.calls(m, "features").len(), 1);
                assert_eq!(e.sels(m)[0], "features");
                assert_eq!(s.planes[0].header.block_bytes, 4);
                assert_eq!(s.planes[1].header.block_bytes, 1);
                assert_eq!({ s.planes[1].mips[0].offset }, 0x100);
                assert_eq!(tr.paged.page, 5);
                assert!(r.state.guest_valid());
                assert_eq!(t.retain_count(), 2);
            }

            #[test]
            fn mipped_supported() {
                let e = env(true);
                let t = e.tex(desc(260, 4, 4, 1, 3), false);
                let m = e.p.call_log().mark();
                let r = make(&e, t, &d32s8(3), len(3)).unwrap().unwrap();
                assert_eq!(e.calls(m, "features").len(), 2);
                let (_, s) = parts(&r);
                assert_eq!(s.planes[1].mips.len(), 3);
                assert_eq!(s.planes[1].header.mip_levels(), 3);
            }

            /// Plane 1's count is plane 0's
            #[test]
            fn plane_one_takes_plane_zero_count() {
                let e = env(false);
                let t = e.tex(desc(260, 4, 4, 1, 1), false);
                let r = rec2(
                    header(0, 4, 1),
                    vec![mip(0, 0x40, 16, 4, 4, 1)],
                    header(0, 1, 5),
                    vec![mip(0x100, 0x10, 4, 4, 4, 1)],
                );
                let res = make(&e, t, &r, len(1)).unwrap().unwrap();
                let p1 = &parts(&res).1.planes[1];
                assert_eq!(p1.header.mip_levels(), 1);
                assert_eq!(p1.mips.len(), 1);
            }

            /// Mipped unsupported: raise, texture leaks
            #[test]
            fn mipped_unsupported_raises() {
                let e = env(false);
                let t = e.tex(desc(260, 4, 4, 1, 3), false);
                let m = e.p.call_log().mark();
                let err = make(&e, t.clone_ref(), &d32s8(3), len(3)).err().unwrap();
                assert_eq!(raised(&err), Some(RaiseKind::MippedMultiPlane));
                assert_eq!(e.sels(m), vec!["features"]);
                // The resource released its reference; the caller's leaked.
                assert_eq!(t.retain_count(), 2);
            }

            /// Short descriptor releases both
            #[test]
            fn short_length() {
                let e = env(false);
                let t = e.tex(desc(260, 4, 4, 1, 1), false);
                let err = make(&e, t.clone_ref(), &d32s8(1), 0x2f).err().unwrap();
                assert_eq!(raised(&err), Some(RaiseKind::DimensionLength));
                assert_eq!(t.retain_count(), 1);
            }

            #[test]
            fn invalid_nil() {
                let e = env(false);
                // RGBA8 is not two-plane: "Dual plane mismatch".
                let t = e.tex(desc(70, 4, 4, 1, 1), false);
                let r = make(&e, t.clone_ref(), &d32s8(1), len(1)).unwrap();
                assert!(r.is_none());
                assert_eq!(t.retain_count(), 1);
            }

            /// Dead task at the mipped check
            #[test]
            fn dead_task_dies() {
                let e = env(false);
                let t = e.tex(desc(260, 4, 4, 1, 3), false);
                let dead = {
                    let d: Arc<dyn crate::resources::host::ResourceHost<P>> =
                        crate::testkit::resource_host::FakeResourceHost::new(e.p.clone());
                    Arc::downgrade(&d)
                };
                let got = catch_death(|| {
                    e.run(|| DualPlaneBlit::new(dead, &paging(0, 0x200), &d32s8(3), len(3), t))
                });
                assert_eq!(
                    got.err(),
                    Some(crate::resources::DeathKind::GoneTaskFeatures.site())
                );
            }
        }

        fn made(e: &Env) -> Arc<Resource<P>> {
            let t = e.tex(desc(260, 4, 4, 1, 1), false);
            make(e, t, &d32s8(1), len(1)).unwrap().unwrap()
        }

        mod paging {
            use super::*;

            /// Plane 0 then plane 1, depth then stencil option
            #[test]
            fn plane_major_with_options() {
                let e = env(false);
                let r = made(&e);
                let (tr, s) = parts(&r);
                tr.paged.ensure_physical(Hold::NONE).unwrap();
                let enc = e.blit();
                let m = e.p.call_log().mark();
                s.page(&tr.paged, Some(&enc), false).unwrap();
                let c = e.calls(m, UP);
                assert_eq!(c.len(), 2);
                assert_eq!((u(&c[0], 1), u(&c[0], 2), u(&c[0], 9)), (0, 16, 1));
                assert_eq!((u(&c[1], 1), u(&c[1], 2), u(&c[1], 9)), (0x100, 4, 2));
                let m = e.p.call_log().mark();
                s.page(&tr.paged, Some(&enc), true).unwrap();
                let c = e.calls(m, DOWN);
                assert_eq!((u(&c[0], 6), u(&c[0], 9)), (0, 1));
                assert_eq!((u(&c[1], 6), u(&c[1], 9)), (0x100, 2));
            }

            /// The upload assert
            #[test]
            fn upload_assert() {
                let e = env(false);
                let r = made(&e);
                let (tr, _) = parts(&r);
                let s = DualPlaneBlit {
                    planes: [
                        rec1(header(0, 4, 1), vec![mip(0, 0, 16, 4, 4, 1)]),
                        rec1(header(0, 1, 1), vec![mip(0, 0, 3, 4, 4, 1)]),
                    ],
                };
                let got = catch_death(|| s.page::<P>(&tr.paged, None, false));
                assert_eq!(got.err(), Some(died(DeathKind::ShortUploadRow)));
            }
        }

        mod ensure_physical {
            use super::*;

            #[test]
            fn binds_once() {
                let e = env(false);
                let r = made(&e);
                let (tr, _) = parts(&r);
                let m = e.p.call_log().mark();
                tr.paged.ensure_physical(Hold::NONE).unwrap();
                tr.paged.ensure_physical(Hold::NONE).unwrap();
                let c = e.calls(m, "new_buffer_for_virtual_page");
                assert_eq!(c.len(), 1);
                assert_eq!(c[0].args[0], crate::testkit::Arg::U32(5));
                assert!(tr.paged.transfer_buffer.lock().is_some());
            }

            /// The task's raise propagates
            #[test]
            fn raise() {
                let e = env(false);
                let r = made(&e);
                let (tr, _) = parts(&r);
                e.p.script().push(
                    "new_buffer_for_virtual_page",
                    Step::raise("NSGenericException", "Failed to create buffer"),
                );
                let hold = Hold::NONE;
                assert!(tr.paged.ensure_physical(hold).is_err());
                assert!(tr.paged.transfer_buffer.lock().is_none());
            }
        }

        mod prepare_synchronize {
            use super::*;

            /// No mipmaps; protected is valid unpaged
            #[test]
            fn prepare() {
                let e = env(false);
                let r = made(&e);
                let enc = e.blit();
                let m = e.p.call_log().mark();
                r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
                assert_eq!(e.sels(m), vec!["new_buffer_for_virtual_page", UP, UP]);
                assert!(r.state.host_valid());

                let r = made(&e);
                r.state.set_guest_valid(false);
                parts(&r).0.paged.texture.set_protection_options(1);
                let m = e.p.call_log().mark();
                let (prepared, lines) = logged(|| r.prepare_in_encoder(Hold::NONE, Some(&enc)));
                prepared.unwrap();
                assert!(e.sels(m).is_empty());
                assert!(r.state.host_valid());
                assert_eq!(lines, vec![PREPARE_INVALID]);
            }

            #[test]
            fn synchronize() {
                let e = env(false);
                let r = made(&e);
                let m = e.p.call_log().mark();
                r.synchronize_in_encoder(Hold::NONE, None).unwrap();
                assert!(e.sels(m).is_empty());
                r.state.set_guest_valid(false);
                let enc = e.blit();
                let (synchronized, lines) =
                    logged(|| r.synchronize_in_encoder(Hold::NONE, Some(&enc)));
                synchronized.unwrap();
                assert_eq!(e.sels(m), vec!["new_buffer_for_virtual_page", DOWN, DOWN]);
                assert!(r.state.guest_valid());
                assert_eq!(lines, vec![SYNCHRONIZE_INVALID]);
            }
        }

        #[test]
        fn the_last_release() {
            let e = env(false);
            let t = e.tex(desc(260, 4, 4, 1, 1), false);
            let r = make(&e, t.clone_ref(), &d32s8(1), len(1)).unwrap().unwrap();
            assert_eq!(t.retain_count(), 2);
            drop(r);
            assert_eq!(t.retain_count(), 1);
        }
    }
}
