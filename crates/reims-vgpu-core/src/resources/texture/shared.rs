//! Shared textures: task 0's shared backings ([`SharedSingle`], [`SharedDual`], object types 17
//! and 18), paged by blits and prepared on demand ([`prepare_now`]), and another task's texture
//! over one ([`SharedTextureResource`], type 16), opened by a shared texture handle.

use std::sync::Arc;

use crate::nil_send;
use crate::num::Truncate;
use crate::platform::blit::{BlitCmd, BlitEncoder};
use crate::platform::command::{CommandBuffer, CommandQueue};
use crate::platform::resource::Texture;
use crate::platform::{Handle, Platform};
use crate::raise::{Kept, keep_forever};
use crate::raise::{OnFault, RaiseResult};
use crate::resources::resource::AnyResource;
use crate::resources::resource::{
    Resource, ResourceClass, ResourceKind, ResourceState, TaskRef, features_of, platform_for,
};
use crate::resources::texture::blit::{
    BLIT_OPTION_ROW_LINEAR_PVRTC, DEPTH_STENCIL_BLIT_OPTIONS, page,
};
use crate::resources::texture::dimension::{
    TextureDimensionRecord, TexturePagingInfo, validate_texture_dimension,
};
use crate::resources::texture::{
    DIMENSION_SIZE, DeathKind, MIP_DIMENSION_SIZE, PagedTexture, RaiseKind, TextureLayout,
    into_resource, raise, record_at,
};
use crate::rt::{Held, Hold};

/// A one-plane shared backing, paged by blits.
#[derive(Debug, Clone, Default)]
pub struct SharedSingle {
    /// The validated dimension, its mips included.
    pub(crate) dimension: TextureDimensionRecord,
    /// [`BLIT_OPTION_ROW_LINEAR_PVRTC`] for a PVRTC texture, else 0.
    pub(crate) blit_option: u64,
}

/// A two-plane shared backing, paged by blits as the two-plane blit layout is.
#[derive(Debug, Clone, Default)]
pub struct SharedDual {
    /// The two planes' validated dimensions.
    pub(crate) planes: [TextureDimensionRecord; 2],
}

/// Prepares now: one command buffer from `queue`, `prepare` in its blit encoder, then end,
/// commit and wait, all in a real pool of its own, which holds Metal's `+0` command buffer and
/// encoder to its end. A raise from `prepare` leaves the encoder open, the buffer uncommitted
/// and the real pool pushed, so both live on until the enclosing pop.
///
/// It is also a landing frame ([`Held::raising`]), and `prepare` runs under a hold of the
/// frame's holder: a deep hold the paging's guest reads make (the page-table read's device
/// handle) is released at its end, before the pool pops, or goes on with its raise.
pub(crate) fn prepare_now<P: Platform>(
    queue: Option<&P::CommandQueue>,
    prepare: impl FnOnce(Hold<'_>, Option<&P::BlitEncoder>) -> RaiseResult<()>,
) -> RaiseResult<()> {
    crate::rt::pool::around_here(|| {
        Held::raising(|held| {
            let hold = Hold::of(held);
            let command_buffer: Option<P::CommandBuffer> =
                nil_send!(queue, command_buffer(OnFault::Return))?;
            let encoder: Option<P::BlitEncoder> =
                nil_send!(command_buffer, blit_encoder(OnFault::Return))?;
            prepare(hold, encoder.as_ref())?;
            nil_send!(encoder, encode(OnFault::Return, BlitCmd::EndEncoding))?;
            nil_send!(command_buffer, commit(OnFault::Return))?;
            nil_send!(command_buffer, wait_until_completed(OnFault::Return))?;
            Ok(())
        })
    })
}

impl SharedSingle {
    /// A one-plane shared backing of object type 17. `None` when validation refuses the record,
    /// or with no platform scope. `texture` is the caller's reference, shareable (the manager
    /// made it so).
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
        // The shareable assert, kept though unreachable.
        crate::fatal_unless!(texture.is_shareable(), DeathKind::NotShareable);
        let Some(platform) = platform_for(&task) else {
            drop(texture);
            return Ok(None);
        };
        let mut paged = PagedTexture::new(&platform, task, &texture);
        paged.page = paging_info.page;
        paged.length = paging_info.length;
        let mip_levels = u64::from(dimension.header.mip_levels());
        let size = DIMENSION_SIZE.wrapping_add(mip_levels.wrapping_mul(MIP_DIMENSION_SIZE));
        // Two checks, the header's then the mips'. Unreachable: the manager sizes the copy.
        if dimension_length < DIMENSION_SIZE || dimension_length < size {
            // The half-made texture is released, then the caller's texture.
            drop(paged);
            drop(texture);
            return raise(RaiseKind::DimensionLength);
        }
        let mut dest = TextureDimensionRecord::default();
        let features = features_of(&paged.task);
        let valid = validate_texture_dimension::<P, 1>(
            [dimension],
            [&mut dest],
            mip_levels.low_u32(),
            false,
            paged.length,
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
        drop(texture);
        Ok(Some(into_resource(
            paged,
            TextureLayout::SharedSingle(SharedSingle {
                dimension: dest,
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

impl SharedDual {
    /// A two-plane shared backing of object type 18 in a format the blits page. `None` when
    /// validation refuses the record, or with no platform scope. `texture` is the caller's
    /// reference, shareable.
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
        // The shareable assert, kept though unreachable.
        crate::fatal_unless!(texture.is_shareable(), DeathKind::NotShareable);
        let Some(platform) = platform_for(&task) else {
            drop(texture);
            return Ok(None);
        };
        let mut paged = PagedTexture::new(&platform, task, &texture);
        paged.page = paging_info.page;
        paged.length = paging_info.length;
        let mip_levels = u64::from(dimension.header.mip_levels());
        // The task's features are read here only when mipped.
        if mip_levels != 1 && !features_of(&paged.task).supports_multiplane_mipped_blit_textures {
            drop(paged);
            // The caller's texture is kept for good: its resource index and GPU VA stay taken,
            // and later placements are numbered after them.
            keep_forever(texture, Kept::MetalNumbering);
            return raise(RaiseKind::MippedMultiPlane);
        }
        let size = DIMENSION_SIZE.wrapping_add(mip_levels.wrapping_mul(MIP_DIMENSION_SIZE));
        if dimension_length < 2_u64.wrapping_mul(size) {
            // The half-made texture is released, then the caller's texture. Unreachable: the
            // manager sizes the copy.
            drop(paged);
            drop(texture);
            return raise(RaiseKind::DimensionLength);
        }
        // Plane 1 after plane 0's mips, with plane 0's count.
        let plane1 = record_at(dimension, size, mip_levels);
        let mut dest = [
            TextureDimensionRecord::default(),
            TextureDimensionRecord::default(),
        ];
        let features = features_of(&paged.task);
        let [d0, d1] = &mut dest;
        let valid = validate_texture_dimension::<P, 2>(
            [dimension, &plane1],
            [d0, d1],
            mip_levels.low_u32(),
            false,
            paged.length,
            &texture,
            &mut paged.is_pvrtc,
            &features,
        );
        if !valid {
            drop(paged);
            drop(texture);
            return Ok(None);
        }
        // Unreachable once validated.
        if dest[0].header.is_compressed() {
            drop(paged);
            // The caller's texture is released.
            drop(texture);
            return raise(RaiseKind::CompressedMultiPlane);
        }
        drop(texture);
        Ok(Some(into_resource(
            paged,
            TextureLayout::SharedDual(SharedDual { planes: dest }),
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

/// Another task's texture over a shared backing, opened by its handle: always valid, never
/// paged. It names its backing so that an exec's guest invalidation reaches it. It keeps no
/// task.
pub struct SharedTextureResource<P: Platform> {
    texture: P::Texture,
    shared_texture_backing_id: u32,
}

impl<P: Platform> SharedTextureResource<P> {
    /// A texture over the backing `shared_texture_backing_id` names. `texture` is the caller's
    /// reference, shareable; the resource holds its own. Never `None`, never a raise.
    #[expect(
        clippy::new_ret_no_self,
        reason = "a kind is made as its shared `Resource`"
    )]
    pub fn new(
        task: TaskRef<P>,
        texture: P::Texture,
        shared_texture_backing_id: u32,
    ) -> RaiseResult<Option<Arc<Resource<P>>>> {
        let _ = task;
        // The shareable assert, kept though unreachable.
        crate::fatal_unless!(texture.is_shareable(), DeathKind::NotShareable);
        let r = SharedTextureResource {
            texture: texture.clone_ref(),
            shared_texture_backing_id,
        };
        drop(texture);
        Ok(Some(Resource::new(ResourceKind::SharedTexture(r))))
    }

    /// The backing's id.
    pub fn shared_texture_backing_id(&self) -> u32 {
        self.shared_texture_backing_id
    }
}

impl<P: Platform> ResourceClass<P> for SharedTextureResource<P> {
    /// Always valid: a store to the flag is dead state.
    fn is_guest_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    fn is_host_valid(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    fn supports_discard(&self, this: &Resource<P>) -> bool {
        let _ = this;
        true
    }

    /// Nothing: the texture is the backing's own memory.
    fn prepare_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = (this, encoder);
        Ok(())
    }

    fn synchronize_in_encoder(
        &self,
        _hold: Hold<'_>,
        this: &Resource<P>,
        encoder: Option<&P::BlitEncoder>,
    ) -> RaiseResult<()> {
        let _ = (this, encoder);
        Ok(())
    }

    fn discard(&self, this: &Resource<P>) -> RaiseResult<()> {
        let _ = this;
        Ok(())
    }

    fn get_resource(&self, this: &Resource<P>) -> RaiseResult<Option<AnyResource<P>>> {
        let _ = this;
        Ok(Some(AnyResource::Texture(self.texture.clone_ref())))
    }

    fn get_texture(&self, this: &Resource<P>) -> RaiseResult<Option<P::Texture>> {
        let _ = this;
        Ok(Some(self.texture.clone_ref()))
    }

    fn get_shared_texture_backing_id(&self, this: &Resource<P>) -> RaiseResult<u32> {
        let _ = this;
        Ok(self.shared_texture_backing_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::texture::test_support::*;
    use crate::testkit::FakePlatform;

    type P = FakePlatform;

    mod prepare_now {
        use super::*;
        use crate::platform::device::GpuDevice;
        use crate::rt::Pool;

        /// Order: buffer, encoder, prepare, end, commit, wait
        #[test]
        fn order() {
            let e = Env::new();
            e.p.world()
                .set_auto_complete(Some(crate::testkit::CompleteOn::Caller));
            let q = e.p.device().new_command_queue().unwrap();
            let m = e.p.call_log().mark();
            let mut seen = None;
            prepare_now::<P>(Some(&q), |_, enc| {
                seen = enc.map(Handle::id);
                Ok(())
            })
            .unwrap();
            assert!(seen.is_some());
            assert_eq!(
                e.sels(m),
                vec![
                    "commandBuffer",
                    "blitCommandEncoder",
                    "endEncoding",
                    "commit",
                    "waitUntilCompleted"
                ]
            );
        }

        /// A raise from the prepare: no end, no commit
        #[test]
        fn raise_skips_the_rest() {
            let e = Env::new();
            let q = e.p.device().new_command_queue().unwrap();
            let m = e.p.call_log().mark();
            let r = prepare_now::<P>(Some(&q), |_, _| Err(a_raise()));
            assert!(r.is_err());
            assert_eq!(e.sels(m), vec!["commandBuffer", "blitCommandEncoder"]);
        }

        /// A normal exit pops the pool: both released before it returns
        #[test]
        fn normal_exit_pops_pool() {
            let e = Env::new();
            e.p.world()
                .set_auto_complete(Some(crate::testkit::CompleteOn::Caller));
            let q = e.p.device().new_command_queue().unwrap();
            let _outer = Pool::push(&*e.p);
            let m = e.p.call_log().mark();
            let mut enc_id = None;
            prepare_now::<P>(Some(&q), |_, enc| {
                enc_id = enc.map(Handle::id);
                Ok(())
            })
            .unwrap();
            let cb_id = e.calls(m, "blitCommandEncoder")[0].obj;
            assert!(e.p.world().find(enc_id.unwrap()).is_none());
            assert!(e.p.world().find(cb_id).is_none());
        }

        /// The real pool is pushed before `-commandBuffer` and popped after `-waitUntilCompleted`,
        /// so Metal's own +0 autoreleases land in it
        #[test]
        fn real_pool_brackets_the_metal_calls() {
            use crate::rt::pool::test_native::{events, note, with_test_native};
            let e = Env::new();
            e.p.world()
                .set_auto_complete(Some(crate::testkit::CompleteOn::Caller));
            let q = e.p.device().new_command_queue().unwrap();
            with_test_native(|| {
                let _outer = Pool::push(&*e.p);
                let _ = events();
                let m = e.p.call_log().mark();
                let mut at_prepare = (Vec::new(), Vec::new());
                prepare_now::<P>(Some(&q), |_, _| {
                    note("prepare");
                    at_prepare = (events(), e.sels(m));
                    Ok(())
                })
                .unwrap();
                assert_eq!(
                    at_prepare,
                    (
                        vec!["push 0x101".to_owned(), "prepare".to_owned()],
                        vec!["commandBuffer", "blitCommandEncoder"]
                    )
                );
                assert_eq!(events(), vec!["pop 0x101"]);
                assert_eq!(
                    e.sels(m),
                    vec![
                        "commandBuffer",
                        "blitCommandEncoder",
                        "endEncoding",
                        "commit",
                        "waitUntilCompleted"
                    ]
                );
            });
        }

        /// A raise leaves the pool pushed (no pop at the raise); the enclosing pool's pop takes it
        /// with its own
        #[test]
        fn raise_leaves_the_real_pool_pushed() {
            use crate::rt::pool::test_native::{events, with_test_native};
            let e = Env::new();
            let q = e.p.device().new_command_queue().unwrap();
            with_test_native(|| {
                let outer = Pool::push(&*e.p);
                let r = prepare_now::<P>(Some(&q), |_, _| Err(a_raise()));
                assert!(r.is_err());
                assert_eq!(events(), vec!["push 0x100", "push 0x101"]);
                drop(outer);
                assert_eq!(events(), vec!["pop 0x100"]);
            });
        }

        /// A deep hold the prepare makes (the page-table read's device handle) lands in the
        /// prepare's own holder: released at its end, before its pool pops; on a raise it goes on
        /// with the raise, and the catcher's holder releases it after the report
        #[test]
        fn a_deep_hold_lands_in_the_prepare() {
            use crate::rt::pool::test_native::{events, note, with_test_native};
            struct Rel(&'static str);
            impl Drop for Rel {
                fn drop(&mut self) {
                    note(format!("release {}", self.0));
                }
            }
            let e = Env::new();
            e.p.world()
                .set_auto_complete(Some(crate::testkit::CompleteOn::Caller));
            let q = e.p.device().new_command_queue().unwrap();
            with_test_native(|| {
                prepare_now::<P>(Some(&q), |hold, _| {
                    hold.keep(Rel("device"));
                    Ok(())
                })
                .unwrap();
                assert_eq!(events(), ["push 0x100", "release device", "pop 0x100"]);
                let catcher = Held::new();
                {
                    let hold = Hold::of(&catcher);
                    let r = prepare_now::<P>(Some(&q), |hold, _| {
                        hold.keep(Rel("device"));
                        Err(a_raise())
                    });
                    note("report");
                    hold.land(r.unwrap_err());
                }
                drop(catcher);
                assert_eq!(events(), ["push 0x101", "report", "release device"]);
            });
        }

        /// A nil queue: every send is to nil
        #[test]
        fn nil_queue() {
            let e = Env::new();
            let m = e.p.call_log().mark();
            let mut called = false;
            prepare_now::<P>(None, |_, enc| {
                assert!(enc.is_none());
                called = true;
                Ok(())
            })
            .unwrap();
            assert!(called);
            assert!(e.sels(m).is_empty());
        }
    }

    /// Another task's texture over a backing.
    mod texture {

        use super::*;
        use crate::testkit::FakePlatform;
        use crate::testkit::capture::catch_death;

        type P = FakePlatform;

        fn inner(r: &Resource<P>) -> &SharedTextureResource<P> {
            match &r.kind {
                ResourceKind::SharedTexture(s) => s,
                _ => panic!("not a shared texture"),
            }
        }

        #[test]
        fn new_keeps_texture_and_id() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), true);
            let m = e.p.call_log().mark();
            let r = SharedTextureResource::new(e.task(), t.clone_ref(), 0x1234)
                .unwrap()
                .unwrap();
            assert!(e.sels(m).is_empty());
            assert_eq!(inner(&r).shared_texture_backing_id(), 0x1234);
            assert_eq!(r.get_shared_texture_backing_id().unwrap(), 0x1234);
            assert_eq!(t.retain_count(), 2);
            // The flags start false; the getters answer valid regardless.
            assert!(!r.state.host_valid());
            assert!(!r.state.guest_valid());
            assert!(r.is_host_valid() && r.is_guest_valid());
            assert!(!r.needs_prepare() && !r.needs_sync());
            drop(r);
            assert_eq!(t.retain_count(), 1);
        }

        /// Not shareable: the assert, before anything else
        #[test]
        fn not_shareable_dies() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), false);
            let task = e.task();
            let got = catch_death(|| SharedTextureResource::new(task, t, 1));
            assert_eq!(got.err(), Some(died(DeathKind::NotShareable)));
        }

        #[test]
        fn always_valid_never_paged() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 1), true);
            let r = SharedTextureResource::new(e.task(), t.clone_ref(), 7)
                .unwrap()
                .unwrap();
            let enc = e.blit();
            let m = e.p.call_log().mark();
            r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
            r.discard().unwrap();
            r.set_is_guest_valid(false);
            r.set_is_host_valid(false);
            assert!(e.sels(m).is_empty());
            assert!(r.supports_discard());
            assert!(r.is_host_valid());
            assert!(r.is_guest_valid());
            assert_eq!(r.get_texture().unwrap().unwrap().id(), t.id());
            match r.get_resource().unwrap().unwrap() {
                AnyResource::Texture(x) => assert_eq!(x.id(), t.id()),
                _ => panic!("not a texture"),
            }
            // Compute paging is not answered.
            {
                let hold = Hold::NONE;
                assert!(r.prepare_in_compute_encoder(hold, None).is_err());
            };
        }
    }

    /// One-plane backings.
    mod single {

        use super::*;
        use crate::platform::Handle;
        use crate::platform::device::GpuDevice;
        use crate::platform::types::PixelFormatInfo;
        use crate::resources::resource::ResourceKind;
        use crate::resources::texture::TextureResource;
        use crate::testkit::capture::catch_death;
        use crate::testkit::{CompleteOn, FakePlatform, Step};

        type P = FakePlatform;

        const UP: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:";
        const DOWN: &str = "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:";

        fn rec() -> TextureDimensionRecord {
            rec1(
                header(0x60, 4, 2),
                vec![mip(0, 0x40, 16, 4, 4, 1), mip(0x40, 0x10, 8, 2, 2, 1)],
            )
        }

        const LEN: u64 = 0x18 + 2 * 0x24;

        fn make(
            e: &Env,
            t: Tex,
            r: &TextureDimensionRecord,
            l: u64,
        ) -> RaiseResult<Option<Arc<Resource<P>>>> {
            e.run(|| SharedSingle::new(e.task(), &paging(4, 0x60), r, l, t))
        }

        fn parts(r: &Resource<P>) -> (&TextureResource<P>, &SharedSingle) {
            match &r.kind {
                ResourceKind::Texture(t) => match &t.layout {
                    TextureLayout::SharedSingle(s) => (t, s),
                    _ => panic!("not a single-plane backing"),
                },
                _ => panic!("not a texture"),
            }
        }

        fn made(e: &Env) -> Arc<Resource<P>> {
            let t = e.tex(desc(70, 4, 4, 1, 2), true);
            make(e, t, &rec(), LEN).unwrap().unwrap()
        }

        mod new {
            use super::*;

            #[test]
            fn validates_every_level() {
                let e = Env::new();
                let t = e.tex(desc(70, 4, 4, 1, 2), true);
                let m = e.p.call_log().mark();
                let r = make(&e, t.clone_ref(), &rec(), LEN).unwrap().unwrap();
                let (tr, s) = parts(&r);
                assert_eq!(s.dimension, rec());
                assert_eq!(s.blit_option, 0);
                assert_eq!(tr.paged.page, 4);
                assert_eq!(tr.paged.length, 0x60);
                assert!(r.state.guest_valid());
                assert_eq!(e.sels(m)[0], "features");
                assert_eq!(t.retain_count(), 2);
            }

            #[test]
            fn pvrtc() {
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
                let r = made(&e);
                assert_eq!(parts(&r).1.blit_option, 4);
            }

            /// Not shareable: the assert
            #[test]
            fn not_shareable_dies() {
                let e = Env::new();
                let t = e.tex(desc(70, 4, 4, 1, 2), false);
                let got = catch_death(|| make(&e, t, &rec(), LEN));
                assert_eq!(got.err(), Some(died(DeathKind::NotShareable)));
            }

            /// Both length checks
            #[test]
            fn short_lengths() {
                let e = Env::new();
                for l in [0x17u64, LEN - 1] {
                    let t = e.tex(desc(70, 4, 4, 1, 2), true);
                    let err = make(&e, t.clone_ref(), &rec(), l).err().unwrap();
                    assert_eq!(raised(&err), Some(RaiseKind::DimensionLength));
                    // The half-made texture and the caller's reference are released.
                    assert_eq!(t.retain_count(), 1);
                }
            }

            #[test]
            fn invalid_nil() {
                let e = Env::new();
                let t = e.tex(desc(70, 4, 4, 1, 1), true);
                let r = make(&e, t.clone_ref(), &rec(), LEN).unwrap();
                assert!(r.is_none());
                assert_eq!(t.retain_count(), 1);
            }

            #[test]
            fn dead_task() {
                let e = Env::new();
                let t = e.tex(desc(70, 4, 4, 1, 2), true);
                let dead = {
                    let d: Arc<dyn crate::resources::host::ResourceHost<P>> =
                        crate::testkit::resource_host::FakeResourceHost::new(e.p.clone());
                    Arc::downgrade(&d)
                };
                let got = catch_death(|| {
                    e.run(|| SharedSingle::new(dead, &paging(0, 0x60), &rec(), LEN, t))
                });
                assert_eq!(
                    got.err(),
                    Some(crate::resources::DeathKind::GoneTaskFeatures.site())
                );
            }
        }

        mod paging {
            use super::*;

            #[test]
            fn prepare_then_synchronize() {
                let e = Env::new();
                let r = made(&e);
                let enc = e.blit();
                let m = e.p.call_log().mark();
                r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
                assert_eq!(e.sels(m), vec!["new_buffer_for_virtual_page", UP, UP]);
                assert!(r.state.host_valid());
                r.state.set_guest_valid(false);
                let m = e.p.call_log().mark();
                r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
                assert_eq!(e.sels(m), vec![DOWN, DOWN]);
                assert!(r.state.guest_valid());
            }

            /// The upload assert
            #[test]
            fn upload_assert() {
                let e = Env::new();
                let r = made(&e);
                let (tr, _) = parts(&r);
                let s = SharedSingle {
                    dimension: rec1(header(0, 4, 1), vec![mip(0, 0, 1, 2, 1, 1)]),
                    blit_option: 0,
                };
                let got = catch_death(|| s.page::<P>(&tr.paged, None, false));
                assert_eq!(got.err(), Some(died(DeathKind::ShortUploadRow)));
            }

            #[test]
            fn logs() {
                let e = Env::new();
                let r = made(&e);
                let ((), lines) = logged(|| {
                    r.state.set_guest_valid(false);
                    r.prepare_in_encoder(Hold::NONE, None).unwrap();
                    r.state.set_host_valid(false);
                    r.state.set_guest_valid(false);
                    r.synchronize_in_encoder(Hold::NONE, None).unwrap();
                });
                assert_eq!(lines, vec![PREPARE_INVALID, SYNCHRONIZE_INVALID]);
            }
        }

        mod shared {
            use super::*;

            #[test]
            fn prepare_shared_waits() {
                let e = Env::new();
                e.p.world().set_auto_complete(Some(CompleteOn::Caller));
                let q = e.p.device().new_command_queue().unwrap();
                let r = made(&e);
                let (tr, _) = parts(&r);
                let m = e.p.call_log().mark();
                tr.prepare_shared_texture_backing(&r.state, Some(&q))
                    .unwrap();
                assert_eq!(
                    e.sels(m),
                    vec![
                        "commandBuffer",
                        "blitCommandEncoder",
                        "new_buffer_for_virtual_page",
                        UP,
                        UP,
                        "endEncoding",
                        "commit",
                        "waitUntilCompleted"
                    ]
                );
                assert!(r.state.host_valid());
            }

            /// A raise in the prepare: never ended or committed
            #[test]
            fn prepare_shared_raise() {
                let e = Env::new();
                let q = e.p.device().new_command_queue().unwrap();
                let r = made(&e);
                let (tr, _) = parts(&r);
                e.p.script().push(
                    "new_buffer_for_virtual_page",
                    Step::raise("NSGenericException", "x"),
                );
                let m = e.p.call_log().mark();
                assert!(
                    tr.prepare_shared_texture_backing(&r.state, Some(&q))
                        .is_err()
                );
                let sels = e.sels(m);
                assert!(!sels.contains(&"endEncoding"));
                assert!(!sels.contains(&"commit"));
            }

            #[test]
            fn handle() {
                let e = Env::new();
                let r = made(&e);
                let (tr, _) = parts(&r);
                let m = e.p.call_log().mark();
                let h = tr.new_shared_texture_handle().unwrap();
                assert!(h.is_some());
                assert_eq!(e.sels(m), vec!["newSharedTextureHandle"]);
                e.p.script().push("newSharedTextureHandle", Step::Nil);
                assert!(tr.new_shared_texture_handle().unwrap().is_none());
                assert!(!r.needs_compute_paging());
                assert!(tr.is_shared_texture_backing());
            }
        }

        #[test]
        fn the_last_release() {
            let e = Env::new();
            let t = e.tex(desc(70, 4, 4, 1, 2), true);
            let r = make(&e, t.clone_ref(), &rec(), LEN).unwrap().unwrap();
            drop(r);
            assert_eq!(t.retain_count(), 1);
        }
    }

    /// Two-plane backings.
    mod dual {

        use super::*;
        use crate::guest::records::GuestFeatures;
        use crate::platform::Handle;
        use crate::platform::device::GpuDevice;
        use crate::resources::resource::ResourceKind;
        use crate::resources::texture::TextureResource;
        use crate::testkit::capture::catch_death;
        use crate::testkit::{CompleteOn, FakePlatform, Step};

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

        /// A 4x4 D32S8 record of `levels` mips.
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
            e.run(|| SharedDual::new(e.task(), &paging(6, 0x200), r, l, t))
        }

        fn parts(r: &Resource<P>) -> (&TextureResource<P>, &SharedDual) {
            match &r.kind {
                ResourceKind::Texture(t) => match &t.layout {
                    TextureLayout::SharedDual(s) => (t, s),
                    _ => panic!("not a dual-plane backing"),
                },
                _ => panic!("not a texture"),
            }
        }

        fn made(e: &Env) -> Arc<Resource<P>> {
            let t = e.tex(desc(260, 4, 4, 1, 1), true);
            make(e, t, &d32s8(1), len(1)).unwrap().unwrap()
        }

        mod new {
            use super::*;

            #[test]
            fn one_level() {
                let e = env(false);
                let t = e.tex(desc(260, 4, 4, 1, 1), true);
                let m = e.p.call_log().mark();
                let r = make(&e, t.clone_ref(), &d32s8(1), len(1)).unwrap().unwrap();
                let (tr, s) = parts(&r);
                assert_eq!(e.sels(m)[0], "features");
                assert_eq!(e.calls(m, "features").len(), 1);
                assert_eq!(s.planes[1].header.block_bytes, 1);
                assert_eq!(tr.paged.page, 6);
                assert!(r.state.guest_valid());
                assert_eq!(t.retain_count(), 2);
            }

            #[test]
            fn mipped_supported() {
                let e = env(true);
                let t = e.tex(desc(260, 4, 4, 1, 3), true);
                let r = make(&e, t, &d32s8(3), len(3)).unwrap().unwrap();
                assert_eq!(parts(&r).1.planes[1].mips.len(), 3);
            }

            /// Mipped unsupported raises, texture leaks
            #[test]
            fn mipped_unsupported() {
                let e = env(false);
                let t = e.tex(desc(260, 4, 4, 1, 3), true);
                let err = make(&e, t.clone_ref(), &d32s8(3), len(3)).err().unwrap();
                assert_eq!(raised(&err), Some(RaiseKind::MippedMultiPlane));
                assert_eq!(t.retain_count(), 2);
            }

            #[test]
            fn short_length() {
                let e = env(false);
                let t = e.tex(desc(260, 4, 4, 1, 1), true);
                let err = make(&e, t.clone_ref(), &d32s8(1), len(1) - 1)
                    .err()
                    .unwrap();
                assert_eq!(raised(&err), Some(RaiseKind::DimensionLength));
                // The half-made texture and the caller's reference are released.
                assert_eq!(t.retain_count(), 1);
            }

            /// Not shareable: the assert
            #[test]
            fn not_shareable() {
                let e = env(false);
                let t = e.tex(desc(260, 4, 4, 1, 1), false);
                let got = catch_death(|| make(&e, t, &d32s8(1), len(1)));
                assert_eq!(got.err(), Some(died(DeathKind::NotShareable)));
            }

            #[test]
            fn invalid_nil() {
                let e = env(false);
                let t = e.tex(desc(70, 4, 4, 1, 1), true);
                let r = make(&e, t.clone_ref(), &d32s8(1), len(1)).unwrap();
                assert!(r.is_none());
                assert_eq!(t.retain_count(), 1);
            }
        }

        mod paging {
            use super::*;

            #[test]
            fn prepare_then_synchronize() {
                let e = env(false);
                let r = made(&e);
                let enc = e.blit();
                let m = e.p.call_log().mark();
                r.prepare_in_encoder(Hold::NONE, Some(&enc)).unwrap();
                let c = e.calls(m, UP);
                assert_eq!((u(&c[0], 9), u(&c[1], 9)), (1, 2));
                assert!(r.state.host_valid());
                r.state.set_guest_valid(false);
                let m = e.p.call_log().mark();
                r.synchronize_in_encoder(Hold::NONE, Some(&enc)).unwrap();
                assert_eq!(e.sels(m), vec![DOWN, DOWN]);
                assert!(r.state.guest_valid());
            }

            /// The invalid-state logs are this class's own lines
            #[test]
            fn logs() {
                let e = env(false);
                let r = made(&e);
                let ((), lines) = logged(|| {
                    r.state.set_guest_valid(false);
                    r.prepare_in_encoder(Hold::NONE, None).unwrap();
                    r.state.set_host_valid(false);
                    r.state.set_guest_valid(false);
                    r.synchronize_in_encoder(Hold::NONE, None).unwrap();
                });
                assert_eq!(lines, vec![PREPARE_INVALID, SYNCHRONIZE_INVALID]);
            }

            /// The upload assert
            #[test]
            fn upload_assert() {
                let e = env(false);
                let r = made(&e);
                let (tr, _) = parts(&r);
                let s = SharedDual {
                    planes: [
                        rec1(header(0, 4, 1), vec![mip(0, 0, 0, 4, 4, 1)]),
                        rec1(header(0, 1, 1), vec![]),
                    ],
                };
                let got = catch_death(|| s.page::<P>(&tr.paged, None, false));
                assert_eq!(got.err(), Some(died(DeathKind::ShortUploadRow)));
            }
        }

        mod shared {
            use super::*;

            #[test]
            fn prepare_shared() {
                let e = env(false);
                e.p.world().set_auto_complete(Some(CompleteOn::Caller));
                let q = e.p.device().new_command_queue().unwrap();
                let r = made(&e);
                let (tr, _) = parts(&r);
                let m = e.p.call_log().mark();
                tr.prepare_shared_texture_backing(&r.state, Some(&q))
                    .unwrap();
                let sels = e.sels(m);
                assert_eq!(&sels[..2], ["commandBuffer", "blitCommandEncoder"]);
                assert_eq!(
                    &sels[sels.len() - 3..],
                    ["endEncoding", "commit", "waitUntilCompleted"]
                );
                assert_eq!(e.calls(m, UP).len(), 2);
                // Host-valid now: a second prepare encodes nothing but still commits and waits.
                let m = e.p.call_log().mark();
                tr.prepare_shared_texture_backing(&r.state, Some(&q))
                    .unwrap();
                assert!(e.calls(m, UP).is_empty());
                assert_eq!(e.calls(m, "waitUntilCompleted").len(), 1);
            }

            #[test]
            fn handle() {
                let e = env(false);
                let r = made(&e);
                let (tr, _) = parts(&r);
                assert!(tr.new_shared_texture_handle().unwrap().is_some());
                e.p.script().push(
                    "newSharedTextureHandle",
                    Step::raise("NSGenericException", "x"),
                );
                assert!(tr.new_shared_texture_handle().is_err());
                assert!(!r.needs_compute_paging());
            }
        }

        #[test]
        fn the_last_release() {
            let e = env(false);
            let t = e.tex(desc(260, 4, 4, 1, 1), true);
            let r = make(&e, t.clone_ref(), &d32s8(1), len(1)).unwrap().unwrap();
            drop(r);
            assert_eq!(t.retain_count(), 1);
        }
    }
}
