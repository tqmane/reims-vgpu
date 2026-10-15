#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdatomic.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nkernel void verify(device uint *out [[buffer(0)]],constant uint &value [[buffer(1)]],uint i [[thread_position_in_grid]]) { out[i] = value + i; }" options:nil error:&error];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"verify"] error:&error];
        if (!pipeline) { fprintf(stderr, "pipeline: %s\n", error.description.UTF8String); return 2; }
        __block atomic_uint failures = 0, frames = 0;
        dispatch_apply(4, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t worker) {
            @autoreleasepool {
                id<MTLCommandQueue> queue = [device newCommandQueue];
                id<MTLBuffer> buffer = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
                if (!queue || !buffer || !buffer.contents) { atomic_fetch_add(&failures, 1); return; }
                for (uint32_t frame = 0; frame < 25; frame++) {
                    @autoreleasepool {
                        uint32_t base = (uint32_t)worker * 10000 + frame * 100;
                        memset(buffer.contents, 0xff, 256);
                        id<MTLCommandBuffer> command = [queue commandBuffer];
                        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                        if (!command || !encoder) { atomic_fetch_add(&failures, 1); return; }
                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:buffer offset:0 atIndex:0];
                        [encoder setBytes:&base length:sizeof(base) atIndex:1];
                        [encoder dispatchThreadgroups:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(16,1,1)];
                        [encoder endEncoding];
                        dispatch_semaphore_t done = dispatch_semaphore_create(0);
                        [command addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(done); }];
                        [command commit];
                        long timeout = dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC));
                        if (timeout || command.status != MTLCommandBufferStatusCompleted) {
                            fprintf(stderr, "worker=%zu frame=%u timeout=%ld status=%lu error=%s\n", worker, frame, timeout, (unsigned long)command.status, command.error.description.UTF8String);
                            atomic_fetch_add(&failures, 1); return;
                        }
                        const uint32_t *values = buffer.contents;
                        for (unsigned i = 0; i < 64; i++) {
                            if (values[i] != base + i) {
                                fprintf(stderr, "worker=%zu frame=%u element=%u actual=%u expected=%u\n", worker, frame, i, values[i], base+i);
                                atomic_fetch_add(&failures, 1); return;
                            }
                        }
                        atomic_fetch_add(&frames, 1);
                    }
                }
            }
        });
        unsigned completed = atomic_load(&frames), failed = atomic_load(&failures);
        printf("{\"queues\":4,\"completed_frames\":%u,\"failures\":%u,\"passed\":%s}\n",
            completed, failed, completed == 100 && !failed ? "true" : "false");
        return completed == 100 && !failed ? 0 : 3;
    }
}
