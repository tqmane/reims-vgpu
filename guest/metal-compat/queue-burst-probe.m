#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdatomic.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return 1;
        NSMutableArray<id<MTLBuffer>> *buffers = [NSMutableArray new];
        dispatch_group_t done = dispatch_group_create();
        __block atomic_uint failures = 0;
        for (unsigned i=0;i<64;i++) {
            id<MTLBuffer> buffer = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
            if (!buffer) return 2;
            memset(buffer.contents, 0, 4096);
            [buffers addObject:buffer];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
            if (!command || !encoder) return 3;
            [encoder fillBuffer:buffer range:NSMakeRange(0, 4096) value:i+1];
            [encoder endEncoding];
            dispatch_group_enter(done);
            [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                if (completed.status != MTLCommandBufferStatusCompleted) {
                    fprintf(stderr, "frame=%u error=%s\n", i, completed.error.description.UTF8String);
                    atomic_fetch_add(&failures, 1);
                }
                dispatch_group_leave(done);
            }];
            [command commit];
        }
        if (dispatch_group_wait(done, dispatch_time(DISPATCH_TIME_NOW, 15*NSEC_PER_SEC))) return 4;
        if (atomic_load(&failures)) return 5;
        for (unsigned i=0;i<64;i++) for (unsigned j=0;j<4096;j++)
            if (((const uint8_t *)buffers[i].contents)[j] != i+1) return 6;
        puts("{\"stage\":\"queue-burst\",\"frames\":64,\"bytes_checked\":262144,\"passed\":true}");
        return 0;
    }
}
