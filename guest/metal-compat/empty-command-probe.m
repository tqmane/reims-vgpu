#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return 1;
        for (unsigned i=0;i<32;i++) {
            @autoreleasepool {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                if (!command) return 2;
                dispatch_semaphore_t done = dispatch_semaphore_create(0);
                [command addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(done); }];
                [command commit];
                if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10*NSEC_PER_SEC)) ||
                    command.status != MTLCommandBufferStatusCompleted) {
                    fprintf(stderr, "empty command %u: %s\n", i, command.error.description.UTF8String);
                    return 3;
                }
            }
        }
        puts("{\"stage\":\"empty-command\",\"commands\":32,\"passed\":true}");
        return 0;
    }
}
