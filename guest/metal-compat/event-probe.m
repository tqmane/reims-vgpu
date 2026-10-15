#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static BOOL finish(id<MTLCommandBuffer> command) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [command addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(done); }];
    [command commit];
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10*NSEC_PER_SEC))) return NO;
    if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "command: %s\n", command.error.description.UTF8String);
        return NO;
    }
    return YES;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        BOOL signalOnly = argc == 2 && !strcmp(argv[1], "--signal-only");
        if (argc != 1 && !signalOnly) return 1;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLSharedEvent> event = [device newSharedEvent];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> data = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        if (!event || !queue || !data) return 2;
        memset(data.contents, 0, 4096);
        id<MTLCommandBuffer> signal = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [signal blitCommandEncoder];
        [encoder fillBuffer:data range:NSMakeRange(0, 4096) value:0x5a];
        [encoder endEncoding];
        [signal encodeSignalEvent:event value:1];
        if (!finish(signal) || event.signaledValue != 1) return 3;
        for (unsigned i=0;i<4096;i++) if (((const uint8_t *)data.contents)[i] != 0x5a) return 4;
        puts("{\"stage\":\"shared-event-signal\",\"passed\":true}"); fflush(stdout);
        if (signalOnly) return 0;
        id<MTLBuffer> copied = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        if (!copied) return 5;
        memset(copied.contents, 0, 4096);
        id<MTLCommandBuffer> wait = [queue commandBuffer];
        [wait encodeWaitForEvent:event value:1];
        encoder = [wait blitCommandEncoder];
        [encoder copyFromBuffer:data sourceOffset:0 toBuffer:copied destinationOffset:0 size:4096];
        [encoder endEncoding];
        if (!finish(wait)) return 6;
        for (unsigned i=0;i<4096;i++) if (((const uint8_t *)copied.contents)[i] != 0x5a) return 7;
        puts("{\"stage\":\"shared-event-wait\",\"passed\":true}");
        id<MTLCommandQueue> consumer = [device newCommandQueue];
        if (!consumer) return 8;
        id<MTLCommandBuffer> blocked = [consumer commandBuffer];
        [blocked encodeWaitForEvent:event value:2];
        encoder = [blocked blitCommandEncoder];
        [encoder copyFromBuffer:data sourceOffset:0 toBuffer:copied destinationOffset:0 size:4096];
        [encoder endEncoding];
        dispatch_semaphore_t consumed = dispatch_semaphore_create(0);
        [blocked addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(consumed); }];
        [blocked commit];
        id<MTLCommandBuffer> producer = [queue commandBuffer];
        encoder = [producer blitCommandEncoder];
        [encoder fillBuffer:data range:NSMakeRange(0, 4096) value:0x6b];
        [encoder endEncoding];
        [producer encodeSignalEvent:event value:2];
        if (!finish(producer) || dispatch_semaphore_wait(consumed, dispatch_time(DISPATCH_TIME_NOW, 10*NSEC_PER_SEC)) ||
            blocked.status != MTLCommandBufferStatusCompleted || event.signaledValue != 2) return 9;
        for (unsigned i=0;i<4096;i++) if (((const uint8_t *)copied.contents)[i] != 0x6b) return 10;
        puts("{\"stage\":\"cross-queue-future-event\",\"passed\":true}");
        return 0;
    }
}
