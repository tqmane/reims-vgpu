#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static BOOL finish(id<MTLCommandBuffer> command) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [command addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(done); }];
    [command commit];
    return dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC)) == 0 &&
        command.status == MTLCommandBufferStatusCompleted;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLBuffer> input = [device newBufferWithLength:4096 options:MTLResourceStorageModeManaged];
        id<MTLBuffer> output = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        if (!device || !queue || !input || !output) return 2;
        for (unsigned i = 0; i < 64; i++) ((uint32_t *)input.contents)[i] = 17 + i;
        [input didModifyRange:NSMakeRange(0, 256)];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nkernel void verify(device const uint *in [[buffer(0)]],device uint *out [[buffer(1)]],uint i [[thread_position_in_grid]]){out[i]=in[i]*2+3;}" options:nil error:&error];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"verify"] error:&error];
        if (!pipeline) return 3;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
        [compute setComputePipelineState:pipeline];
        [compute setBuffer:input offset:0 atIndex:0];
        [compute setBuffer:output offset:0 atIndex:1];
        [compute dispatchThreadgroups:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(16,1,1)];
        [compute endEncoding];
        BOOL bufferPassed = finish(command);
        for (unsigned i = 0; i < 64; i++) bufferPassed &= ((uint32_t *)output.contents)[i] == (17+i)*2+3;
        printf("{\"stage\":\"managed-buffer-upload\",\"passed\":%s}\n", bufferPassed ? "true" : "false");
        fflush(stdout);
        if (!bufferPassed) return 4;

        MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:16 height:16 mipmapped:NO];
        desc.textureType = MTLTextureType2DMultisample;
        desc.sampleCount = 4;
        desc.storageMode = MTLStorageModeMemoryless;
        desc.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> multisample = [device newTextureWithDescriptor:desc];
        desc.textureType = MTLTextureType2D;
        desc.sampleCount = 1;
        desc.storageMode = MTLStorageModeManaged;
        id<MTLTexture> resolved = [device newTextureWithDescriptor:desc];
        if (!multisample || !resolved) return 5;
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = multisample;
        pass.colorAttachments[0].resolveTexture = resolved;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(1,0,0,1);
        command = [queue commandBuffer];
        [[command renderCommandEncoderWithDescriptor:pass] endEncoding];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        [blit synchronizeResource:resolved];
        [blit endEncoding];
        BOOL finished = finish(command);
        BOOL texturePassed = finished;
        uint8_t pixels[16*16*4] = {0};
        [resolved getBytes:pixels bytesPerRow:64 fromRegion:MTLRegionMake2D(0,0,16,16) mipmapLevel:0];
        for (unsigned i = 0; i < 256; i++)
            texturePassed &= pixels[i*4] == 0 && pixels[i*4+1] == 0 && pixels[i*4+2] == 255 && pixels[i*4+3] == 255;
        if (!texturePassed) fprintf(stderr, "finished=%d status=%lu error=%s first_pixel=%u,%u,%u,%u\n",
            finished, (unsigned long)command.status, command.error.description.UTF8String,
            pixels[0],pixels[1],pixels[2],pixels[3]);
        printf("{\"stage\":\"memoryless-msaa-resolve\",\"pixels\":256,\"passed\":%s}\n", texturePassed ? "true" : "false");
        fflush(stdout);
        if (!texturePassed) return 6;
        desc.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> uploaded = [device newTextureWithDescriptor:desc];
        if (!uploaded) return 7;
        for (unsigned i = 0; i < 256; i++) {
            pixels[i*4] = 0; pixels[i*4+1] = 255; pixels[i*4+2] = 0; pixels[i*4+3] = 255;
        }
        [uploaded replaceRegion:MTLRegionMake2D(0,0,16,16) mipmapLevel:0 withBytes:pixels bytesPerRow:64];
        memset(output.contents, 0x55, 4096);
        command = [queue commandBuffer];
        blit = [command blitCommandEncoder];
        [blit copyFromTexture:uploaded sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
            sourceSize:MTLSizeMake(16,16,1) toBuffer:output destinationOffset:0 destinationBytesPerRow:256 destinationBytesPerImage:4096];
        [blit endEncoding];
        BOOL uploadPassed = finish(command);
        const uint8_t *readback = output.contents;
        for (unsigned y = 0; y < 16; y++) for (unsigned x = 0; x < 16; x++) {
            unsigned offset = y*256 + x*4;
            uploadPassed &= readback[offset] == 0 && readback[offset+1] == 255 && readback[offset+2] == 0 && readback[offset+3] == 255;
        }
        printf("{\"stage\":\"managed-texture-upload\",\"pixels\":256,\"passed\":%s}\n", uploadPassed ? "true" : "false");
        return uploadPassed ? 0 : 8;
    }
}
