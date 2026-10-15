#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static BOOL finish(id<MTLCommandBuffer> command) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [command addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(done); }];
    [command commit];
    if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 10*NSEC_PER_SEC))) return NO;
    if (command.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "command failed: %s\n", command.error.description.UTF8String);
        return NO;
    }
    return YES;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nvertex float4 vertexMain(uint id [[vertex_id]]) { const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)}; return float4(p[id],0,1); }\nfragment float4 fragmentMain(texture2d<float> source [[texture(0)]]) { return source.read(uint2(0)); }" options:nil error:&error];
        MTLRenderPipelineDescriptor *pipelineDesc = [MTLRenderPipelineDescriptor new];
        pipelineDesc.vertexFunction = [library newFunctionWithName:@"vertexMain"];
        pipelineDesc.fragmentFunction = [library newFunctionWithName:@"fragmentMain"];
        pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
        if (!pipeline || !queue) { fprintf(stderr,"pipeline: %s\n",error.description.UTF8String); return 2; }
        MTLTextureDescriptor *targetDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
        targetDesc.storageMode = MTLStorageModePrivate;
        targetDesc.usage = MTLTextureUsageRenderTarget;
        id<MTLTexture> target = [device newTextureWithDescriptor:targetDesc];
        id<MTLBuffer> readback = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        if (!target || !readback) return 3;
        for (unsigned round = 0; round < 100; round++) {
            @autoreleasepool {
                MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:16 height:16 mipmapped:NO];
                desc.storageMode = MTLStorageModeMemoryless;
                desc.usage = MTLTextureUsageRenderTarget;
                id<MTLTexture> temporary = [device newTextureWithDescriptor:desc];
                if (!temporary) return 4;
                MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[0].texture = temporary;
                pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
                pass.colorAttachments[0].clearColor = MTLClearColorMake(0,1,0,1);
                id<MTLCommandBuffer> command = [queue commandBuffer];
                [[command renderCommandEncoderWithDescriptor:pass] endEncoding];
                if (!finish(command)) return 5;
            }
            @autoreleasepool {
                MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:17 height:19 mipmapped:NO];
                desc.storageMode = MTLStorageModeManaged;
                desc.usage = MTLTextureUsageShaderRead;
                id<MTLTexture> sampled = [device newTextureWithDescriptor:desc];
                if (!sampled) return 6;
                uint16_t texels[17*19*4] = {0};
                for (unsigned i=0;i<17*19;i++) {texels[i*4]=0x3c00;texels[i*4+3]=0x3c00;}
                [sampled replaceRegion:MTLRegionMake2D(0,0,17,19) mipmapLevel:0 withBytes:texels bytesPerRow:136];
                MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
                pass.colorAttachments[0].texture = target;
                pass.colorAttachments[0].loadAction = MTLLoadActionClear;
                pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLRenderCommandEncoder> render = [command renderCommandEncoderWithDescriptor:pass];
                [render setRenderPipelineState:pipeline];
                [render setFragmentTexture:sampled atIndex:0];
                [render drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                [render endEncoding];
                id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
                [blit copyFromTexture:target sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0)
                    sourceSize:MTLSizeMake(16,16,1) toBuffer:readback destinationOffset:0 destinationBytesPerRow:256 destinationBytesPerImage:4096];
                [blit endEncoding];
                if (!finish(command)) return 7;
                const uint8_t *pixels = readback.contents;
                for (unsigned y=0;y<16;y++) for(unsigned x=0;x<16;x++) {
                    unsigned i=y*256+x*4;
                    if(pixels[i]!=255||pixels[i+1]||pixels[i+2]||pixels[i+3]!=255) return 8;
                }
                printf("{\"round\":%u,\"passed\":true}\n",round);fflush(stdout);
            }
        }
        return 0;
    }
}
