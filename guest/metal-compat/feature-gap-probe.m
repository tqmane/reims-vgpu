// Authored offscreen Metal feature probes. No guest shaders or assets.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s astc|half-fast|half-rounded|float-fast|float-rounded\n", argv[0]);
            return 2;
        }
        setbuf(stdout, NULL);
        NSString *mode = @(argv[1]);
        NSSet *modes = [NSSet setWithArray:@[@"astc", @"half-fast", @"half-rounded", @"float-fast", @"float-rounded"]];
        if (![modes containsObject:mode]) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "No Metal device\n"); return 1; }
        printf("probe=%s device=%s\n", argv[1], device.name.UTF8String);
        if ([mode isEqualToString:@"astc"]) {
            MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatASTC_8x8_LDR width:8 height:8 mipmapped:NO];
            d.storageMode = MTLStorageModePrivate;
            d.usage = MTLTextureUsageShaderRead;
            printf("creating one ASTC 8x8 LDR texture; no upload or draw\n");
            id<MTLTexture> texture = [device newTextureWithDescriptor:d];
            printf("texture=%s\n", texture ? "created" : "nil");
            if (!texture) return 1;
            // Force deferred host resource creation. Texels are unspecified;
            // this only tests resource acceptance, never their contents.
            id<MTLBuffer> buffer = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(8,8,1) toBuffer:buffer destinationOffset:0 destinationBytesPerRow:256 destinationBytesPerImage:256];
            [blit endEncoding];
            printf("submitting one texture-to-buffer copy to force host creation\n");
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr, "copy failed: %s\n", command.error.description.UTF8String);
                return 1;
            }
            printf("copy completed; contents intentionally not checked\n");
            return 0;
        }
        BOOL half = [mode hasPrefix:@"half-"];
        NSString *type = half ? @"half4" : @"float4";
        NSString *source = [NSString stringWithFormat:
            @"#include <metal_stdlib>\nusing namespace metal;\n"
             "vertex float4 probe_vertex(uint i [[vertex_id]]) {"
             "float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};"
             "return float4(p[i],0,1);}\n"
             "struct Out { %@ color [[color(1)]]; };\n"
             "fragment Out probe_fragment(%@ previous [[color(1)]]) {"
             "Out o; o.color=previous * %@(0.25) + %@ (0.125,0.25,0.375,0.5); return o;}\n",
             type, type, type, type];
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = [mode hasSuffix:@"fast"] ? MTLMathModeFast : MTLMathModeSafe;
        options.languageVersion = MTLLanguageVersion2_3;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) { fprintf(stderr, "compile: %s\n", error.description.UTF8String); return 1; }
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [library newFunctionWithName:@"probe_vertex"];
        d.fragmentFunction = [library newFunctionWithName:@"probe_fragment"];
        d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        d.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
        MTLRenderPipelineColorAttachmentDescriptor *a = d.colorAttachments[1];
        a.pixelFormat = MTLPixelFormatRGBA16Float;
        a.blendingEnabled = YES;
        a.sourceRGBBlendFactor = a.sourceAlphaBlendFactor = MTLBlendFactorOne;
        a.destinationRGBBlendFactor = a.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        a.rgbBlendOperation = a.alphaBlendOperation = MTLBlendOperationAdd;
        printf("creating RGBA16Float slot-1 fetch pipeline with source-over blending\n");
        id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!pipeline) { fprintf(stderr, "pipeline: %s\n", error.description.UTF8String); return 1; }
        printf("pipeline created; submitting one offscreen draw\n");
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:8 height:8 mipmapped:NO];
        td.storageMode = MTLStorageModeManaged;
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        id<MTLTexture> target = [device newTextureWithDescriptor:td];
        td.pixelFormat = MTLPixelFormatBGRA8Unorm;
        id<MTLTexture> unused = [device newTextureWithDescriptor:td];
        if (!target || !unused) return 1;
        MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = unused;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
        pass.colorAttachments[1].texture = target;
        pass.colorAttachments[1].loadAction = MTLLoadActionClear;
        pass.colorAttachments[1].storeAction = MTLStoreActionStore;
        pass.colorAttachments[1].clearColor = MTLClearColorMake(0.2,0.4,0.6,0.8);
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
        [encoder setRenderPipelineState:pipeline];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [encoder endEncoding];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        [blit synchronizeResource:target];
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "command failed: %s\n", command.error.description.UTF8String);
            return 1;
        }
        uint16_t pixel[4] = {0};
        [target getBytes:pixel bytesPerRow:sizeof(pixel) fromRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0];
        printf("draw completed; first pixel half bits=%04x,%04x,%04x,%04x (diagnostic, not an accuracy assertion)\n", pixel[0], pixel[1], pixel[2], pixel[3]);
        return 0;
    }
}
