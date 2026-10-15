#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 2;
        unsigned completed = 0;
        for (unsigned round = 0; round < 100; round++) {
            @autoreleasepool {
                NSMutableArray *buffers = [NSMutableArray array];
                for (unsigned index = 0; index < 32; index++) {
                    NSUInteger size = 32 + 128 * index;
                    id<MTLBuffer> buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
                    if (!buffer || buffer.length != size || !buffer.contents) return 3;
                    memset(buffer.contents, index + round, size);
                    [buffers addObject:buffer];
                }
                for (unsigned index = 0; index < buffers.count; index++) {
                    id<MTLBuffer> buffer = buffers[index];
                    const uint8_t *bytes = buffer.contents;
                    for (NSUInteger offset = 0; offset < buffer.length; offset++)
                        if (bytes[offset] != (uint8_t)(index + round)) return 4;
                }
                MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:16 height:16 mipmapped:NO];
                desc.storageMode = MTLStorageModeManaged;
                desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
                id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
                if (!texture || texture.width != 16 || texture.height != 16) return 5;
                completed++;
            }
        }
        NSError *error = nil;
        NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\nkernel void fill(device uint *out [[buffer(0)]],uint i [[thread_position_in_grid]]) { out[i] = i + 7; }";
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"fill"];
        id<MTLComputePipelineState> pipeline = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
        NSDictionary *result = @{@"resource_rounds":@(completed), @"buffers_per_round":@32,
            @"library":@(library != nil), @"pipeline":@(pipeline != nil),
            @"passed":@(completed == 100 && pipeline != nil)};
        NSData *json = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil];
        puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
        if (error) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
        return pipeline ? 0 : 6;
    }
}
