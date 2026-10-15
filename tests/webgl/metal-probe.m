// Authored dynamic Metal compilation and GPU/CPU readback probe.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(void) {
    @autoreleasepool {
        NSMutableDictionary *result = [NSMutableDictionary dictionary];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        result[@"device"] = device.name ?: @"none";
        result[@"mac1"] = @([device supportsFamily:MTLGPUFamilyMac1]);
        result[@"mac2"] = @([device supportsFamily:MTLGPUFamilyMac2]);
        result[@"argument_buffers_tier"] = @(device.argumentBuffersSupport);
        result[@"read_write_texture_tier"] = @(device.readWriteTextureSupport);
        result[@"raster_order_groups"] = @(device.rasterOrderGroupsSupported);
        result[@"programmable_sample_positions"] = @(device.programmableSamplePositionsSupported);
        result[@"shader_barycentrics"] = @(device.supportsShaderBarycentricCoordinates);
        NSMutableDictionary *families = [NSMutableDictionary dictionary];
        for (NSInteger family=1001; family<=1010; family++) families[@(family).stringValue]=@([device supportsFamily:(MTLGPUFamily)family]);
        result[@"apple_families"] = families;
        result[@"registry_id"] = @(device.registryID);
        result[@"headless"] = @(device.headless);
        result[@"passed"] = @NO;
        NSError *error = nil;
        NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void verify(device uint *out [[buffer(0)]], uint i [[thread_position_in_grid]]) {"
            "out[i]=0x12340000u+i;}";
        CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        result[@"source_compile_ms"] = @((CFAbsoluteTimeGetCurrent()-start)*1000);
        if (!library) {
            result[@"stage"] = @"source_compile";
            result[@"error"] = error.localizedDescription ?: @"No Metal device/library";
        } else {
            id<MTLFunction> function = [library newFunctionWithName:@"verify"];
            start = CFAbsoluteTimeGetCurrent();
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            result[@"pipeline_ms"] = @((CFAbsoluteTimeGetCurrent()-start)*1000);
            if (!pipeline) {
                result[@"stage"] = @"pipeline";
                result[@"error"] = error.localizedDescription ?: @"No pipeline";
            } else {
                id<MTLBuffer> output = [device newBufferWithLength:4096 options:MTLResourceStorageModeShared];
                id<MTLCommandQueue> queue = [device newCommandQueue];
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:output offset:0 atIndex:0];
                [encoder dispatchThreadgroups:MTLSizeMake(4,1,1) threadsPerThreadgroup:MTLSizeMake(16,1,1)];
                [encoder endEncoding];
                dispatch_semaphore_t done = dispatch_semaphore_create(0);
                [command addCompletedHandler:^(id<MTLCommandBuffer> _) {dispatch_semaphore_signal(done);}];
                start = CFAbsoluteTimeGetCurrent();
                [command commit];
                long expired = dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 30*NSEC_PER_SEC));
                result[@"gpu_completion_ms"] = @((CFAbsoluteTimeGetCurrent()-start)*1000);
                result[@"stage"] = @"readback";
                BOOL correct = !expired && command.status == MTLCommandBufferStatusCompleted && output;
                if (correct) {
                    const uint32_t *values = output.contents;
                    for (uint32_t i=0; i<64; i++) if (values[i] != 0x12340000u+i) correct=NO;
                }
                result[@"passed"] = @(correct);
                result[@"error"] = expired ? @"GPU completion timeout" : (command.error.localizedDescription ?: @"");
            }
        }
        NSData *json = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil];
        puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
        return [result[@"passed"] boolValue] ? 0 : 1;
    }
}
