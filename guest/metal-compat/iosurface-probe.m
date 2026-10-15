#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
int main(void) {
 @autoreleasepool {
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  IOSurfaceRef surface=IOSurfaceCreate((__bridge CFDictionaryRef)@{
   (id)kIOSurfaceWidth:@16,(id)kIOSurfaceHeight:@17,(id)kIOSurfaceBytesPerElement:@4,
   (id)kIOSurfaceBytesPerRow:@64,(id)kIOSurfaceAllocSize:@1088,(id)kIOSurfacePixelFormat:@(0x42475241)});
  if(!surface)return 10;
  MTLTextureDescriptor *descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:16 height:17 mipmapped:NO];
  descriptor.storageMode=MTLStorageModeShared;
  descriptor.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
  id<MTLTexture> texture=[device newTextureWithDescriptor:descriptor iosurface:surface plane:0];
  if(!texture){fprintf(stderr,"texture allocation failed\n");return 2;}
  fprintf(stderr,"texture storage=%lu unified-memory=%d\n",(unsigned long)texture.storageMode,device.hasUnifiedMemory);
  id<MTLCommandQueue> queue=[device newCommandQueue];
  id<MTLCommandBuffer> command=[queue commandBuffer];
  MTLRenderPassDescriptor *pass=[MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture=texture;
  pass.colorAttachments[0].loadAction=MTLLoadActionClear;
  pass.colorAttachments[0].storeAction=MTLStoreActionStore;
  pass.colorAttachments[0].clearColor=MTLClearColorMake(1,0,0,1);
  id<MTLRenderCommandEncoder> encoder=[command renderCommandEncoderWithDescriptor:pass];
  if(!queue||!command||!encoder){fprintf(stderr,"render encoder failed\n");return 3;}
  [encoder endEncoding];
  dispatch_semaphore_t done=dispatch_semaphore_create(0);
  [command addCompletedHandler:^(id<MTLCommandBuffer> _){dispatch_semaphore_signal(done);}];
  [command commit];
  if(dispatch_semaphore_wait(done,dispatch_time(DISPATCH_TIME_NOW,10*NSEC_PER_SEC)))return 4;
  if(command.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"render: %s\n",command.error.description.UTF8String);return 5;}
  id<MTLBuffer> staging=[device newBufferWithLength:8192 options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> copy=[queue commandBuffer];
  id<MTLBlitCommandEncoder> blit=[copy blitCommandEncoder];
  if(!staging||!copy||!blit)return 7;
  [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(16,17,1) toBuffer:staging destinationOffset:0 destinationBytesPerRow:256 destinationBytesPerImage:8192];
  [blit endEncoding];
  dispatch_semaphore_t copied=dispatch_semaphore_create(0);
  [copy addCompletedHandler:^(id<MTLCommandBuffer> _){dispatch_semaphore_signal(copied);}];
  [copy commit];
  if(dispatch_semaphore_wait(copied,dispatch_time(DISPATCH_TIME_NOW,10*NSEC_PER_SEC))||copy.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"blit: %s\n",copy.error.description.UTF8String);return 8;}
  const uint8_t *readback=staging.contents;
  for(unsigned y=0;y<17;y++)for(unsigned x=0;x<16;x++)if(readback[y*256+x*4]!=0||readback[y*256+x*4+1]!=0||readback[y*256+x*4+2]!=255||readback[y*256+x*4+3]!=255){fprintf(stderr,"blit pixel %u,%u: %u %u %u %u\n",x,y,readback[y*256+x*4],readback[y*256+x*4+1],readback[y*256+x*4+2],readback[y*256+x*4+3]);return 9;}
  puts("{\"stage\":\"iosurface-render-blit-readback\",\"pixels\":272,\"passed\":true}");fflush(stdout);
  IOReturn locked=IOSurfaceLock(surface,kIOSurfaceLockReadOnly,NULL);
  if(locked!=kIOReturnSuccess)return 11;
  const uint8_t *pixels=IOSurfaceGetBaseAddress(surface);
  size_t stride=IOSurfaceGetBytesPerRow(surface);BOOL correct=YES;
  for(unsigned y=0;y<17;y++)for(unsigned x=0;x<16;x++)if(pixels[y*stride+x*4]!=0||pixels[y*stride+x*4+1]!=0||pixels[y*stride+x*4+2]!=255||pixels[y*stride+x*4+3]!=255)correct=NO;
  IOSurfaceUnlock(surface,kIOSurfaceLockReadOnly,NULL);
  CFRelease(surface);
  if(!correct){fputs("IOSurface CPU pixel mismatch\n",stderr);return 6;}
  puts("{\"stage\":\"iosurface-lock-cpu-readback\",\"pixels\":272,\"passed\":true}");
  return 0;
 }
}
