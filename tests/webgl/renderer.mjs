export function isSoftwareRenderer(name) {
  return /swiftshader|llvmpipe|softpipe|apple software renderer|software rasterizer/i.test(name || '');
}
