import assert from 'node:assert/strict';
import {isSoftwareRenderer} from './renderer.mjs';

for (const name of ['SwiftShader Device', 'llvmpipe (LLVM)', 'softpipe',
  'ANGLE (Apple Inc., Apple Software Renderer, OpenGL 4.1)', 'Software Rasterizer']) {
  assert.equal(isSoftwareRenderer(name), true, name);
}
for (const name of ['ANGLE Metal Renderer: Apple Paravirtual device',
  'NVIDIA GeForce RTX 4070 SUPER', 'Apple M3 Max', null]) {
  assert.equal(isSoftwareRenderer(name), false, name);
}
console.log('Known software renderers are rejected; hardware names remain distinct.');
