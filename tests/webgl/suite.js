/* Authored GLES inputs; no browser, guest or Apple shader binaries are saved. */
'use strict';
import {isSoftwareRenderer} from './renderer.mjs';
const params = new URLSearchParams(location.search);
const report = {schema:1, run:params.get('run') || 'manual', browser:params.get('browser') || 'unknown',
  started:new Date().toISOString(), userAgent:navigator.userAgent, tests:[], contexts:[]};
const assert = (ok, message) => { if (!ok) throw new Error(message); };
const timeout = (p, ms, message) => new Promise((resolve, reject) => {
  const id=setTimeout(()=>reject(new Error(message)),ms);
  p.then(v=>{clearTimeout(id);resolve(v)},e=>{clearTimeout(id);reject(e)});
});
async function publish() {
  document.querySelector('#results').textContent=JSON.stringify(report,null,2);
  const response=await fetch('/results/'+encodeURIComponent(report.run),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(report)});
  assert(response.ok,'Result collector rejected the report');
}
async function check(api, name, run) {
  report.active={api,name};await publish();
  const start=performance.now();
  try { const detail=await run(); report.tests.push({api,name,passed:true,milliseconds:performance.now()-start,detail}); }
  catch(e) { report.tests.push({api,name,passed:false,milliseconds:performance.now()-start,error:String(e)}); }
  await publish();
}
function pixel(gl,x,y,expected) {
  const p=new Uint8Array(4);gl.readPixels(x,y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,p);
  assert(gl.getError()===gl.NO_ERROR,'readPixels returned a GL error');
  assert(expected.every((v,i)=>Math.abs(p[i]-v)<=2),`pixel ${Array.from(p)} != ${expected}`);
  return Array.from(p);
}
function program(gl, modern, textured=false) {
  const head=modern?'#version 300 es\n':'';
  const v=head+(modern?'in':'attribute')+' vec3 position;\n'+(modern?'out':'varying')+' vec2 uv;\nvoid main(){gl_Position=vec4(position,1.0);uv=position.xy*.5+.5;}';
  const f=head+'precision highp float;\n'+(modern?'in':'varying')+' vec2 uv;uniform vec4 color;uniform sampler2D image;\n'+(modern?'out vec4 outputColor;\n':'')+
    'void main(){'+(modern?'outputColor':'gl_FragColor')+'='+(textured?(modern?'texture':'texture2D')+'(image,uv)':'color')+';}';
  const p=gl.createProgram();
  for (const [type,src] of [[gl.VERTEX_SHADER,v],[gl.FRAGMENT_SHADER,f]]) {
    const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);
    const ok=gl.getShaderParameter(s,gl.COMPILE_STATUS), log=gl.getShaderInfoLog(s);
    if(!ok){gl.deleteShader(s);gl.deleteProgram(p);throw new Error('shader: '+log);}
    gl.attachShader(p,s);gl.deleteShader(s);
  }
  gl.linkProgram(p);
  const ok=gl.getProgramParameter(p,gl.LINK_STATUS), log=gl.getProgramInfoLog(p);
  if(!ok){gl.deleteProgram(p);throw new Error('link: '+log);}
  return p;
}
async function context(api) {
  const canvas=document.getElementById(api), modern=api==='webgl2';
  canvas.scrollIntoView({block:'center'});
  let creationError='', unexpectedLosses=0, deliberateLoss=false;
  canvas.addEventListener('webglcontextcreationerror',e=>creationError=e.statusMessage);
  canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();if(!deliberateLoss)unexpectedLosses++;});
  report.active={api,name:'context'};await publish();
  const gl=canvas.getContext(api,{alpha:false,depth:true,antialias:false,preserveDrawingBuffer:true});
  await check(api,'context',()=>{
    assert(gl,'getContext returned null: '+creationError);
    const debug=gl.getExtension('WEBGL_debug_renderer_info');
    const info={api,version:gl.getParameter(gl.VERSION),renderer:gl.getParameter(gl.RENDERER),
      unmaskedRenderer:debug?gl.getParameter(debug.UNMASKED_RENDERER_WEBGL):null,
      lost:gl.isContextLost(),error:gl.getError(),creationError};
    report.contexts.push(info);
    assert(!info.lost && info.error===gl.NO_ERROR && info.version!==null,'Context failed during initialization: '+JSON.stringify(info));
    assert(info.unmaskedRenderer,'Renderer identity unavailable; hardware acceleration is unverified');
    assert(!isSoftwareRenderer(info.unmaskedRenderer),'Software fallback is not GPU validation');return info;
  });
  if(!gl || gl.isContextLost()) return;
  let solid,buffer;
  function draw(color=[1,0,0,1],z=0,p=solid,full=false) {
    gl.useProgram(p);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    const points=full?[-1,-1,z,3,-1,z,-1,3,z]:[-.8,-.8,z,.8,-.8,z,0,.8,z];
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(points),gl.STREAM_DRAW);
    const a=gl.getAttribLocation(p,'position');gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.FLOAT,false,0,0);
    gl.uniform4fv(gl.getUniformLocation(p,'color'),color);gl.drawArrays(gl.TRIANGLES,0,3);
    assert(gl.getError()===gl.NO_ERROR,'drawArrays failed');
  }
  function reset() {
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.viewport(0,0,canvas.width,canvas.height);
    gl.disable(gl.BLEND);gl.disable(gl.DEPTH_TEST);gl.depthMask(true);gl.clearDepth(1);
    gl.clearColor(0,0,1,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  }
  await check(api,'shader_compile_link',()=>{solid=program(gl,modern);buffer=gl.createBuffer();});
  if(!solid) return;
  await check(api,'triangle_and_vertex_buffer',()=>{reset();draw();pixel(gl,320,180,[255,0,0,255]);return pixel(gl,0,359,[0,0,255,255]);});
  await check(api,'texture_upload_readback',()=>{
    const tex=gl.createTexture(), p=program(gl,modern,true);
    try {
      reset();gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,tex);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,2,2,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([0,255,0,255,0,255,0,255,0,255,0,255,0,255,0,255]));
      draw([1,1,1,1],0,p,true);return pixel(gl,320,180,[0,255,0,255]);
    } finally {gl.deleteTexture(tex);gl.deleteProgram(p);}
  });
  await check(api,'fbo_render_readpixels',()=>{
    const tex=gl.createTexture(),fbo=gl.createFramebuffer();
    try {
      gl.bindTexture(gl.TEXTURE_2D,tex);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,64,64,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
      gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
      assert(gl.checkFramebufferStatus(gl.FRAMEBUFFER)===gl.FRAMEBUFFER_COMPLETE,'FBO incomplete');
      gl.viewport(0,0,64,64);gl.clearColor(0,0,1,1);gl.clear(gl.COLOR_BUFFER_BIT);draw();return pixel(gl,32,32,[255,0,0,255]);
    } finally {gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.deleteFramebuffer(fbo);gl.deleteTexture(tex);}
  });
  await check(api,'blend',()=>{reset();gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);draw([1,0,0,.5]);return pixel(gl,320,180,[128,0,127,255]);});
  await check(api,'depth',()=>{reset();gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LESS);draw([1,0,0,1],.2);draw([0,1,0,1],.8);pixel(gl,320,180,[255,0,0,255]);draw([0,1,0,1],-.2);return pixel(gl,320,180,[0,255,0,255]);});
  await check(api,'repeated_frames',async()=>{
    reset();draw([1,0,0,1],0,solid,true);
    const colorLocation=gl.getUniformLocation(solid,'color'), samples=[], cpuSamples=[];
    let previous,frames=0,finalColor,request;
    try {
      await timeout(new Promise((resolve,reject)=>{
        function frame(t){try{
          const begin=performance.now();
          if(previous!==undefined)samples.push(t-previous);previous=t;
          finalColor=frames%2?[1,0,0,1]:[0,1,0,1];
          gl.uniform4fv(colorLocation,finalColor);gl.drawArrays(gl.TRIANGLES,0,3);frames++;
          cpuSamples.push(performance.now()-begin);
          if(frames===180)resolve();else request=requestAnimationFrame(frame);
        }catch(e){reject(e)}}request=requestAnimationFrame(frame);
      }),30000,'Repeated frames timed out');
    } catch(e) { throw new Error(`${e.message}; ${frames}/180 frames; ${document.visibilityState}; max callback ${Math.max(0,...cpuSamples)}ms`); }
    finally { if(request!==undefined)cancelAnimationFrame(request); }
    pixel(gl,320,180,finalColor.map(x=>x*255));const ordered=[...samples].sort((a,b)=>a-b);
    return {frames,raf_fps:1000*samples.length/samples.reduce((a,b)=>a+b,0),median_ms:ordered[Math.floor(ordered.length*.5)],p95_ms:ordered[Math.floor(ordered.length*.95)],p99_ms:ordered[Math.floor(ordered.length*.99)],metric:'requestAnimationFrame intervals, not physical display presents'};
  });
  await check(api,'resource_lifetime',()=>{
    for(let i=0;i<100;i++){const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,4096,gl.STREAM_DRAW);gl.deleteBuffer(b);}
    draw();return pixel(gl,320,180,[255,0,0,255]);
  });
  await check(api,'context_loss_recovery',async()=>{
    const ext=gl.getExtension('WEBGL_lose_context');assert(ext,'WEBGL_lose_context unavailable');
    deliberateLoss=true;
    const lost=new Promise(resolve=>canvas.addEventListener('webglcontextlost',resolve,{once:true}));
    ext.loseContext();await timeout(lost,5000,'No context-loss event');
    // WebKit records preventDefault after dispatch finishes, beyond this microtask.
    await new Promise(resolve=>setTimeout(resolve,0));
    const restored=new Promise(resolve=>canvas.addEventListener('webglcontextrestored',resolve,{once:true}));
    ext.restoreContext();await timeout(restored,10000,'Context did not restore');
    deliberateLoss=false;solid=program(gl,modern);buffer=gl.createBuffer();reset();draw();return pixel(gl,320,180,[255,0,0,255]);
  });
  await check(api,'context_stability',()=>{assert(unexpectedLosses===0&&!gl.isContextLost(),'Unexpected context loss');return {unexpectedLosses};});
  gl.deleteProgram(solid);gl.deleteBuffer(buffer);
}
(async()=>{
  try {await context('webgl');await context('webgl2');report.finished=new Date().toISOString();report.passed=report.tests.every(x=>x.passed);}
  catch(e){report.fatal=String(e);report.passed=false;}
  document.querySelector('#status').textContent=report.passed?'PASS — both APIs rendered and returned expected pixels':'FAIL — inspect the recorded stage';
  document.querySelector('#status').className=report.passed?'pass':'fail';await publish();
})();
