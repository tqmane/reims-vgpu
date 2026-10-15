# Guest WebGL acceptance

Run the authored inputs in the real guest without GPU feature overrides:

```powershell
python tests/webgl/server.py --output .work/diagnostics/webgl --port 9087
```

From QEMU's NAT guest, open
`http://10.0.2.2:9087/index.html?run=candidate-safari&browser=safari`.
Use another unique run ID for Chrome and for the old engine. The collector
listens only on host loopback, accepts bounded JSON with matching run IDs,
and writes progress after every stage. A lost context is a failure even if
`getContext` returned an object. Keep the page in front for the 180-frame run.

Both WebGL versions must pass all stages and show the rendered triangle.
Save a QMP screendump and correlate its run/timestamp with the JSON, guest
GPU-process PIDs/start times, crash reports, host/serial logs and QMP status.
The page's context-stability check does not independently observe OS process
restarts. Do not publish those private captures or guest-generated shaders.

For native Metal diagnosis, copy `metal-probe.m` to the validation guest and
compile it there with `clang -fobjc-arc -fblocks -framework Foundation
-framework Metal metal-probe.m -o metal-probe`. Run it and save its JSON.
The probe compiles an authored kernel dynamically and checks GPU readback;
its success is not a WebGL pass. Do not copy compiled guest artifacts into Git.

`docs/WEBGL_DIAGNOSIS.md` records the initialization blocker. Host-side fixture
validation is useful but cannot replace the real guest's pixel assertions.

`egl-probe.m` loads the ANGLE/libEGL library supplied as its sole argument
and records normal EGL initialization and the Metal backend's initialization
separately. Compile with `clang -fobjc-arc -framework Foundation egl-probe.m
-o egl-probe`. It changes no feature flags or guest libraries. Save its JSON
and stderr to private diagnostics. Initialization success alone is not a
WebGL pixel test.
