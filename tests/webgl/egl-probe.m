// Authored initialization probe for the ANGLE library actually shipped in the guest.
#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>

typedef void *EglDisplay;
typedef int32_t EglInt;
typedef unsigned EglBoolean;
typedef EglDisplay (*GetDisplay)(void *);
typedef EglDisplay (*GetPlatformDisplay)(unsigned, void *, const EglInt *);
typedef EglBoolean (*Initialize)(EglDisplay, EglInt *, EglInt *);
typedef EglBoolean (*Terminate)(EglDisplay);
typedef EglInt (*GetError)(void);
typedef const char *(*QueryString)(EglDisplay, EglInt);
typedef void *(*GetProcAddress)(const char *);
typedef EglBoolean (*ChooseConfig)(EglDisplay, const EglInt *, void **, EglInt, EglInt *);
typedef void *(*CreateSurface)(EglDisplay, void *, const EglInt *);
typedef void *(*CreateContext)(EglDisplay, void *, void *, const EglInt *);
typedef EglBoolean (*MakeCurrent)(EglDisplay, void *, void *, void *);
typedef EglBoolean (*Destroy)(EglDisplay, void *);
typedef const unsigned char *(*GlString)(unsigned);

static void *entry(void *library, const char *standard, const char *webkit) {
    void *found = dlsym(library, webkit);
    return found ? found : dlsym(library, standard);
}

static NSDictionary *renderer(void *library, EglDisplay display, GetError error) {
    ChooseConfig choose = (ChooseConfig)entry(library, "eglChooseConfig", "EGL_ChooseConfig");
    CreateSurface surface = (CreateSurface)entry(library, "eglCreatePbufferSurface", "EGL_CreatePbufferSurface");
    CreateContext context = (CreateContext)entry(library, "eglCreateContext", "EGL_CreateContext");
    MakeCurrent current = (MakeCurrent)entry(library, "eglMakeCurrent", "EGL_MakeCurrent");
    Destroy dropContext = (Destroy)entry(library, "eglDestroyContext", "EGL_DestroyContext");
    Destroy dropSurface = (Destroy)entry(library, "eglDestroySurface", "EGL_DestroySurface");
    GlString string = (GlString)entry(library, "glGetString", "GL_GetString");
    if (!choose || !surface || !context || !current || !dropContext || !dropSurface || !string)
        return @{@"error": @"Missing GLES context/query entry point"};
    // EGL_SURFACE_TYPE/PBUFFER_BIT, EGL_RENDERABLE_TYPE/OPENGL_ES2_BIT.
    const EglInt configAttributes[] = {0x3033, 1, 0x3040, 4, 0x3038};
    const EglInt surfaceAttributes[] = {0x3057, 16, 0x3056, 16, 0x3038};
    const EglInt contextAttributes[] = {0x3098, 2, 0x3038};
    void *config = NULL;
    EglInt count = 0;
    if (!choose(display, configAttributes, &config, 1, &count) || !count)
        return @{@"error": @"No GLES2 pbuffer config", @"egl_error": @(error())};
    void *pbuffer = surface(display, config, surfaceAttributes);
    void *ctx = context(display, config, NULL, contextAttributes);
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    if (pbuffer && ctx && current(display, pbuffer, pbuffer, ctx)) {
        const unsigned char *r = string(0x1f01), *v = string(0x1f02);
        result[@"renderer"] = r ? @((const char *)r) : @"";
        result[@"version"] = v ? @((const char *)v) : @"";
    } else result[@"error"] = @"GLES context/current failed";
    result[@"egl_error"] = @(error());
    current(display, NULL, NULL, NULL);
    if (ctx) dropContext(display, ctx);
    if (pbuffer) dropSurface(display, pbuffer);
    return result;
}

static NSDictionary *initialize(void *library, EglDisplay display, Initialize init, Terminate end,
                                GetError error, QueryString query) {
    EglInt major = 0, minor = 0;
    EglBoolean ok = init(display, &major, &minor);
    EglInt code = error();
    NSMutableDictionary *result = [@{
        @"display_created": @(display != NULL), @"initialized": @(ok != 0),
        @"egl_error": @(code), @"major": @(major), @"minor": @(minor)
    } mutableCopy];
    if (ok) {
        const char *vendor = query(display, 0x3053); // EGL_VENDOR
        const char *version = query(display, 0x3054); // EGL_VERSION
        result[@"vendor"] = vendor ? @(vendor) : @"";
        result[@"version"] = version ? @(version) : @"";
        result[@"gles"] = renderer(library, display, error);
        end(display);
    }
    return result;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 3 || (strcmp(argv[2], "metal") && strcmp(argv[2], "default"))) {
            fputs("usage: egl-probe /path/to/ANGLE-or-libEGL metal|default\n", stderr); return 2;
        }
        NSMutableDictionary *result = [@{@"library": @(argv[1]), @"backend": @(argv[2]), @"passed": @NO} mutableCopy];
        // Bind this ANGLE image, not a similarly named entry point in its dependencies.
        void *library = dlopen(argv[1], RTLD_LOCAL | RTLD_NOW | RTLD_FIRST);
        if (!library) {
            result[@"stage"] = @"dlopen";
            result[@"error"] = @(dlerror());
        } else {
            GetDisplay get = (GetDisplay)entry(library, "eglGetDisplay", "EGL_GetDisplay");
            Initialize init = (Initialize)entry(library, "eglInitialize", "EGL_Initialize");
            Terminate end = (Terminate)entry(library, "eglTerminate", "EGL_Terminate");
            GetError error = (GetError)entry(library, "eglGetError", "EGL_GetError");
            QueryString query = (QueryString)entry(library, "eglQueryString", "EGL_QueryString");
            GetProcAddress proc = (GetProcAddress)entry(library, "eglGetProcAddress", "EGL_GetProcAddress");
            GetPlatformDisplay platform = (GetPlatformDisplay)entry(library, "eglGetPlatformDisplayEXT", "EGL_GetPlatformDisplayEXT");
            if (!platform && proc) platform = (GetPlatformDisplay)proc("eglGetPlatformDisplayEXT");
            if (!get || !init || !end || !error || !query || !platform) {
                result[@"stage"] = @"symbols";
                result[@"error"] = @"A required EGL entry point is missing";
            } else {
                result[@"stage"] = @"initialize";
                const char *extensions = query(NULL, 0x3055); // EGL_EXTENSIONS
                result[@"metal_platform_advertised"] = @(extensions && strstr(extensions, "EGL_ANGLE_platform_angle_metal"));
                error();
                // Each process opens one backend: display caching must not choose its peer's.
                const EglInt attributes[] = {0x3203, 0x3489, 0x3038};
                EglDisplay display = !strcmp(argv[2], "metal") ? platform(0x3202, NULL, attributes) : get(NULL);
                result[@"display_error"] = @(error());
                NSDictionary *selected = initialize(library, display, init, end, error, query);
                result[@"result"] = selected;
                NSDictionary *gles = selected[@"gles"];
                NSString *rendererName = gles[@"renderer"];
                BOOL valid = [selected[@"initialized"] boolValue] && rendererName.length &&
                    [gles[@"version"] length] && [gles[@"egl_error"] integerValue] == 0x3000;
                if (!strcmp(argv[2], "metal")) valid &= [rendererName containsString:@"ANGLE Metal Renderer"];
                result[@"passed"] = @(valid);
            }
            dlclose(library);
        }
        NSData *json = [NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil];
        puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
        return [result[@"passed"] boolValue] ? 0 : 1;
    }
}
