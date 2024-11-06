// exec_via_dlopen
// Mark Mentovai
// 2024-11-06

// clang -Wall -Wextra -Werror -o exec_via_dlopen exec_via_dlopen.c

// Context: https://trac.macports.org/ticket/66358
//
// Runs a Mach-O executable by loading it via `dlopen`, finding its entry point
// (`main` function), and calling it. This is a technique to run Apple-produced
// executables more freely and with more control over the loader environment,
// albeit without Apple’s identity or associated entitlements. Desirable control
// over the loader environment might include setting environment variables such
// as `DYLD_INSERT_LIBRARIES`, which is normally forbidden under the “hardened
// runtime”. For Apple-signed executables, escaping the hardened runtime
// requires removing the signature, and on arm64, removing Apple’s signature may
// render an arm64e executable difficult to run. Apple ships all of its own
// arm64 executables as arm64e. There is, however, no restriction on non-Apple
// code, including programs that do not request the “hardened runtime”, against
// loading (such as via `dlopen`) other Mach-O images that are arm64e or
// Apple-signed.
//
// This approrach is not entirely transparent. Some observable differences are
// listed below. There will be other differences as well, mostly concentrated in
// the interfaces between the program and both the kernel and dyld.
//  - The architecture of this program as run must match the architecture of the
//    program that it executes. For this purpose, arm64 and arm64e are
//    considered equivalent.
//  - `_NSGetExecutablePath` will refer to this program and not the one that it
//    executes.
//  - `NXArgc` and `NXArgv` will refer to this program’s argument vector and not
//    the slightly transformed version passed to the program it executes.
//
// This should work for any position-independent executable (Mach-O filetype
// `MH_EXECUTE` with flags `MH_PIE`), as it is possible to `dlopen` such
// executables. `MH_PIE` is the default for executables when targeting Mac OS X
// 10.7 and later. At some point in the future, this technique may stop working.
// https://github.com/apple-oss-distributions/dyld/blob/dyld-940/dyld/DyldAPIs.cpp#L1248
// (from macOS 12.0.1): “only allow dlopen() of main executables on macOS
// (eventually ban there too)”. This still does work as of macOS 15.1.

#include <AvailabilityMacros.h>
#include <dlfcn.h>
#include <err.h>
#include <libgen.h>
#include <mach-o/loader.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Define HIDDEN_MAIN_ARGS to be 0 to invoke `main` as `main(argc, argv)`
// instead of `main(argc, argv, envp, apple)`.
#if !defined(HIDDEN_MAIN_ARGS)
#define HIDDEN_MAIN_ARGS 1
#endif

#if HIDDEN_MAIN_ARGS
typedef int(MainType)(int, char **, char **, char **);
#else
typedef int(MainType)(int, char **);
#endif

// 64/32-bit compatibility definitions.
#if defined(__LP64__)
typedef struct mach_header_64 MachHeaderType;
static uint32_t const kMhMagic = MH_MAGIC_64;
#else
typedef struct mach_header MachHeaderType;
static uint32_t const kMhMagic = MH_MAGIC;
#endif

// Declarations of useful Apple-private interfaces.
#if defined(MAC_OS_VERSION_13_0) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_13_0
MachHeaderType const * _dyld_get_dlopen_image_header(void *)
#if MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_VERSION_13_0
    __attribute__((weak_import))
#endif
    ;
#endif
typedef MachHeaderType const *(DyldGetDlopenImageHeaderType)(void *);

// Support for older SDKs.
#if !defined(MAC_OS_X_VERSION_10_8) || \
    MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_8
#define LC_MAIN (0x28 | LC_REQ_DYLD)
struct entry_point_command {
  uint32_t cmd;
  uint32_t cmdsize;
  uint64_t entryoff;
  uint64_t stacksize;
};
#endif

static int const kFailureExitStatus = 127;

// Given a `dlopen` handle, returns a pointer to its `MachHeaderType` resident
// in memory, or `NULL` if it cannot be determined.
static MachHeaderType const * GetMachHeaderFromDlHandle(void * dlh) {
#if defined(MAC_OS_VERSION_13_0) && \
    MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_VERSION_13_0
  // Deployment target ≥ macOS 13: direct call is possible.
  return _dyld_get_dlopen_image_header(dlh);
#else
  // Deployment target < macOS 13: verify the symbol was resolved at runtime
  // before calling it.
#if !defined(MAC_OS_VERSION_13_0) || \
    MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_VERSION_13_0
  // SDK < macOS 13: it’s not possible to resolve this symbol at link time, so
  // look it up dynamically at runtime.
  DyldGetDlopenImageHeaderType * _dyld_get_dlopen_image_header =
      (DyldGetDlopenImageHeaderType *)dlsym(RTLD_DEFAULT,
                                            "_dyld_get_dlopen_image_header");
#else
  // SDK ≥ macOS 13. Link directly against the symbol, but verify it’s present
  // at runtime.
#endif
  if (_dyld_get_dlopen_image_header) {
    return _dyld_get_dlopen_image_header(dlh);
  }
#endif

  // Runtime < macOS 13. Absent `_dyld_get_dlopen_image_header`, look up the
  // `__mh_execute_header` symbol in the module. This is a convention for
  // `MH_EXECUTE`-type images, but it’s not required to be present or to resolve
  // to the image base. That said, it will be present and work as desired for
  // images created with standard tools.
  return (MachHeaderType const *)dlsym(dlh, "_mh_execute_header");
}

// Given a `MachHeaderType`, returns a function pointer to its entry point
// resident in memory, which is its `main` function taken from its `LC_MAIN`
// load command. If the entry point can’t be located via this mechanism, returns
// NULL. If there are structural errors in the Mach-O image, exits with
// `kFailureExitStatus` after writing an explanation to `stderr`.
//
// `LC_MAIN` is only supported on OS X 10.8 and later. Executable images
// targeting earlier OS versions do not use `LC_MAIN`, but use `LC_UNIXTHREAD`
// instead, with an initial program counter value traditionally configured to
// enter the `_start` symbol, not the main function, following a different
// calling convention. That form is not handled here.
static MainType * GetEntryPointFromMachHeader(MachHeaderType const * mh) {
  if (mh->magic != kMhMagic) {
    errx(kFailureExitStatus,
         "magic 0x%08x != kMhMagic 0x%08x",
         mh->magic,
         kMhMagic);
  }
  if (mh->filetype != MH_EXECUTE) {
    errx(kFailureExitStatus,
         "filetype 0x%x != MH_EXECUTE 0x%x",
         mh->filetype,
         MH_EXECUTE);
  }

  char const * mh_c = (char const *)mh;
  char const * lc_base_c = (char const *)(mh + 1);

  MainType * entry_point = NULL;
  uint32_t entry_point_lc_index = UINT32_MAX;
  uint32_t command_offset = 0;
  for (uint32_t command_index = 0; command_index < mh->ncmds; ++command_index) {
    if (mh->sizeofcmds - command_offset < sizeof(struct load_command)) {
      errx(kFailureExitStatus,
           "command_index = %u: command_offset = %u < mh->sizeofcmds %u - "
           "sizeof(struct load_command) %zu",
           command_index,
           command_offset,
           mh->sizeofcmds,
           sizeof(struct load_command));
    }

    struct load_command const * lc =
        (struct load_command const *)(lc_base_c + command_offset);
    if (lc->cmdsize < sizeof(struct load_command)) {
      errx(kFailureExitStatus,
           "command_index = %u: cmdsize %u < sizeof(struct load_command) %zu",
           command_index,
           lc->cmdsize,
           sizeof(struct load_command));
    }
    if (lc->cmdsize > mh->sizeofcmds - command_offset) {
      errx(kFailureExitStatus,
           "command_index = %u: cmdsize %u > sizeofcmds %u - command_offset %u",
           command_index,
           lc->cmdsize,
           mh->sizeofcmds,
           command_offset);
    }
    command_offset += lc->cmdsize;

    if (lc->cmd == LC_MAIN) {
      if (lc->cmdsize != sizeof(struct entry_point_command)) {
        errx(kFailureExitStatus,
             "cmdsize %u != sizeof(struct entry_point_command) %zu",
             lc->cmdsize,
             sizeof(struct entry_point_command));
      }

      if (entry_point_lc_index != UINT32_MAX) {
        errx(kFailureExitStatus,
             "duplicate entry point load command at indices %u and %u",
             command_index,
             entry_point_lc_index);
      }
      entry_point_lc_index = command_index;

      struct entry_point_command const * ep =
          (struct entry_point_command const *)lc;
      if (ep->entryoff > UINTPTR_MAX) {
        errx(kFailureExitStatus,
             "entryoff 0x%llx > max 0x%llx",
             ep->entryoff,
             (uint64_t)UINTPTR_MAX);
      }

      entry_point = (MainType *)(mh_c + ep->entryoff);
    }
  }

  if (command_offset != mh->sizeofcmds) {
    errx(kFailureExitStatus,
         "final command_offset %u != sizeofcmds %u",
         command_offset,
         mh->sizeofcmds);
  }

  return entry_point;
}

// Given a `dlopen` handle, returns a function pointer to its entry point
// resident in memory, which is its `main` function. This determined by
// `GetEntryPointFromMachHeader(GetMachHeaderFromDlHandle(dlh))` if possible,
// falling back to `dlsym(dlh, "main")`. If the entry point cannot be
// determined, exits with `kFailureExitStatus` after writing an explanation to
// `stderr`.
static MainType * GetEntryPointFromDlHandle(void * dlh) {
  MainType * entry_point = NULL;

  MachHeaderType const * mh = GetMachHeaderFromDlHandle(dlh);
  if (mh) {
    entry_point = GetEntryPointFromMachHeader(mh);
  }

  if (!entry_point) {
    // This will not work if the symbol table has been stripped.
    entry_point = (MainType *)dlsym(dlh, "main");
    if (!entry_point) {
      errx(kFailureExitStatus, "%s", dlerror());
    }
  }

  return entry_point;
}

#if HIDDEN_MAIN_ARGS
int main(int argc, char * argv[], char * envp[], char * apple[]) {
#else
int main(int argc, char * argv[]) {
#endif
  if (argc < 2) {
    fprintf(stderr, "usage: %s command [argument ...]\n", basename(argv[0]));
    return EXIT_FAILURE;
  }

  void * dlh = dlopen(argv[1], RTLD_LAZY | RTLD_GLOBAL | RTLD_NODELETE);
  if (!dlh) {
    errx(kFailureExitStatus, "%s", dlerror());
  }

  MainType * entry_point = GetEntryPointFromDlHandle(dlh);

#if HIDDEN_MAIN_ARGS
  return entry_point(argc - 1, argv + 1, envp, apple);
#else
  return entry_point(argc - 1, argv + 1);
#endif
}
