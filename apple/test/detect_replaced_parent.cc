// clang++ -std=c++20 -Wall -Werror -o detect_replaced_parent \
//     detect_replaced_parent.cc

// Context:
// https://chromium-review.googlesource.com/c/4117866/comment/b63d1660_2c5c0b96/
//
// Define MODE to 1 for the parent to execv, 2 for the parent to posix_spawn
// with POSIX_SPAWN_SETEXEC, and 3 for the parent to exit prematurely. No
// malfeasance is attempted with MODE undefined or set to 0.

#include <err.h>
#include <inttypes.h>
#include <libproc.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <limits>
#include <type_traits>

extern "C" {

char ** environ;

// 13.1 xnu-8792.61.2 bsd/sys/proc_info.h in PRIVATE, but p_idversion only
// exists as far back as 11.0 (2020).
struct proc_uniqidentifierinfo {
  uint8_t p_uuid[16];
  uint64_t p_uniqueid;
  uint64_t p_puniqueid;
  int32_t p_idversion;
  uint32_t p_reserve2;
  uint64_t p_reserve3;
  uint64_t p_reserve4;
};

#define PROC_PIDUNIQIDENTIFIERINFO 17

}  // extern "C"

namespace {

template <typename T>
std::make_unsigned_t<T> ToUnsigned(T const value) {
  return static_cast<std::make_unsigned_t<T>>(value);
}

auto GetProcIdVersionForPid(pid_t const pid) {
  proc_uniqidentifierinfo unique_identifier_info;
  int const rv = proc_pidinfo(pid,
                              PROC_PIDUNIQIDENTIFIERINFO,
                              0,
                              &unique_identifier_info,
                              sizeof(unique_identifier_info));
  if (rv == 0) {
    err(EXIT_FAILURE, "proc_pidinfo");
  }
  if (rv < sizeof(unique_identifier_info)) {
    errx(EXIT_FAILURE,
         "proc_pidinfo: returned %d, expected %zu",
         rv,
         sizeof(unique_identifier_info));
  }

  return ToUnsigned(unique_identifier_info.p_idversion);
}

void Parent(pid_t const child_pid) {
#if defined(MODE) && MODE != 0
#if MODE == 1 || MODE == 2
  static constexpr char kExecutable[] = "/bin/sleep";
  static constexpr char const * kArgv[] = {"sleep", ".1", nullptr};
#if MODE == 1
  execv(kExecutable, const_cast<char * const *>(kArgv));
  err(EXIT_FAILURE, "execv");
#elif MODE == 2
  posix_spawnattr_t posix_spawn_attrs;
  int rv = posix_spawnattr_init(&posix_spawn_attrs);
  if (rv != 0) {
    errc(EXIT_FAILURE, rv, "posix_spawnattr_init");
  }
  rv = posix_spawnattr_setflags(&posix_spawn_attrs, POSIX_SPAWN_SETEXEC);
  if (rv != 0) {
    errc(EXIT_FAILURE, rv, "posix_spawnattr_setflags");
  }
  rv = posix_spawn(nullptr,
                   kExecutable,
                   nullptr,
                   &posix_spawn_attrs,
                   const_cast<char * const *>(kArgv),
                   environ);
  errc(EXIT_FAILURE, rv, "posix_spawn");
#endif
#elif MODE != 3
#error unknown MODE
#endif
#else
  usleep(100e3);
#endif
}

void Child() {
  // Give the parent an opportunity to do its thing.
  usleep(10e3);

  pid_t const parent_pid = getppid();

  if (parent_pid == 1) {
    errx(2, "child: detected replaced parent: exit");
  }

  auto const self_proc_id_version = GetProcIdVersionForPid(getpid());
  auto const parent_proc_id_version = GetProcIdVersionForPid(parent_pid);

  if (self_proc_id_version - parent_proc_id_version >
      std::numeric_limits<
          std::make_signed_t<decltype(self_proc_id_version)>>::max()) {
    errx(2, "child: detected replaced parent: execve or POSIX_SPAWN_SETEXEC");
  }

  printf("child: original parent is intact\n");
}

}  // namespace

int main(int const argc, char * const argv[]) {
  pid_t const pid = fork();
  if (pid < 0) {
    err(EXIT_FAILURE, "fork");
  }

  if (pid != 0) {
    Parent(pid);
  } else {
    Child();
  }

  return EXIT_SUCCESS;
}
