# Disk recovery: repeated metadata checks and measured storage

Recovery shares one dependency observation per namespace/file ID and reuses its
temporary validation index. This reduces repeated filesystem work. It does not
change cache grants, admission guards, configured budgets or lazy backing, and
does not establish resident-memory savings on the workstation.

## Owner evidence, November 12, 2026

The owner built Gufo 0.34.0 plus the ten rebased commits, composed revision
`50145f4936ffe1b64665966c0a4b1c35b1993257`. Four sessions loaded with full context,
MTP and no upfront vision charge. The requested RAM cache was 4,294,967,296 bytes;
the grant was 3,840,673,164 bytes, a **454,294,132-byte deficit**. Its inputs were
3,765,428,224 bytes of remaining host ceiling, 76,293,516 bytes of owned manifest
storage and 1,048,576 bytes of staging. The existing reconstruction gives:

```text
3765428224 + 76293516 - 1048576 = 3840673164
```

Between post-cache-resources and pre-slots, host availability fell from
38,897,332,224 to 37,921,247,232 bytes: **976,084,992 bytes**. HIP free stayed at
40,447,311,872 bytes. Recovery read 75,143,800 manifest bytes from 100 manifests,
performed 1,137,375 dependency stats and rejected none. Logical live/peak RAM
charges were 76,356,300 / 213,979,703 bytes. They are not RSS measurements.
The previous run's 3,896,735,116-byte grant follows its different host observation;
these uncontrolled runs do not establish a release regression.

Backing separately reached 2,092,957,696 bytes and refused another
130,809,856-byte segment. Closing the grant deficit would not establish full
backing growth or image capacity. All target settings and reserves stay intact.

## Production change and bounds

`DiskStore` enumerates managed directories to account for disk usage. The platform
enumerator stats every entry, including files not referenced by an accepted
manifest. Manifest decoding then validates each manifest's structure. Previously,
dependency checking repeated `StatAt()` for every reference before the separate
cross-manifest validation pass.

The existing temporary map now first records actual size and regular/single-link
status for each `(private namespace, file ID)`. Each reference compares its own
declared size against that observation. A wrong first claim cannot poison a
correct later owner. Only manifests with every dependency available participate
in geometry, private-alias and duplicate-checkpoint checks. All sides of a
conflict are still rejected, independently of enumeration order. Cold recovery
continues to validate the entire discovered catalog; publication-only candidate
validation is unchanged.

There is one map, not an additional filesystem index. Its existing ledger
reservation includes the enlarged nodes, worst-case distinct references and
compact owner arrays before allocation. Nodes now need eight more bytes on the
tested x86-64 ABI. Invalid decoded manifests remain owned until the dependency
pass, so heavily damaged persistence can also increase temporary storage versus
immediate per-manifest rejection. The ledger still bounds this work and releases
all temporary charges on failure. None of it receives physical-memory credit.

Observations expire with construction under the existing exclusive directory
lock. They are not persisted or reused across reopen. An external writer ignoring
that ownership contract was never made coherent by repeated stat calls. Restore
still checks its opened file, no-follow/single-link/size properties, stable
identity and payload checksum before execution can use the restored state.

The allocation-failure sweep exposed pathname allocation inside `StatAt()`'s
no-throw boundary. Both platform implementations now return `not_enough_memory`;
recovery propagates allocation failure instead of terminating or rejecting valid
owners. Other filesystem exception boundaries are outside this patch.

`disk_recovered` adds total decoded `dependency_references` and
`distinct_dependency_files`; `dependency_stats` remains the actual dependency
stat count. Distinct means namespace/file ID, including missing dependencies,
not distinct inodes. Initial enumeration is additional work. The additive
extension record is [disk-recovery-observations.md](../../include/gufo/ext/changes/0.3/disk-recovery-observations.md);
checkpoint formats and publication compatibility do not change.

## Matched CPU evidence

The [raw results](cache-recovery-observations-results.json) retain every run.
Baseline is `6f34af9f9fea624c10b9fa69c5ba6b4c841e2032` with phase callbacks only;
candidate uses the production correction. Both use the same benchmark, additive
stats layout, GCC 15.3.0, `-O2 -g`, and platform library. The VM runs Linux
7.3.0-10-generic; fixtures were on ext4, not its tmpfs `/tmp`. No GPU or models
were used. Separate processes recovered identical files created once, without
flushing caches or trimming the allocator. This is warm filesystem evidence.

The larger synthetic fixture has 100 manifests, 1,137,400 references, 122,650
distinct payload identities and 41,492,800 manifest bytes. Its sparse payloads
require metadata operations, not tens of GiB of writes. Its sharing, geometry and
manifest sizes do not reproduce the owner's unknown contents.

| Larger fixture | Baseline | Candidate |
| --- | ---: | ---: |
| Dependency stats | 1,137,400 | 122,650 |
| Ordinary C++ allocations during recovery | 5,934,241 | 1,875,241 |
| Requested C++ live bytes after construction | 42,670,955 | 42,670,955 |
| Requested C++ peak bytes | 54,446,383 | 55,427,681 |
| Logical live RAM charge | 42,631,984 | 42,631,984 |
| Logical peak RAM charge | 180,258,412 | 189,357,612 |
| Recovery ms, baseline/candidate/candidate/baseline order | 2,207.09; 2,029.62 | 835.967; 825.344 |
| Process maximum RSS, KiB | 61,292; 61,328 | 61,400; 61,408 |

The measured improvement is **1,014,750 fewer dependency stats and 4,059,000
fewer temporary C++ allocations**. Requested peak heap increases by 981,298
bytes; logical peak increases by 9,099,200 bytes. There is no measured retained
heap saving or meaningful RSS reduction. Do not subtract the syscall or
allocation-count reduction from the owner's grant deficit.

Exclusive subphase times in ms, retaining both runs:

| Phase | Baseline | Candidate |
| --- | ---: | ---: |
| Directory enumeration | 207.108; 189.208 | 192.546; 191.423 |
| Manifest read/decode | 191.795; 169.481 | 166.145; 171.871 |
| Dependency checks | 1,576.69; 1,439.51 | 273.898; 273.489 |
| Shared-reference validation | 205.340; 202.619 | 181.055; 180.825 |

Map construction moves from reference validation into dependency checking. The
unique-file control (16 manifests, 8,256 references/files) performs 8,256 stats
and 58,137 C++ allocations in both variants. Live heap stays 323,531 bytes;
peak rises from 1,116,379 to 1,182,525 bytes. Its single recovery observations
were 38.7782 / 31.0071 ms and 5,872 / 5,848 KiB maximum RSS; they do not establish
a general unique-file performance improvement.

The heap instrument counts ordinary requested `new`/`new[]` bytes, not allocator
usable capacity or all native/libc allocations. Temporary and retained counted
storage reaches zero after destruction, while RSS remains about 60 MiB in the
larger runs. That distinction is consistent with retained allocator/native
storage; it does not attribute every resident page. Logical reservations are a
separate, deliberately conservative quantity.

Constructor `/proc/self/io` physical reads/writes were zero in these warm runs.
Both variants read the same 41,492,800 manifest bytes and no payload contents.
Reported `rchar`/`syscr` also include `/proc` sampling: baseline performs more
phase-exit observations. They are not dependency-stat counters or evidence of
less payload I/O. Exclusive phase times exclude sampling; whole startup includes
it. RSS/high-water and system observations are approximate and not atomic.

Initial fixture creation itself touches unique filesystem metadata. Later
directory scans still touch it in both variants. Host availability, cache/slab
samples and process RSS therefore cannot attribute the owner's approximately
976 MB decrease to C++ metadata, a leak, or ZFS ARC. That attribution remains
unresolved in this VM.

## Validation and reproduction

The supported native CPU build and nine selected regressions passed, **9/9 in
20.00 seconds**. They cover recovery/restore, missing and truncated dependencies,
symlinks/hard links, namespace separation, wrong-size claims, all conflicting
owners, duplicate IDs, corruption, publication failures, allocation/ledger
failure cleanup, cache admission/backing and the independent GGUF identity fix.
The first disk-test run exposed the no-throw allocation termination described
above; the final run includes its correction.

```sh
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset cpu-test --target cache_disk_startup_bench \
  cache_disk_test cache_disk_publication_test component_text_cache_test \
  cache_tiered_test cache_streaming_test cache_ledger_test cache_full_backing_test \
  platform_fs_test gguf_identity_test --parallel 4
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  ctest --test-dir build/cpu-test \
  -R '^(cache_disk_test|cache_disk_publication_test|component_text_cache_test|cache_tiered_test|cache_streaming_test|cache_ledger_test|cache_full_backing_test|platform_fs_test|gguf_identity_test)$' \
  --output-on-failure --parallel 4 --timeout 180
c++ -std=c++20 -O2 -Wall -Wextra -pthread -I. -Iinclude \
  -c src/cache/disk.cpp -o /tmp/gufo-recovery-disk-production.o
python3 tools/ci/check-extension-api.py --base HEAD --labels '["ext-api"]'
python3 tools/ci/generate-infrastructure.py --check
python3 tools/ci/check-docs.py
python3 tools/ci/check-format.py
```

API 0.3.5 and infrastructure checks pass. Full formatting reports only the two
untouched baseline violations in `audio_tts_api.cpp:473` and
`openai_chat_test.cpp:6750`; changed C++ files pass. Real HIP and Windows builds
are unavailable. No HIP translation unit changes; `fs_windows.cpp` is the changed
platform path not compiled here. CPU fake-HIP tests do not validate GPU execution.

## One later model-free owner measurement

Use a maintenance window with production chat unloaded and auxiliary services
running. Leave both live persistence and its backup untouched. Use a new
disposable directory on the same disk-backed filesystem as persistence; do not
flush caches or change ZFS/kernel/queue settings. No model load is needed.

Build the CPU benchmark with the repository toolchain, then compile baseline
and candidate with identical flags. The [baseline patch](cache-recovery-baseline-hooks.patch)
adds observation callbacks only; apply it to a temporary source file:

```sh
nix develop -c cmake --preset cpu-test -DGUFO_BUILD_TOOLS=ON
nix develop -c cmake --build --preset cpu-test \
  --target cache_disk_startup_bench --parallel 4
# Choose a disposable disk-backed parent, outside every real cache directory.
probe=$(mktemp -d /path/to/disposable-parent/gufo-recovery-XXXXXX)
git show 6f34af9f9fea624c10b9fa69c5ba6b4c841e2032:src/cache/disk.cpp \
  > "$probe/baseline.cpp"
patch "$probe/baseline.cpp" docs/internal/cache-recovery-baseline-hooks.patch
for variant in baseline candidate; do
  source=src/cache/disk.cpp
  if [ "$variant" = baseline ]; then source="$probe/baseline.cpp"; fi
  nix develop -c c++ -std=c++20 -O2 -g -UNDEBUG -DGUFO_CACHE_TESTING \
    -pthread -I. -Iinclude tests/cache/disk_startup_bench.cpp "$source" \
    -Wl,--start-group build/cpu-test/src/cache/libgufo_cache.a \
    build/cpu-test/src/core/platform/libgufo_platform.a \
    -Wl,--end-group -ldl -o "$probe/$variant"
done
"$probe/candidate" --create "$probe/fixture" 100 11250 124 10250 \
  > "$probe/create.log"
"$probe/baseline" --recover "$probe/fixture" > "$probe/baseline.log"
"$probe/candidate" --recover "$probe/fixture" > "$probe/candidate.log"
```

Creation reports system observations before and after unique files are touched;
recovery reports each subphase, live/peak C++ bytes, RSS, I/O and destruction.
This distinguishes process-owned recovery storage and repeated work from changes
already visible during filesystem population. It does not reproduce a rebooted
filesystem cache or prove a specific ZFS mechanism. Keep all three logs and the
filesystem/runtime identity; no repeated full-model attempt is requested.
Neither reduced syscall counts nor this probe qualify the 4 GiB grant, full
backing growth, four-request peaks or occasional image capacity.
