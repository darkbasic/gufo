# Cache transfer resources: native evidence and qualification

This investigation is separate from vision admission and GGUF identity. It
targets resident native cache-transfer resources while retaining the serving,
cache and completion contracts. The production optimization is enabled for
Flash-Next's session-count-aware serving provider; other providers retain their
existing pool policy. Hardware observations below were supplied by
the owner on November 11, 2026; the development VM did not run HIP or load models.

## Owner-provided native measurements

The owner ran all 24 [native probe](transfer-resource-probe.md) cases with the
matching r6 runtime and effective `GPU_MAX_HW_QUEUES=2`: streams, events, pairs
and the production pool, each at counts 1, 2, 4, 8, 16 and 64. Every case
succeeded. All processes had the same initialized HIP free-memory baseline.

For every stream-bearing mode, creation above the warmed baseline was exactly:

```text
20,291,584 + native_stream_count * 8,388,608 bytes
```

Destruction returned exactly `native_stream_count * 8 MiB`. Events alone had no
observable additional HIP allocation; first transfer use and synchronization
added no HIP allocation in this probe. The fixed 20,291,584-byte term remained
after destruction. The probe establishes these costs for the tested runtime,
not a universal HIP allocation size or a correspondence with hardware queues.

The owner did not reboot between cases. That does not invalidate each process's
creation/destruction comparison with its own warmed baseline. Embedding and
Open WebUI were stopped, so this is native resource-cost evidence, not a daily
workload or full-capacity qualification.

## Resource bound and progress

Flash-Next constructs `N + 1` native FIFO streams and `18 * N + 1` logical event
leases for the actual `N` execution sessions. The independent event for each
lease lets several transfers share a native stream without sharing their
completion identity. All streams and events are created before slots, cache
grant and backing preparation; acquisition creates no native resource.

The logical bound conservatively covers one permanent preservation lease, one
16-transfer foreground batch and one already-submitted idle piece per slot,
plus the one transfer using `StreamedStore`'s serialized staging buffer. An
execution slot has one owner, including through supported typed callers.
Existing per-operation batching remains unchanged. The extra idle allowance
avoids relying on exact worker handoff timing. Zero sessions and overflowing
plans are rejected.

| Execution sessions | Native streams | Event leases |
| ---: | ---: | ---: |
| 1 | 2 | 19 |
| 2 | 3 | 37 |
| 4 | 5 | 73 |
| 7 | 8 | 127 |

The native count retains one lane per slot producer plus disk, rather than one
resident stream for every retained completion. Logical leases do not exclusively
occupy a lane. Submission uses HIP's FIFO ordering; it neither waits for another
lease to finish nor introduces a polling/fairness queue. A healthy completion
waits for its own event fence, so later work from another producer cannot extend
that wait. Existing cancellation and source-ownership waits remain in their
consumers. Before completion seals a lease, explicit synchronization and
destruction also fence work submitted through a previously returned native
handle; callers do not need to reacquire that handle after every synchronization.

Shared-mode events request a system-scope release, retaining the host visibility
previously supplied by stream synchronization; see HIP's
[coherence-control contract](https://rocmdocs.amd.com/projects/HIP/en/develop/how-to/hip_runtime_api/memory_management/coherence_control.html).
A lease's event remains owned until both its stream facade and completion
release it. Failed or partially submitted work is drained before source owners
unwind. Pool-observed native failures quarantine affected resources instead of creating
replacements or waiting forever for capacity that cannot return. External bound
execution streams retain their owner and failure scope. Shutdown drains owners
before destroying the eager native pool. The Flash-Next temporary key-copy path
also waits for its lease fence before freeing its temporary buffer, avoiding an
unrelated whole-stream wait. Errors reported directly by an adapter's `Native()`
submission retain that adapter's existing failure handling; a successful drain
does not invent a native lane failure.

Serving passes its actual session count through the additive
`CreateComponentCacheResourcesForSessions()` provider method.
Its default forwards to the existing two-argument factory, preserving unsupported
providers. Flash-Next's old unsized entry point likewise retains the previous
exclusive pool. This adds a source extension API revision, not a promise of
binary compatibility; extensions must rebuild.

## VM validation scope

CPU regressions exercise production resource planning, provider dispatch,
transfer-pool ownership and serving/cache consumers with small synthetic data.
The serving cases include four concurrent 16-transfer captures, concurrent RAM
restores, a background copy held at a controlled completion while foreground
capture progresses, durable publication, restart/disk restore and cold
reassignment. Legacy providers and several actual session counts remain covered.
The native-pool tests separately check shared-lane fences, retained completions,
creation/submission/synchronization failure and quarantine.

On November 11, 2026, the native CPU PR gate passed **116/116 tests in 52.20
seconds** of CTest time. The focused serving transfer-resource cases also passed.
The following commands ran with the isolated local dependency prefix; they are
CPU checks, not a Nix or HIP build:

```sh
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --preset cpu-test -DCMAKE_PREFIX_PATH=/tmp/gufo-cpu-deps/usr \
  -DZLIB_LIBRARY_RELEASE=/usr/lib/x86_64-linux-gnu/libz.so.1
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset pr --parallel 4
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  timeout 30s build/cpu-test/component_text_cache_test \
  --transfer-resources-only > /tmp/gufo-transfer-component-test.log 2>&1
```

The PR gate log is `/tmp/gufo-transfer-pr.log`; the focused serving log is
`/tmp/gufo-transfer-component-test.log`. The initial rebuild encountered stale
dependency paths from the earlier VM session; restoring the isolated prefix
allowed normal CMake configuration and the successful native gate above.

A final compatibility correction covered reuse of a cached native handle after
manual synchronization, including destruction without a completion. After that
change, rebuilding the affected targets and running their regressions passed
**2/2 in 1.38 seconds**, recorded in `/tmp/gufo-transfer-final-tests.log`:

```sh
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset cpu-test --target cache_full_backing_test \
  component_text_cache_test --parallel 4
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  ctest --test-dir build/cpu-test \
  -R '^(cache_full_backing_test|component_text_cache_test)$' \
  --output-on-failure --timeout 30
```

A focused ThreadSanitizer build also passed with no diagnostics after the final
correction. It instruments the test and the four production HIP resource sources
below, using fake HIP; reused cache/platform/vision static libraries are
uninstrumented. This is not whole-tree sanitizer or GPU validation.

```sh
c++ -std=c++20 -O1 -g -UNDEBUG -DGUFO_CACHE_TESTING -pthread \
  -fsanitize=thread -fno-omit-frame-pointer -Itests/cache/fake_hip -I. -Iinclude \
  tests/cache/full_backing_test.cpp src/core/hip/component_cache_resources.cpp \
  src/core/hip/committed_backing.cpp src/core/hip/host_allocations.cpp \
  src/core/hip/transfer_pool.cpp -Wl,--start-group \
  build/cpu-test/tests/cache/libgufo_cache_fake.a \
  build/cpu-test/src/cache/libgufo_cache.a \
  build/cpu-test/src/families/qwen/vision/libgufo_qwen_vision.a \
  build/cpu-test/src/core/platform/libgufo_platform.a -Wl,--end-group -ldl \
  -o /tmp/gufo-transfer-full-backing-tsan
env TSAN_OPTIONS=halt_on_error=1 timeout 30s \
  /tmp/gufo-transfer-full-backing-tsan
```

The extension API diff gate passed at **0.2.26 / 130 surface paths**, with the
`ext-api` label. Documentation, `git diff --check` and formatting of all 13 changed
C++ files passed, including after the final fence correction. The repository-wide
formatting check still reports only the
untouched baseline locations `src/cli/serve/audio_tts_api.cpp:473` and
`tests/cli/openai_chat_test.cpp:6633`; those files were not changed. The relevant
repository commands were:

```sh
python3 tools/ci/check-extension-api.py \
  --base 62098131621441b3f8ff9ba524a89b5195ddb39b --labels '["ext-api"]'
python3 tools/ci/check-docs.py
git diff --check
python3 tools/ci/check-format.py
```

The VM has no inference GPU, weights or real HIP toolchain. In particular, the
changed real-runtime paths in `src/core/hip/transfer_pool.cpp`,
`src/core/hip/component_cache_resources.cpp`,
`src/models/qwen38_flash_next/runner.cpp` and
`src/models/qwen38_flash_next/continuation_adapter.cpp` remain uncompiled against
HIP here. Compiling the first two with fake HIP does not validate the SDK, GPU
coherence or physical memory fit. Build the candidate through the normal kit's
matching HIP SDK/runtime; the repository's native release commands are:

```sh
cmake --preset release
cmake --build --preset release --parallel 4
```

Retain the kit's existing extension selection, runtime and build flags. These
commands are for a capable environment, not executed in this VM.

## Connection to the r6 cache refusal

The [r6 production observations](models/qwen3.8-flash-next/VISION-MEMORY-ADMISSION.md#r6-resource-observations-at-1426e9d7)
lost exactly 536,870,912 HIP bytes while constructing 64 native stream/event
pairs. No cache backing existed yet. Four full-context execution sessions then
loaded, but the launcher rejected the effective RAM-cache grant:

```text
requested_bytes = 4,294,967,296
granted_bytes   = 3,851,272,192
grant_gap_bytes =   443,695,104
```

The selected four-session plan retains five native streams, so the measured
count-dependent term predicts removal of 59 * 8 MiB = **494,927,872 bytes
(472 MiB)**. That exceeds the recorded grant gap by only **51,232,768 bytes**.
The fixed probe initialization term is not an additional saving. This is a
conditional prediction from the owner's runtime measurements, not a changed
candidate GPU result. The shared implementation uses more events and system-scope
event releases; their actual footprint and throughput must still be checked.

Grant, planned payload/metadata, committed backing and remaining runtime
capacity are distinct. The r6 plan contained 3,409,051,648 payload bytes,
2,229,338,112 had committed during ordinary startup preparation, and the later
runtime observation had only 9,113,408 bytes free above accepted obligations and
reserves. Recovering the logical grant gap alone does not fund every later
backing segment or image request. A full requested grant also has its own
production `BackingPlan()`; 4 GiB of retention is not 4 GiB of payload pages.

## Owner qualification of a changed candidate

Use the normal workstation kit workflow in an owner-controlled maintenance
window. Unload production Flash-Next chat before starting the candidate.
Preserve production files and caches, and run EmbeddingGemma and Open WebUI for
this workload qualification. They were absent from the probe measurement.
Neither this procedure nor its harness stops services or changes the kit.

Add the reviewed cache-resource source revision to the kit's existing candidate
source selection. Retain its vision-admission and GGUF fixes and the independent
disk-publication patch in the composed build. Do not change queue/kernel limits,
swap, reserves, model quantization, image-resolution defaults or these settings:

```text
sessions=4
context=262144
maximum_output=32000
speculative=shared Q8_0 MTP
draft_tokens=7
vision=loaded BF16 projector
vision_memory_policy=on-demand
cache_ram_bytes=4294967296
cache_disk_bytes=53687091200
```

Use a disposable disk-backed cache and a separate unused port, as in the
existing [workload procedure](models/qwen3.8-flash-next/VISION-MEMORY-VALIDATION.md).
Keep the launcher's requested-session and requested-cache checks. Retain the
candidate source/build identity, effective queue environment and this attempt's
complete log. Stop after startup or grant refusal and inspect that saved log;
do not repeat the same model load without a changed hypothesis.

Separate these outcomes in the result:

1. Record the actual native transfer-resource count and the cache-resource
   phase's host/HIP observations. Expect `event=transfer_resources
   native_streams=5 event_leases=73 allocation=eager` for four sessions. Compare
   the count-dependent saving with r6,
   without treating observations from different phases as one snapshot.
2. Require completed startup with four execution sessions, context 262,144,
   MTP draft limit seven and the projector, followed by the full 4,294,967,296
   byte logical RAM grant. Readiness alone does not qualify the requested cache.
3. Record the production backing plan and actually committed bytes. Exercise
   captures beyond the ordinary preparation target, RAM restore, durable disk
   publication, cold reassignment and disk restore after the harness-owned
   restart. Preserve later backing-growth or pressure refusals explicitly. If
   the workload never grows beyond startup preparation, report cache growth as
   untested; a successful 2 GiB warmup is not proof of full payload capacity.
4. Exercise four text sessions concurrently, proving progress with distinct task
   IDs and progress counters. Include a background publication or preservation
   already in flight when new foreground capture/restore work begins.
5. Exercise an image-bearing request while text sessions progress. Record
   admission or the normal structured resource refusal, then verify text
   recovery. Retain image numerical checks, TTFT/throughput, peak host/device
   memory and swap observations; CPU regressions establish none of these.

Only a complete successful run qualifies the daily workload. A reduced native
resource count and full logical grant do not by themselves establish full cache
growth, image capacity, numerical correctness or unchanged throughput.
