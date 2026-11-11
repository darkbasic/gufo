# Native HIP transfer-resource probe

Measure native stream/event overhead without loading a model. This tool is a
separately reviewable cache-resource investigation, not a change to #920's
admission policy or the production transfer-pool size. As of November 11, 2026,
real HIP compilation and execution of this new tool, and any memory saving,
remain pending. Earlier candidate GPU builds do not compile the new probe.

## Why this measurement is needed

The [owner's r6 observations](models/qwen3.8-flash-next/VISION-MEMORY-ADMISSION.md#r6-resource-observations-at-1426e9d7)
localize 512 MiB of HIP consumption to cache-resource construction. The
production path explains the phase:

- `qwen38_flash_next/continuation_adapter.cpp` builds host layout, identity and
  descriptors; its constructor does not call HIP.
- `core/hip/component_cache_resources.cpp` immediately constructs a
  `TransferPool` with 64 pairs. Its later `commit_backing` callback creates the
  backing pool and prepares pages, after slot creation and the cache grant.
- `core/hip/transfer_pool.cpp` creates one nonblocking stream and one
  timing-disabled event per pair. This implementation is identical to upstream
  Gufo 0.33.0 at `77dff828be02a498e9951694f31b024e7ac87119`.

This establishes the allocation phase, not a cost per stream or a correspondence
between streams and hardware queues. The probe separates fixed initialization,
count-dependent creation and first-use costs before a production change is
selected. No measured savings are claimed from the CPU simulation.

## Build outside the production runtime image

Use the same SDK/compiler/dependency environment that built r6 and retain its
HIP/ROCr versions. The repository's normal release-build prerequisites apply;
this explicit target does not require the optional profiling tools:

```sh
cmake --preset release -DGUFO_BUILD_TOOLS=OFF
cmake --build --preset release --target gufo-hip-resource-probe --parallel 4
build/release/gufo-hip-resource-probe --help
```

If that environment uses the pinned Nix development shell, prefix each command
with `nix develop -c`. Do not substitute a different runtime for comparison.
The target builds native C++ runtime calls and the production transfer pool;
it does not build model kernels or read weights. Its output is
`build/release/gufo-hip-resource-probe`. The target is excluded from ordinary
release builds and packaging; request it explicitly. Record the source revision, compiler
and linked libraries in the build environment:

```sh
git rev-parse HEAD
cmake --build --preset release --target gufo-hip-resource-probe --verbose
ldd build/release/gufo-hip-resource-probe
```

The runtime image needs the native binary and its matching shared libraries,
not `hipcc`, CMake or Python. Use the existing r6 runtime image and GPU device,
user/group, memory/cgroup limits and environment configuration in an owner-controlled diagnostic
container. Add only the probe binary read-only and a new writable evidence
directory. Models and production cache need not be mounted. Do not edit the
workstation launcher or invent a replacement production launch command.

Check that the resolved HIP/ROCr libraries belong to that tested runtime; use
`ldd` inside the diagnostic image when available, or its loader/dependency
inspection in the matching SDK/container build. In particular, a binary with
Nix-store or SDK RPATH entries is not evidence for an unrelated slim image's
runtime. Retain the image identity and dependency listing with the output.
The probe also reports runtime/driver versions and selected queue environment
values; it does not change them.

## Match the effective r6 queue environment

Use the **saved r6** `event=queue_budget` line, not a new model load, to identify
the effective cap. `src/cli/serve/serve.cpp` calls `PlanQueues()` and
`ApplyQueuePlan()` before HIP initialization. Gufo can therefore apply
`GPU_MAX_HW_QUEUES` inside its process even when the launcher's environment leaves
it unset. The standalone probe deliberately does not run that queue planner.

| Saved r6 `cap` field | Required diagnostic environment |
| --- | --- |
| `cap=N` | Reproduce that exact Gufo-applied numerical `GPU_MAX_HW_QUEUES=N` |
| `cap=operator:N` | Preserve the existing operator value `N` |
| `cap=runtime` | Preserve the runtime-default/unset value |
| `cap=operator` | Preserve the exact original operator value; the log alone does not reconstruct it |

Set up only the diagnostic container's environment to reproduce the recorded
effective value; do not choose a new cap or change kernel/queue limits. Preserve
all other queue and visibility variables from r6. Compare the probe's environment
records with the saved line and container configuration. If the effective
value cannot be established or differs, the footprint comparison is inconclusive.
An unset parent variable is not sufficient evidence that the r6 process ran
without a Gufo-applied cap. No deployment change or extra model load is needed.

## Owner-controlled execution

Arrange a maintenance window and unload production Flash-Next chat using the
normal operational procedure before probing. Preserve its files, models and
cache. Retain EmbeddingGemma, Open WebUI and the other auxiliaries represented
in the target budget. Keep the effective queue environment established above, kernel limits,
HugeTLB and swap settings unchanged. The probe never stops or starts services.
Restore production chat through the normal procedure after collecting evidence.

The command below runs inside the diagnostic container described above. Set
`GUFO_PROBE_BINARY` to the mounted executable and `GUFO_PROBE_OUTPUT` to a new
subdirectory of the disposable evidence mount. An existing output directory
causes `mkdir` to fail rather than mixing runs. No Python interpreter is needed:

```sh
set -eu
GUFO_PROBE_BINARY=/probe/gufo-hip-resource-probe
GUFO_PROBE_OUTPUT=/evidence/transfer-resource-r6
mkdir "$GUFO_PROBE_OUTPUT"
"$GUFO_PROBE_BINARY" --help > "$GUFO_PROBE_OUTPUT/help.txt"
for GUFO_PROBE_MODE in streams events pairs pool; do
  for GUFO_PROBE_COUNT in 1 2 4 8 16 64; do
    if "$GUFO_PROBE_BINARY" --mode "$GUFO_PROBE_MODE" \
      --count "$GUFO_PROBE_COUNT" \
      > "$GUFO_PROBE_OUTPUT/$GUFO_PROBE_MODE-$GUFO_PROBE_COUNT.jsonl" \
      2> "$GUFO_PROBE_OUTPUT/$GUFO_PROBE_MODE-$GUFO_PROBE_COUNT.stderr"; then
      :
    else
      echo "Probe failed: $GUFO_PROBE_MODE/$GUFO_PROBE_COUNT; inspect saved output" >&2
      exit 1
    fi
  done
done
```

This is 24 fresh, sequential processes with at most 64 stream/event pairs
(128 native handles) per process.
Each uses only a small device buffer and pinned host buffer, 262,144 bytes each;
native runtime overhead is the quantity being measured. The `streams` and
`events` modes isolate creation; `pairs` records stream and event creation
separately in the same process; `pool` exercises production `TransferPool`.
The tool records ordinary HIP initialization, allocation and default-stream
warmup before measured creation, then first use, synchronization and destruction.
Milestones at 1, 2, 4, 8, 16 and 64 record the first and subsequent creations
and submissions within each raw-mode process; fresh processes also separate
the lifetime peaks for different pool capacities.
Each stream uses a distinct 4 KiB slice for ordered host/device round trips.
The tool drains work and attempts explicit destruction, including failure paths.
In pool mode it retains production cleanup behavior: the pool destructor does
not expose native destroy failures. `checked_cleanup_succeeded` therefore
describes checked waits/drains, not a guarantee that every driver destruction
returned memory.

Read `live_streams`, `live_events` and `submitted_transfers` with each phase;
`count` is the configured capacity, not the number already created or used.
Before HIP initialization, only a host reading is available. Later host
availability and `hipMemGetInfo` are collected
in the same phase, not atomically; retain their timestamps and external workload
conditions. They constrain the same unified physical memory and must not be
added as separate budgets. Preserve failed runs rather than replacing them with
successful retries. The probe does not establish image numerics or model fit.

## Decide from creation and peak use

Compare each process with its own warmed baseline, then compare the count and
mode curves. Keep fixed initialization, native creation and first-use increments
separate. The default-stream warmup can trigger a fixed runtime/queue cost in
`initialized`, `buffers_ready` or `warmed`; looking only at `created` minus
`warmed` can miss it. Retain all phases before attributing the production
construction interval. Destruction may leave runtime-managed caches; unrecovered free memory
alone does not establish a leak. Cross-process noise or continuing auxiliary
work can make a small difference inconclusive.

A change is worth pursuing only if this evidence supports a reduction in the
peak of a required workload, not a delay until after readiness:

- A large fixed first-stream or first-use cost does not justify predicting
  proportional savings from fewer streams. Moving that cost to first capture
  must not let a larger cache grant spend memory the capture will inevitably
  need.
- A count-dependent cost can motivate a bounded pool proposal only after a
  global forward-progress argument. Four sessions keep four permanent
  preservation leases. Idle/foreground preservation and reassignment need
  additional leases; a pool of four leaves none. Foreground paths can wait,
  while idle work can defer. Capture, RAM restore and disk capture batch at
  most 16 operations each, but that is not a global bound of `sessions + 16`.
- Completion ownership outlives the stream facade. A pair returns only when
  both the stream and completion release it; state outlives pool facade
  destruction. Failure quarantines a pair. Any proposal must preserve draining,
  cancellation, partial-failure cleanup, restore, shutdown and progress after
  exhaustion or quarantine.
- Lazy native construction changes `TryAcquire()`'s current promise of no HIP
  allocation. It needs synchronized initialization, failure cleanup, explicit
  memory ownership/reservations and a latency assessment. An inevitable later
  commitment must remain funded; hiding it behind readiness is not a saving.

Keep a justified allocator/transfer-resource optimization in a separate patch
from #920. Use the production transfer/backing tests to cover permanently held
session leases, overlapping operations, exhaustion/recovery, surviving
completions, injected creation failures and shutdown before a model trial.

Do not request another unchanged reboot/model load. A subsequent changed
candidate needs a measured footprint expectation or a named unresolved
measurement that only models can answer. Then use the existing
[full-workload procedure](models/qwen3.8-flash-next/VISION-MEMORY-VALIDATION.md)
with the same sessions, context, seven MTP drafts, projector, 4 GiB requested RAM
budget, 50 GiB disk budget and auxiliaries. Preserve the launcher's full-grant
check. Text readiness, retention grant, production `BackingPlan()` payload and
metadata, actual committed backing and image/optional-execution headroom remain
separate results; closing the grant gap alone does not qualify the workload.
