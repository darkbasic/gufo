# Vision admission: recorded ZBook evidence

This page preserves owner-reported hardware observations as of November 11,
2026. The latest candidate reached readiness with four sessions and ordinary
copy verification; its cache grant still failed the deployment target. The VM
did not execute these runs.
See the [current policy](VISION-MEMORY.md) and [validation scope](VISION-MEMORY-TESTS.md).

## r6 resource observations at 1426e9d7

The owner-run r6 build used `1426e9d787a8c473919edc71afdd3948f25bb439` and
confirmed on-demand vision, `copy_width_selection rows=8`, four sessions at
context 262,144 and MTP `draft_limit=7`. Native `load_completed` and
`/v1/models` succeeded. The launcher correctly rejected the effective RAM grant;
its requested-session/cache check remains unchanged.

The observations bracket adapter/resource construction before backing pages
exist:

| Observation | Before resources | After resources |
| --- | ---: | ---: |
| HIP free bytes | 40,489,259,008 | 39,952,388,096 |
| Host available bytes | 38,881,210,368 | 38,341,914,624 |
| Explicit execution committed bytes | 0 | 0 |
| Transfer stream count | 0 | 64 |

The interval consumes **536,870,912 HIP bytes (512 MiB)** and
**539,295,744 host bytes (514.3125 MiB)**. Production construction creates the
adapter/component metadata and 64 native stream/event pairs; cache-page
commitment occurs later. This localizes a substantial cost to that phase, but
does not prove a per-stream cost, a hardware-queue count or proportional savings
from reducing the pool. The same eager transfer construction exists upstream
in Gufo 0.33.0. The [independent native probe](../../transfer-resource-probe.md)
separates runtime warmup, stream creation, event creation and first use before
any optimization is chosen.

Slot and grant observations were:

```text
pre_slots:
  hip_free_bytes=39952388096
  host_available_bytes=38000246784
post_slots:
  hip_free_bytes=10214772736
  host_available_bytes=8147288064
  execution_committed_bytes=29492622480
  execution_pending_bytes=1033826496
  runtime_capacity_bytes=2281623360
per_slot:
  hip_free_decrease_bytes=7434403840
  execution_committed_increase_bytes=7373155620
cache_grant:
  host_available_bytes=8147288064
  host_reserve_bytes=4294967296
  staging_capacity_bytes=1048576
  requested_bytes=4294967296
  granted_bytes=3851272192
```

The grant is **443,695,104 bytes (423.140625 MiB)** below 4 GiB. Between resource
construction and `pre_slots`, host availability decreases another 341,667,840
bytes while HIP free is unchanged. Each slot's HIP decrease exceeds its explicit
execution increase by 61,248,220 bytes. Those interval differences remain
unattributed; they are not established allocator rounding, leaks or transfer
storage. Keep them separate from the measured resource-construction interval.

Backing reached its ordinary preparation target:

```text
backing_prepared:
  committed_bytes=2229338112
  segment_bytes=131137536
  segments=17
  planned_payload_bytes=3409051648
  planned_metadata_bytes=53270528
post_backing:
  host_available_bytes=5874778112
  execution_pending_bytes=1033826496
  dynamic_committed_bytes=2229338112
  pending_bytes=0
  runtime_capacity_bytes=9113408
```

Seventeen segments commit 2,229,338,112 bytes, crossing the ordinary 2 GiB
warmup target. No `backing_limited` warning is expected when warmup stops there;
its absence does not prove another segment or the complete payload fits.
The current grant's plan still has **1,179,713,536 uncommitted payload bytes**.
The next 131,137,536-byte segment exceeds the reported 9,113,408-byte runtime
capacity by 122,024,128 bytes, holding that observation fixed. Funding even the
current plan's remaining payload would require 1,170,600,128 additional bytes
of headroom under the same observation, before any images. These differences
are snapshot arithmetic, not new allocation measurements.

Post-backing accounting is consistent with the existing reserves:

```text
5874778112 - 4294967296 - 536870912 - 1033826496 = 9113408
```

Committed backing already lowered host availability and is not subtracted
again. Recovering the 423.140625 MiB grant gap would not establish full backing
or image capacity: a full requested grant has its own larger production plan.
Keep retention grant, planned payload/metadata, actual commitment and remaining
runtime headroom as separate qualification results. No model reload is warranted
until the model-free probe supports a specific changed candidate or resolves a
necessary measurement.

## Ready engine with cache limits at 83c0c7a2

The owner reported four execution sessions at context 262,144, MTP
`draft_limit=7` and the BF16 projector. The engine listened and answered
`/v1/models`. The launcher then stopped it because the effective RAM-cache
grant was below the requested 4 GiB. This is successful initial admission and
readiness, not qualification of four concurrent conversations or the full cache.

The startup observation was:

```text
hip_free_bytes=40497627136
device_admission_bytes=40497627136
host_available_bytes=38883639296
host_ceiling_bytes=34588672000
sessions=4
base_bytes=7529402708
mtp_hidden_bytes=2662400
deferred_scratch_bytes=2321955840
total_text_bytes=30117610832
gate=device_state
deficit_bytes=0
result=eligible
```

The later cache and backing observations were:

```text
requested_bytes=4294967296
granted_bytes=3810430976
automatic_bytes=4053223424
staging_capacity_bytes=1048576
backing_prepared committed_bytes=259522560

segment_bytes=129761280
outstanding_bytes=259522560
pending_bytes=0
capacity_bytes=64586560
execution_pending_bytes=2937769152
```

The grant is **484,536,320 bytes (462.09 MiB)** below the requested limit.
Physical backing stopped at **259,522,560 bytes (247.5 MiB)**; another
129,761,280-byte segment could not fit its 64,586,560-byte runtime capacity.
These are independent limits. `ContinuationLayout::BackingPlan()` divides a
retention grant between payload pages and metadata/headroom; neither the grant
nor full backing means exactly 4 GiB of payload pages. Compare the production
plan with committed backing, and keep the requested and granted limits separate.
The production backing-plan regression gives these logical capacities; they
are source-derived plans, not observed page commitment:

| Retention grant | Planned payload bytes | Planned backing metadata bytes |
| ---: | ---: | ---: |
| 4,294,967,296 | 3,802,267,648 | 59,414,528 |
| 3,810,430,976 | 3,372,875,776 | 52,705,280 |

The remainder supports the cache's other headroom and metadata. A full requested
grant therefore plans 3,802,267,648 payload bytes, while the reported run committed
only 259,522,560 bytes under its smaller grant.

### Optional copied-token width

The device gate admitted 64 copy rows. This optimization is separate from
seven-token neural MTP. Production formulas give:

| Copy rows | MTP hidden bytes per session | Deferred scratch bytes |
| --- | ---: | ---: |
| 8 | 368,640 | 418,013,184 |
| 64 | 2,662,400 | 2,321,955,840 |
| Difference | 2,293,760 | 1,903,942,656 |

Across four sessions, the wider plan adds 9,175,040 bytes of initially committed
hidden storage and 1,903,942,656 bytes of deferred work: 1,913,117,696 bytes,
about 1.782 GiB. Removing a pending obligation is valid only when the actual
selected execution width cannot allocate that work. Narrowing alone recovers
only 9,175,040 initially committed bytes; with every other observation held
constant, **475,361,280 bytes (453.34 MiB)** of the grant shortfall would remain.
No complete-cache-fit claim follows from the scratch reduction.

### Unattributed loading interval

The post-slot host-budget input reconstructs as:

```text
2 * 4053223424 = 8106446848
3810430976 + 1048576 + 4294967296 = 8106446848
38883639296 - 8106446848 = 30777192448
30777192448 - 29501797520 = 1275394928
```

The execution ledger implies 29,501,797,520 committed session bytes. The
remaining **1,275,394,928 bytes** of the host-availability drop are unattributed;
the observations do not prove a leak, allocator rounding or one shared buffer.
Each public session also resizes a host FP32 logits vector to 248,320 values:
993,280 requested bytes per session, or 3,973,120 across four. Native host vectors,
adapter/ledger metadata and `DiskStore` initialization also occur in this interval.
These known allocation sites do not measure or explain the full residual.
Cache-resource construction creates the upstream default 64 HIP stream/event
pairs in `TransferPool` before slots and the post-slot grant. Streams are not
hardware queues, and no byte cost is inferred from their count. Phase observations
in the next changed candidate must distinguish this construction interval from
slot allocation. Independent transfer-resource optimizations require their own
ownership/exhaustion evidence and are outside this admission correction.

## On-demand observation at 0825cdd

The following same-phase values were reported for the deployed on-demand
candidate. They must not be combined with earlier load-stage or later cleanup
readings:

```text
policy=on-demand
phase=post_fingerprint_pre_sessions
hip_free_bytes=40497639424
device_admission_bytes=40497639424
device_host_margin_bytes=0
host_available_bytes=38922952704
cma_free_bytes=0
cma_excluded_bytes=0
cgroup_headroom_bytes=unknown
host_reserve_bytes=4294967296
host_ceiling_bytes=34627985408
selected_limiter=host
memory_capacity_bytes=34627985408
sessions=4
base_bytes=7527108948
total_text_bytes=30108435792
state_committed_bytes=0
cache_committed_bytes=0
vision_committed_bytes=0
outstanding_bytes=0
vision_startup_bytes=0
cache_backing=lazy
cache_reserved_bytes=4294967296
deferred_scratch_bytes=418013184
runtime_margin_bytes=536870912
total_required_bytes=35358287184
state_capacity_bytes=29378134016
deficit_bytes=730301776
refusal_reason=text_state_capacity
result=refused
```

That policy required the full text claims, cache maximum, deferred scratch and
extra runtime margin below the host/device minimum:

```text
30108435792 + 4294967296 + 418013184 + 536870912 = 35358287184
35358287184 - 34627985408 = 730301776 bytes short
```

The corrected text gate uses the existing device-admission result and deducts
deferred execution work once, with the text/MTP session estimate:

```text
40497639424 - 418013184 - 30108435792 = 9971190448 bytes headroom
```

This gate passes without changing any measured input. The platform-aware query
still applies its existing Windows rules; this Linux observation happens to
equal raw HIP free memory. Passing says nothing about later allocations, cache
grant, backing growth or image eligibility.

Under full-claim arithmetic a separate host constraint remains:

```text
38922952704 - 30108435792 - 4294967296 - 418013184 = 4101536432 bytes
```

That is about 3.820 GiB, or 184.47 MiB below the existing 4 GiB host reserve,
before an additional 512 MiB margin. The owner still requires four full-context
sessions and an effective 4 GiB cache; a reduced upstream cache grant does not
qualify that workload. This correction claims no physical allocation saving.

## Earlier fixed-pool requirement

The final admission event supplies a selected capacity of 34,672,033,792 bytes.
It does not identify which of the device budget and host ceiling selected that
minimum. The earlier 40,497,643,520-byte HIP reading and 37,666 MiB host reading
belong to vision preparation, about 49 seconds earlier. Neither they nor the
post-failure cleanup reading can fill in the missing final inputs.

| Final admission item | Bytes |
| --- | ---: |
| Four text/MTP session claims | 30,108,435,792 |
| Deferred shared scratch | 418,013,184 |
| Separate fixed vision pool | 4,294,967,296 |
| Shared retained RAM cache | 4,294,967,296 |
| Runtime margin | 536,870,912 |
| Total remaining requirement | 39,653,254,480 |
| Selected available capacity | 34,672,033,792 |
| Deficit | **4,981,220,688** |

The host ceiling already subtracts the existing 4 GiB OS reserve after CMA
exclusion and cgroup accounting. That reserve and the explicit RAM cache are
different obligations. The logs do not establish duplicate charging.
Hypothetically removing the whole vision deduction still leaves 686,253,392
bytes missing. `--vision-memory-bytes 0` does not perform that experiment: it
selects conservative admission. The fixed pool's guarantee remains unchanged.

`StartupPolicies()` retains its older 40,088,076,288-byte arithmetic fixture,
explicitly labeled historical. A separate regression preserves every number
above and requires refusal. Unequal host/device limits, exact boundaries,
outstanding reservations and already-committed allocations use the production
selector and planner; consumed bytes are never credited back to free memory.

## Fixed-pool ownership audit

`PlanSessionMemory()` now supplies the components used by the real executor's
`SessionBytes()`. Its fixed target fixture reconstructs each byte:

| Per-session component | Bytes |
| --- | ---: |
| Trunk KV | 6,442,450,944 |
| Trunk indexes | 251,661,312 |
| Recurrent, convolution and PLE state | 118,038,528 |
| MTP rollback frontiers | 153,953,280 |
| MTP KV | 536,870,912 |
| MTP indexes | 20,971,776 |
| Retained MTP hidden rows | 368,640 |
| Private draft table | 2,793,472 |
| Control and stable vision descriptor | 84 |
| Total | **7,527,108,948** |

This is already the existing minimum copy width of eight rows with seven draft
tokens. Further copy-width selection cannot recover the recorded deficit.
Context, MTP capability, quantization, cache budgets and all safety reserves
remain unchanged.

The inspected lifetimes do not justify removing another allocation:

- `DeviceModel::Upload()` drains its local upload staging and releases
  conversion sources before returning. The staging pool is not retained
  through final admission.
- Shared executor scratch is allocated before vision preparation. Mutually
  exclusive linear, attention and expert staging already share storage.
  Ngram row caching is deliberately bounded; it is not an accidental full
  model copy.
- Projector upload's local pinned buffer and patch-conversion vector die
  before `vision_prepared`. Image execution scratch remains lazy and funded
  by the vision pool.
- Fingerprinting's four 8 MiB read buffers are local and destroyed before
  final admission. Multiple mappings of the same inode do not prove duplicate
  physical weights. This patch does not change GGUF hashing or identity.
- Execution slots and complete RAM backing are created after admission.
  Their reservations are still uncommitted in the reported phase. Later full
  cache commitment remains required before readiness.
- Deferred scratch conditionally omits existing logits, checkpoint and wide
  buffers. Snapshot-spare headroom deliberately remains reserved even after
  runtime spare allocations; there are no snapshots at startup to deduct.

Loaded weights and shared allocations are already reflected in remaining
capacity. The new counters report explicit retained weight/executor allocation
requests, excluding opaque BLAS/MMQ/driver storage and CPU metadata. They are
ownership diagnostics, not a complete physical-residency measurement or an
extra amount available for admission.

## One admission observation

The opt-in collects host inputs once after fingerprinting, then queries HIP
and the device budget using that same raw host input where needed. Copy-width
selection and final admission reuse this collection. It is one sampling pass,
not an atomic kernel snapshot; `snapshot_collection_us` records its duration.
Conservative admission retains its existing behavior.

`event=vision_admission phase=post_fingerprint_pre_sessions` reports raw HIP
free bytes, the device admission budget and its active host margin, raw host
availability/source, CMA exclusion, cgroup headroom/limiter, OS reserve, host
ceiling, selected limiter, explicit loaded allocations, per-session components,
uncommitted allowances, total requirement and deficit. Initial committed
session/cache/image bytes and outstanding image reservations are zero in this
phase. The event reports refusal reasons without content or credentials.
Unrepresentable arithmetic is marked `overflow`; a plan rejected before it can
be formed reports unavailable totals as `unknown`, with its refusal reason.

The additive host-snapshot and device-query interfaces require the `ext-api`
label and their permanent change record. The numerical state layout, cache
representation and fixed vision-pool semantics do not change.

## Validation scope

On November 10, 2026, the focused CPU build succeeded and all nine selected
CTest regressions passed in 4.80 seconds. The 21 Python harness tests passed
in 0.002 seconds. These are the results of this follow-up, not a rerun of the
earlier complete PR gate. The VM reused its configured `cpu-test` tree and
dependencies under `/tmp`:

```sh
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/tmp/gufo-cpu-python \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset cpu-test --parallel 4 --target \
  text_model_runner_test qwen_vision_admission_test qwen_vision_lifecycle_test \
  qwen38_flash_next_memory_test qwen38_flash_next_runner_options_test \
  admission_memory_test cache_full_backing_test component_text_cache_test \
  text_generation_scheduler_test

env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/tmp/gufo-cpu-python \
  ctest --preset cpu-test --output-on-failure -R \
  '^(text_model_runner_test|qwen_vision_admission_test|qwen_vision_lifecycle_test|qwen38_flash_next\.memory|qwen38_flash_next\.runner_options|admission_memory_test|cache_full_backing_test|component_text_cache_test|text_generation_scheduler_test)$'

python3 tests/functional/test_vision_memory_admission.py
```

Changed C++ files pass `clang-format --dry-run --Werror`. Extension API,
infrastructure generation, extension isolation, documentation and
`git diff --check` checks pass. The whole-tree formatting check still reports
pre-existing violations in unchanged `src/cli/serve/audio_tts_api.cpp:473` and
`tests/cli/openai_chat_test.cpp:6633`; this patch leaves those files untouched.

CPU tests exercise the production host calculation, host/device selector,
startup planner and session component calculation with synthetic observations.
They do not establish that the actual HIP allocator or the target workload
fits. The VM has neither a suitable HIP toolchain nor weights; changed
`runner.cpp`, `engine.cpp`, `executor.cpp` and the real-HIP query wrapper remain
uncompiled here. The changed kernel header shares the existing rollback-size
formula; no kernel arithmetic was changed. Use a capable environment for:

```sh
cmake --preset release
cmake --build --preset release --parallel 4
```

The [single owner workload attempt](VISION-MEMORY-VALIDATION.md#one-targeted-full-workload-attempt)
now distinguishes the narrower text gate, completed startup, effective cache
grant/backing and image refusal/recovery. The historical fixed-pool arithmetic
above remains unchanged; it does not define corrected on-demand acceptance.
Repeated loads without a changed candidate or concrete new hypothesis add no
required evidence. Numerical and physical-capacity qualification remain pending.
