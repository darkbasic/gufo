# Vision admission: recorded ZBook refusal

This page preserves the observed refusals from the fixed-pool and earlier
on-demand implementations as of November 10, 2026. The
[current policy](VISION-MEMORY.md) restores upstream text/cache stage boundaries;
the broader on-demand refusal is historical evidence, not its acceptance rule.
Physical fit and GPU execution remain unverified by the VM.

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
