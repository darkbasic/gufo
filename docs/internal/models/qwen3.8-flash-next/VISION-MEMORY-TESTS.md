# Vision admission: VM validation

The results below distinguish the current scope correction from earlier
iterations in a VM without an inference GPU or model weights. No weights were
downloaded, no inference host was accessed, and no hardware qualification ran.
The [accounting analysis](VISION-MEMORY.md) and
[owner-run GPU procedure](VISION-MEMORY-VALIDATION.md) describe the remaining
physical-capacity, performance and numerical checks.

## Corrective scope follow-up

The correction starts from clean HEAD
`0825cdd0977a5c3e8749f55c8435b628f2bf9500` and preserves the earlier fixes.
Its tests distinguish the device text-state gate from optional cache growth
and runtime image admission. The recorded 730,301,776-byte combined refusal
is retained as historical evidence; the same inputs pass the narrower text
gate with 9,971,190,448 bytes of headroom.

On November 10, 2026, the focused native build and CPU tests passed using
production accounting, resource hooks and ownership with synthetic observations.
The same production cache-policy helpers are used by the runner and host tests;
reduced grants and unchanged upstream lazy/pressure behavior are tested without
compiling the HIP-guarded runner.

| Corrective check | Result |
| --- | --- |
| Focused CPU build, ten targets below | Passed |
| Focused native CTest regressions | 10/10 passed; 4.47 seconds |
| Final native PR gate, including lazy rollback and cache-hook tests | 115/115 passed; 34.14 seconds |
| Harness CPU configuration/evidence/orchestration tests | 36/36 passed; 0.003 seconds |
| Python compilation | Passed |
| Extension API check | Passed; 0.2.24, 130 surface paths |
| Generated infrastructure check | Passed |
| Documentation check | Passed; 165 Markdown files, 712 local links |
| Targeted ASan/UBSan with leak detection | Admission 0.020 seconds; execution/snapshot accounting 0.012 seconds; DeviceInput/prompt lifecycle 0.043 seconds; all passed |

The harness additions distinguish text eligibility from readiness, retain
legitimate cache trims separately from owner qualification and preserve failed
attempt logs. Cache-pressure checks reject owner qualification after a backing
limit on either load, without confusing expected image refusals with cache
pressure or erasing a limit after later successful growth.
The focused tests cover default/fixed/on-demand policy, recorded
and exact boundaries, pending versus committed observations, concurrent images,
optional snapshots and backing, retained history/restoration and failure cleanup.
The model's production execution-commitment helper also checks deferred MTP
rollback growth, partial allocation failure, trimming and reset without letting
optional work consume accepted but unmaterialized rows.

Commands use the existing configured CPU build and local dependency paths;
no production compiler flags or dependencies were changed:

```sh
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/tmp/gufo-cpu-python \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset cpu-test --parallel 4 --target \
  component_text_cache_test admission_memory_test text_model_runner_test \
  text_generation_scheduler_test cache_full_backing_test cache_ledger_test \
  qwen_vision_admission_test qwen_vision_lifecycle_test \
  qwen38_flash_next_runner_options_test qwen38_flash_next_memory_test

env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/tmp/gufo-cpu-python \
  ctest --preset cpu-test --output-on-failure -R \
  '^(component_text_cache_test|admission_memory_test|text_model_runner_test|text_generation_scheduler_test|cache_full_backing_test|cache_ledger_test|qwen_vision_admission_test|qwen_vision_lifecycle_test|qwen38_flash_next\.runner_options|qwen38_flash_next\.memory)$'

env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PYTHONPATH=/tmp/gufo-cpu-python \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset pr --parallel 4

python3 tests/functional/test_vision_memory_admission.py
python3 -m py_compile tests/functional/vision_memory_admission.py tests/functional/test_vision_memory_admission.py
python3 tools/ci/check-extension-api.py
python3 tools/ci/generate-infrastructure.py --check
python3 tools/ci/check-docs.py
```

The initial focused logs are `/tmp/gufo-920-corrected-build.log` and
`/tmp/gufo-920-corrected-focused-ctest.log`. The first broader PR gate exposed
one extension-isolation failure from the new model-owned cache-policy helper;
114 of 115 tests passed. After correcting its build ownership, the final gate
passed 115/115; its log is `/tmp/gufo-920-corrected-pr-final.log`.

The whole-tree `python3 tools/ci/check-format.py` reports only the unchanged
baseline violations in `src/cli/serve/audio_tts_api.cpp:473` and
`tests/cli/openai_chat_test.cpp:6633`. Those unrelated files remain untouched;
this is not a claim that the pinned Nix formatting gate passed.

All three sanitizer runs use GCC with
`-std=c++20 -O1 -g -UNDEBUG -DGUFO_CACHE_TESTING -pthread
-fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all`,
and `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` plus
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. The admission and model memory
tests compile their test source with production `admission.cpp` and `ledger.cpp`;
the model test includes the real execution/snapshot ownership helpers.
`/tmp/gufo-920-corrected-sanitizers.log` retains both exact compiler commands and
the final runs after the lazy-rollback change.

The lifecycle build also instruments production `device_input.cpp` and
`prompt.cpp`, includes `tests/families/qwen/fake_hip`, and links existing CPU
support libraries. Its full compiler/link command and result are in
`/tmp/gufo-920-corrected-lifecycle-sanitizer.log`. None instrument the complete
dependency tree or compile real HIP execution; there were no sanitizer or leak
diagnostics in these three runs.

The changed HIP-guarded translation units
`src/models/qwen38_flash_next/runner.cpp`,
`src/models/qwen38_flash_next/kernels/rocm/executor.cpp` and
`src/models/qwen38_flash_next/kernels/rocm/batch.cpp` require a real HIP build.
Production DeviceInput and cache-backing host logic uses simulated HIP in CPU
tests; that does not compile their real HIP dependencies. The release build
and all hardware execution remain pending. Historical native and sanitizer
counts below do not validate this correction.

## Prior on-demand follow-up

As of November 10, 2026, the earlier on-demand implementation and harness passed
the CPU checks below, starting from `d3e9031bee83b27e9babd75b371cbad2ff85ea94`
and subsequently recorded in `0825cdd0977a5c3e8749f55c8435b628f2bf9500`.
These results do not establish real HIP
compilation, numerical correctness or the target machine's memory fit.

| Historical check | Result |
| --- | --- |
| Build `qwen38_flash_next_runner_options_test` with `cpu-test` | Passed |
| CTest `qwen38_flash_next.runner_options` | 1/1 passed; 0.00 seconds |
| Harness CPU configuration/evidence/orchestration tests | 30/30 passed; final run 0.003 seconds |
| Python compilation, focused C++ formatting and metadata JSON syntax | Passed |
| Combined focused CPU build and ten CTest regressions listed below | Passed; 10/10, 4.52 seconds |
| Additional lazy-backing concurrent-image/allocation-failure case | Rebuilt backing target; focused CTest 1/1 passed, 0.01 seconds |
| Final native PR gate after ownership fixes | 115/115 passed; 34.07 seconds |
| Targeted ASan/UBSan with leak detection | Admission, snapshot accounting, DeviceInput/pixel ownership and cache backing passed; no diagnostics |

The option test uses generated production provider metadata and the production
validation helpers. It covers default/zero/fixed-pool compatibility, explicit
on-demand, projector requirements, conflicting options and unsupported
providers/commands. Harness tests retain concurrent/mixed task attribution,
post-refusal progress and recovery checks; new cases distinguish a full lazy
cache grant from eager commitment and occupied backing from free backing.
The final lifecycle tests exercise production `Prompt::RetainPixels()` through
prompt/image copies and value assignment. The ledger tests refuse a second
execution/cache owner for an on-demand epoch. The real backing allocator grows
from partial to full backing while an image holds all extra capacity, including
injected registration failure and recovery.

Commands, with the dependency environment described below:

```sh
cmake --build --preset cpu-test --parallel 4 --target qwen38_flash_next_runner_options_test
ctest --test-dir build/cpu-test -R '^qwen38_flash_next\.runner_options$' --output-on-failure
python3 tests/functional/test_vision_memory_admission.py
python3 -m py_compile tests/functional/vision_memory_admission.py tests/functional/test_vision_memory_admission.py
clang-format --dry-run --Werror src/models/qwen38_flash_next/runner.hpp tests/models/qwen38_flash_next/runner_options_test.cpp
python3 -m json.tool src/models/qwen38_flash_next/extension.json >/dev/null

cmake --build --preset cpu-test --parallel 4 --target \
  component_text_cache_test admission_memory_test text_model_runner_test \
  text_generation_scheduler_test cache_full_backing_test cache_ledger_test \
  qwen_vision_admission_test qwen_vision_lifecycle_test \
  qwen38_flash_next_runner_options_test qwen38_flash_next_memory_test
ctest --preset cpu-test --output-on-failure -R \
  '^(component_text_cache_test|admission_memory_test|text_model_runner_test|text_generation_scheduler_test|cache_full_backing_test|cache_ledger_test|qwen_vision_admission_test|qwen_vision_lifecycle_test|qwen38_flash_next\.runner_options|qwen38_flash_next\.memory)$'

# After adding the production lazy-backing integration case:
cmake --build --preset cpu-test --target cache_full_backing_test --parallel 2
ctest --preset cpu-test -R '^cache_full_backing_test$' --output-on-failure

# Final wider CPU contract gate, after the ownership fixes:
cmake --build --preset pr --parallel 4
```

The final gate log is `/tmp/gufo-920-on-demand-pr.log`. Targeted sanitizer
compiler commands and results are retained in
`/tmp/gufo-920-on-demand-sanitizers.log`; all four ran with
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. They instrument the production
admission/ledger, prompt ownership/DeviceInput and backing sources as applicable;
other dependencies use the prebuilt CPU libraries. They do not instrument the
real HIP executor or establish full-tree sanitizer coverage.

The real HIP release build and hardware execution remain pending under the
[owner procedure](VISION-MEMORY-VALIDATION.md). This VM has no `hipcc` or Nix,
no `/opt/rocm` or `/usr/lib/rocm`, and no local ROCm container image. The sections
below preserve earlier fixed-pool results; their counts are not new on-demand
results. The whole-tree formatting check still flags only the two unchanged
files listed below; all 18 changed C++/HIP files pass their formatting check.

## Validation review follow-up

As of November 9, 2026, the focused review started from a clean tree at
`e45e0d79932455c1ff6ea3178d82a217d22e70ed`. It changed tests and validation
instructions only; admission policy and target settings are unchanged.

The simulated allocator now checks aggregate simultaneously live dynamic bytes
per owning budget, separately from typed stable descriptors. Negative cases
detect individually fitting allocations whose sum exceeds the reservation,
cross-budget borrowing, descriptor-sized dynamic allocations and premature
release. Encoder fixtures include scratch, pinned upload, retained embeddings
and replacement peaks, with failures injected at each stage.

The hardware harness now requires a cold image request and two text requests
to advance together under mapped scheduler task IDs, including image decoding.
After an oversized image refusal, the same tasks must advance again and finish
successfully before text recovery can pass. CPU tests validate this evidence
logic and orchestration; they do not execute the mixed workload on hardware.

| Follow-up check | Result |
| --- | --- |
| Rebuild `qwen_vision_lifecycle_test` with `cpu-test` | Passed |
| Six focused CTest cases listed below | 6/6 passed; 4.43 seconds wall time |
| Hardware-harness CPU evidence/orchestration tests | 21/21 passed; 0.002 seconds |
| Targeted lifecycle ASan/UBSan with leak detection | Passed; 0.033 seconds, no diagnostics |
| Python compilation and harness `--help` | Passed |
| Changed C++ formatting, documentation and `git diff --check` | Passed |

The sanitizer command and results are retained in
`/tmp/gufo-vision-lifecycle-review-sanitizers.log`. It instruments the lifecycle
test, production DeviceInput, admission and ledger sources; other libraries
remain the prebuilt CPU dependencies. The full CPU build and 114-test PR gate
below are earlier implementation results, not reruns for this focused review.
No suitable ROCm/HIP toolchain is installed here: the real HIP release build
and all hardware execution remain pending under the
[owner-controlled maintenance procedure](VISION-MEMORY-VALIDATION.md).

## Initial implementation results

| Check | Result |
| --- | --- |
| Complete `cpu-test` build | Passed with GCC 15.3, HIP disabled |
| Native `check-pr` gate | 114/114 passed; 57.79 seconds wall time |
| `qwen_vision_admission_test` | Passed; checked startup arithmetic, preparation, history, overflow, retention, failure and concurrency |
| `qwen_vision_lifecycle_test` | Passed; production DeviceInput lifecycle compiled with simulated allocations/copies/encoder |
| `cache_full_backing_test` | Passed; production backing allocator and cache resources compiled with simulated HIP registration/streams |
| Component cache regression | Passed; real RAM reuse and disk restart/restore with logical image claims and small buffers; added case under 0.01 seconds |
| Scheduler/provider regressions | Passed; admission refusal, subsequent text recovery, MTP mocks, deadlines, cancellation and CLI/provider validation |
| Hardware-harness evidence tests | 6/6 passed; harness itself was not run |
| Targeted ASan/UBSan with leak detection | Admission, production DeviceInput lifecycle and production full-cache backing all passed without diagnostics |
| Documentation, extension API and family dependency gates | Passed; extension API derives revision 0.2.21 from its change records |
| Changed C++/HIP formatting and `git diff --check` | Passed |

The new admission, lifecycle and backing CTest cases each take at most
0.01 seconds here. Tests use logical byte counts and small physical buffers;
none allocate the multi-GiB model or image bounds.

The targeted sanitizer builds instrumented `admission.cpp`, `device_input.cpp`,
`ledger.cpp`, `component_cache_resources.cpp`, `committed_backing.cpp`,
`host_allocations.cpp` and `transfer_pool.cpp` as applicable to each test.
Other dependencies used the prebuilt CPU libraries; this is not full-tree
sanitizer coverage. Runs took 0.020, 0.023 and 0.017 seconds respectively, with
leak detection and halt-on-error enabled. Exact compiler commands and results
are retained in `/tmp/gufo-vision-sanitizers.log`.

The whole-tree shared formatting command was also run with the available
clang-format 22.1.8. It flags two unchanged files:
`src/cli/serve/audio_tts_api.cpp` and `tests/cli/openai_chat_test.cpp`.
Formatting their unchanged `HEAD` contents reproduces both failures. Those
unrelated files remain untouched. Nix is unavailable in this VM, so this is
not a claim that the pinned Nix formatting gate passed.

## Reproduce the CPU checks

With the documented development dependencies installed:

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --parallel 4
cmake --build --preset pr --parallel 4
python3 tests/functional/test_vision_memory_admission.py
python3 tools/ci/check-docs.py
python3 tools/ci/check-extension-api.py
python3 tools/ci/check-extension-isolation.py --fail-on-family-graph
python3 tools/ci/check-format.py
```

For focused iteration after the build:

```sh
ctest --preset cpu-test --output-on-failure -R \
  '^(qwen_vision_admission_test|qwen_vision_lifecycle_test|cache_full_backing_test|component_text_cache_test|text_generation_scheduler_test|qwen38_flash_next\.runner_options)$'
build/cpu-test/component_text_cache_test --vision-admission-only
```

This VM lacked system development packages. Development/runtime dependencies
were unpacked into `/tmp/gufo-cpu-deps`, and Python packages into
`/tmp/gufo-cpu-python`; no system or repository compiler flags were changed.
Its build/test environment used:

```sh
export LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu
export PYTHONPATH=/tmp/gufo-cpu-python
export PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps
export PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig
```

The configured prefix and explicit system ICU/curl/zlib/png library paths are
local dependency discovery, not production toolchain or optimization changes.
Build and PR-gate logs from this run are `/tmp/gufo-vision-build.log` and
`/tmp/gufo-vision-check-pr.log`.

## What the VM did not validate

The real HIP implementation of these changed paths was not compiled:

- `src/models/qwen38_flash_next/runner.cpp`, `engine.cpp` and
  `kernels/rocm/executor.cpp`, including startup wiring and eager MTP rollback.
- `src/families/qwen/vision/encoder.hip` and `device_input.hip`.
- The real-runtime build of `device_input.cpp` and
  `src/core/hip/component_cache_resources.cpp`; their host logic was compiled
  and exercised with isolated HIP test doubles.

No model numerical outputs, physical HIP allocation peaks, real hipBLAS
workspace, four-session GPU execution, TTFT or throughput were measured. Use
the release/GPU build commands and separate procedure linked above. CPU success
establishes the accounting contracts, not a hardware or numerical quality pass.
