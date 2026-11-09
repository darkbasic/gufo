# Vision admission: VM validation

As of November 9, 2026, the implementation builds and passes the CPU checks
below in a VM without an inference GPU or model weights. No weights were
downloaded, no inference host was accessed, and no hardware qualification ran.
The [accounting analysis](VISION-MEMORY.md) and
[owner-run GPU procedure](VISION-MEMORY-VALIDATION.md) describe the remaining
physical-capacity, performance and numerical checks.

## Results

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
