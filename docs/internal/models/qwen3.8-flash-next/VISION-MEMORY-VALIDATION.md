# Vision memory admission: owner-run GPU validation

Run this procedure on the ZBook's actual gfx1151 GPU after building the
candidate. **Hardware capacity, performance and numerical image correctness
remain unverified until these checks run.** CPU tests of the accounting and
simulated allocations do not establish a physical-memory or numerical pass.

This procedure creates a separate process, loopback port and disposable cache.
It never downloads weights, changes kernel parameters, clears another cache or
stops another service. The existing deployment stays running; if its residency
leaves insufficient memory for this second model, retain the refusal and arrange
a separate owner-controlled validation window. Do not lower context, sessions,
cache capacity, MTP settings or quantization to turn a refusal into a pass.

## Build and record inputs

Use the same compiler/dependency versions for candidate and matched timing
controls. On a capable build machine:

```sh
cmake --preset release
cmake --build --preset release --parallel 4
cmake --preset gpu-test
cmake --build --preset gpu-test --parallel 4 --target \
  qwen38_flash_next_continuation_adapter_test qwen38_flash_next_session_test
```

The CPU tests compile production DeviceInput lifecycle and HIP cache-backing
sources against simulated HIP providers. They do not validate hipBLAS or the
real allocator. The Flash-Next runner/engine/executor and Qwen encoder/injection
kernels still require the GPU build, which also compiles the cache-backing
sources against the real HIP headers and runtime.

Set your existing paths; the first model shard finds its siblings. The two
sidecars may live anywhere. Use a new result directory for every attempt.

```sh
export GUFO_BINARY=/absolute/path/to/candidate/gufo
export GUFO_MODEL=/absolute/path/to/first-UD-Q4_K_XL-shard.gguf
export GUFO_MTP=/absolute/path/to/mtp-shared-Q8_0.gguf
export GUFO_PROJECTOR=/absolute/path/to/mmproj-BF16.gguf
export GUFO_VALIDATION_ROOT
GUFO_VALIDATION_ROOT=$(mktemp -d /tmp/gufo-vision-validation.XXXXXXXX)
"$GUFO_BINARY" --version > "$GUFO_VALIDATION_ROOT/version.txt"
"$GUFO_BINARY" diagnose > "$GUFO_VALIDATION_ROOT/diagnose.txt"
git rev-parse HEAD > "$GUFO_VALIDATION_ROOT/source-revision.txt"
```

Record the build toolchain, weight revisions/hashes, GPU/driver version and
other processes' memory use. The harness records its command and source hash,
but does not hash all multi-gigabyte model shards automatically.

## Capacity, real concurrency, image refusal and cache lifecycle

Run from the source checkout. Port 18092 must be unused; the harness refuses an
occupied port. It creates `run/disposable-cache`, starts its own child, drains
and restarts that child once for disk restoration, and leaves all evidence.

```sh
python3 tests/functional/vision_memory_admission.py \
  --output "$GUFO_VALIDATION_ROOT/run" --port 18092 \
  --prompt-tokens 229000 --max-tokens 32000 --timeout 14400 -- \
  "$GUFO_BINARY" serve llm \
  --model "$GUFO_MODEL" --mtp-model "$GUFO_MTP" \
  --mmproj "$GUFO_PROJECTOR" \
  --sessions 4 --context 262144 --speculative mtp --draft-tokens 7 \
  --cache-ram-bytes 4294967296 --cache-disk-bytes 53687091200 \
  --vision-memory-bytes 4294967296 \
  --think auto --reasoning-effort xhigh --preserve-thinking on
```

Allow substantial time: four roughly 229,000-token prefills and up to 32,000
output tokens each are intentional full-context capacity work. Output limits
remain inside the 262,144-token window. The model may stop before its output
limit. A shorter smoke run is useful for debugging, but does not replace this
qualification.

The harness requires all of the following:

- `/props` reports four slots, context 262,144 and vision; logs report loaded
  MTP and `event=vision_cache_grant ... granted_bytes=4294967296` before and
  after restart. Readiness also requires the opt-in's full checkpoint backing
  commitment: `event=backing_prepared` must report `committed_bytes` at least
  `required_payload_bytes`. Successful text responses contain actual draft work.
- Four distinct `/slots` task IDs hold all four execution sessions together;
  every task's prefill/decode counters advance during that overlap. Four
  accepted client connections or four queued requests alone do not pass.
- A small synthetic image succeeds. Its checkpoint is published to the owned
  disk cache, and reuse after four other active histories reports the RAM tier.
- Five synthetic 4096 × 4096 images exceed the 4 GiB image budget while fitting
  the context limit. A request asking for streaming receives HTTP 503 with
  `resource_exhausted` before SSE. A following text request succeeds.
- After restart, the image request reports a disk hit and reused tokens. Its
  greedy continuation matches the RAM result; a following text request succeeds.

Inspect `report.json`, each request's response, both server logs and
`samples.jsonl`/`restart-samples.jsonl`. The samples retain `/slots`, `/metrics`,
host available RAM/swap, process RSS/high-water RSS/swap, and any Linux DRM
VRAM/GTT usage files available. Missing DRM readings mean device peaks are
unmeasured; use the installed AMD monitoring tool to capture those peaks while
rerunning. Do not infer device peaks from process RSS on unified memory.

Evaluate each request's `usage.gufo`/`timings`: TTFT, prefill/decode time,
throughput, cache tier/work and refusal reason. Record peak host/device usage
and swap growth. These are observations, not automatic performance approval.
Set acceptable peak/swap limits for the machine before the run; sustained swap
growth or unexpected resource refusals fail capacity qualification. Compare
matched request histories against the same toolchain/control revision using
the repository's per-request 5% and 10 ms timing gates. Keep all attempts;
unmatched or noisy measurements remain inconclusive, never averaged into a pass.

## Numerical and scheduling checks

The capacity harness uses synthetic images and checks continuation equality;
it does not replace an independent image encoder/position numerical oracle.
Run the existing focused image, long-context, cache and layered-prefill tests
on the candidate with local weights. The provider's typed API retains its
conservative default, so test both native numerical paths and the opt-in HTTP
path above.

The native tests discover an adjacent canonical projector. To use your separate
projector without modifying its directory, create a temporary symlink view:

```sh
python3 - <<'PY'
import os
from pathlib import Path
view = Path(os.environ['GUFO_VALIDATION_ROOT']) / 'model-view'
view.mkdir()
model = Path(os.environ['GUFO_MODEL'])
for shard in model.parent.glob('*.gguf'):
    if shard.name != 'mmproj-BF16.gguf':
        (view / shard.name).symlink_to(shard.resolve())
(view / 'mmproj-BF16.gguf').symlink_to(Path(os.environ['GUFO_PROJECTOR']).resolve())
PY
GUFO_VIEW_MODEL="$GUFO_VALIDATION_ROOT/model-view/$(basename "$GUFO_MODEL")"
build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_continuation_adapter_test \
  --model "$GUFO_VIEW_MODEL" --mtp-model "$GUFO_MTP" --image-only
build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_session_test \
  --model "$GUFO_VIEW_MODEL" --mtp-model "$GUFO_MTP" --batch-only
```

The selected checks cover image attachment/position restoration in AR and MTP,
snapshot continuation logits and concurrent execution. Retain their output.
Run the standard native speed benchmark separately after the owned server has
exited, matching its control's toolchain and cache history:

```sh
"$GUFO_BINARY" bench --model "$GUFO_MODEL" \
  --speculative mtp --mtp-model "$GUFO_MTP" --draft-tokens 7 \
  --n-prompt 2048 --n-gen 128 --concurrency 1,4 \
  > "$GUFO_VALIDATION_ROOT/native-bench.txt"
```

Follow `tests/functional/README.md` for explicitly selected `image-inputs`,
`image-detail`, `long-context` and `cache` HTTP suites and matched phase timing.
Do not run `--suite all` or download models for this change. The new flag belongs
to `serve llm`; the native benchmark keeps its existing interface.

## Return to conservative admission

For the normal native command, omit `--vision-memory-bytes` or pass
`--vision-memory-bytes 0`. A Python deployment launcher must forward the single
`--vision-memory-bytes N` option; no launcher modification is included here.
The conservative four-session configuration may refuse startup as before.

Keep the result directory and disposable disk cache until review finishes.
Remove only that exact directory when you choose to discard the evidence.
