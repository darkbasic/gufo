# Vision memory admission: owner-run GPU validation

Run this procedure on the ZBook's actual gfx1151 GPU after building the
candidate. **Hardware capacity, performance and numerical image correctness
remain unverified until checked on that GPU.** CPU accounting and simulated
allocations do not establish GPU compilation, physical fit or numerical quality.

The harness owns a separate loopback port and disposable cache. It never
downloads weights, changes kernel parameters, clears production caches or stops
services. Arrange an **owner-controlled maintenance window and unload production
Flash-Next chat before loading the candidate**. Do not load a second Flash-Next
beside production, or reduce context, sessions, cache, MTP or quantization to
turn a refusal into a pass.

## Owner-controlled maintenance window

Before any model-dependent check, the owner must:

1. Record production and required auxiliary services, including their memory
   use. Drain production chat and stop its model-serving process through the
   normal operational procedure; verify its model memory has been released.
2. Retain EmbeddingGemma 2 on the same APU, Open WebUI and other auxiliary
   services required by the target budget. Leave HugeTLB, GTT/TTM and kernel
   settings unchanged. Stopping auxiliaries invalidates this workload test.
3. Preserve production models, configuration, launcher and cache files. Use the
   candidate binary, separate port and new disk-backed directory below.
4. After the candidate exits, restore production chat through the normal
   procedure and verify health. Retain all validation evidence until reviewed.

The harness cannot arrange this window or identify the deployment's services.
Its cleanup terminates only its own candidate child. If a maintenance window
is unavailable, leave execution pending; building needs no model load.

## Build and record inputs

The real HIP release build of this on-demand change remains pending in the
development VM, which has no suitable ROCm/HIP toolchain. Prior candidate
HIP/container results do not compile this diff. On a capable environment:

```sh
cmake --preset release
cmake --build --preset release --parallel 4
```

With the pinned Nix development toolchain, use:

```sh
nix develop -c cmake --preset release
nix develop -c cmake --build --preset release --parallel 4
```

The production runner and executor wiring and real HIP image/cache allocation
paths need that build. CPU lifecycle tests use simulated HIP providers; they
do not compile GPU kernels or validate hipBLAS and the real allocator. Keep
compiler/dependency versions fixed for any later timing comparison.

Set existing local paths; the first model shard finds its siblings. The MTP and
projector sidecars may live elsewhere. Choose a disk filesystem with sufficient
space for the 50 GiB disposable cache and reports:

```sh
export GUFO_BINARY=/absolute/path/to/candidate/gufo
export GUFO_MODEL=/absolute/path/to/first-UD-Q4_K_XL-shard.gguf
export GUFO_MTP=/absolute/path/to/mtp-shared-Q8_0.gguf
export GUFO_PROJECTOR=/absolute/path/to/mmproj-BF16.gguf
export GUFO_VALIDATION_PARENT=/absolute/path/on/a/disk/filesystem
case "$(findmnt -n -o FSTYPE -T "$GUFO_VALIDATION_PARENT")" in
  tmpfs|ramfs|"") echo "Choose an existing disk-backed validation directory" >&2; exit 1 ;;
esac
export GUFO_VALIDATION_ROOT
GUFO_VALIDATION_ROOT=$(mktemp -d "$GUFO_VALIDATION_PARENT/gufo-vision-validation.XXXXXXXX")
"$GUFO_BINARY" --version > "$GUFO_VALIDATION_ROOT/version.txt"
git rev-parse HEAD > "$GUFO_VALIDATION_ROOT/source-revision.txt"
```

Record weight revisions/hashes, GPU/driver version and other processes' memory
use. The harness records its command and source hash; it does not hash every
model shard automatically. Start any installed AMD memory monitor now if Linux
DRM usage files are unavailable, avoiding a repeat solely for missing peaks.

## One targeted full-workload attempt

Run the following once from the source checkout after the maintenance steps.
Port 18092 must be unused. The harness creates `run/disposable-cache`, performs
the workload in one owned server, and restarts that server once only if needed
to verify disk restoration after successful first-phase checks. It does not
require an initial diagnostic-only load or another load after startup refusal.

```sh
python3 tests/functional/vision_memory_admission.py \
  --output "$GUFO_VALIDATION_ROOT/run" --port 18092 \
  --prompt-tokens 229000 --max-tokens 32000 --timeout 14400 -- \
  "$GUFO_BINARY" serve llm \
  --model "$GUFO_MODEL" --mtp-model "$GUFO_MTP" \
  --mmproj "$GUFO_PROJECTOR" \
  --sessions 4 --context 262144 --speculative mtp --draft-tokens 7 \
  --cache-ram-bytes 4294967296 --cache-disk-bytes 53687091200 \
  --vision-memory-policy on-demand \
  --think auto --reasoning-effort xhigh --preserve-thinking on
```

Retain `event=vision_admission policy=on-demand` at
`phase=post_fingerprint_pre_sessions`. It must show `vision_startup_bytes=0`,
the final raw HIP/device and host/CMA/cgroup inputs, reserves, selected limiter,
loaded allocation scope, total requirement and deficit. The
[recorded capacity](VISION-MEMORY-ADMISSION.md) still yields a 686,253,392-byte
on-demand deficit; the change does not manufacture a fit. Do not substitute an
earlier `vision_prepared` or later cleanup `load_failed` reading.

If startup refuses, **stop this attempt and return its log and report**. The
new evidence is the exact final pair of limits and whether the on-demand total
fits with the required auxiliary services present. Do not repeat full loads,
clear caches, change budgets or stop auxiliary services to seek readiness.
Restore production chat only after the candidate has exited. A future attempt
needs a changed candidate or a concrete capacity hypothesis to test.

If admitted, allow substantial time: four roughly 229,000-token prefills and up
to 32,000 output tokens each intentionally exercise the full configured window.
The model can stop before its output cap. Short smoke work does not replace
this qualification. The harness requires:

- Four execution slots, context 262,144, an available projector, loaded MTP and
  actual draft work. Both loads must grant the full 4,294,967,296-byte RAM cache
  with `backing=lazy`, and the component cache must report its full capacity.
  Startup backing may be partial; eager 4 GiB commitment is not required.
- Four distinct `/slots` task IDs occupying four sessions together, with each
  task's prefill/decode counters advancing during that overlap. Accepted client
  connections and queued requests alone cannot pass.
- A small image succeeds, publishes a checkpoint to the owned disk cache and
  reuses it from RAM after four other active histories. Following the long
  text requests, responses must show occupied RAM checkpoint backing; free
  preallocated storage alone cannot establish a populated cache.
- Two text tasks and a fresh image-bearing task advance together under mapped
  task IDs. The image task must decode, proving consumption of its cold image
  prefix while the text peers remain active. The fourth slot stays free for
  the pressure request; this does not replace the four-text-session check.
- While those tasks remain active, the image pressure request receives HTTP
  503 `resource_exhausted` before SSE. The same three task IDs advance afterward,
  all complete successfully, and subsequent text work succeeds. Reassignment,
  sequential success or early peer completion fails the overlap check.
- After the one restart, the image prefix restores from disk with reused
  tokens, its greedy continuation matches the RAM result, and text still works.

The pressure request defaults to five 4096 × 4096 images. On-demand capacity
depends on the protected obligations and retained peers, so this count is not
a guaranteed exhaustion threshold. `--pressure-images N` before the separator
selects 1–15 images within context. If the request is admitted, refusal
qualification fails and retains that evidence; do not report a refusal pass or
rerun the whole model without deciding what new pressure evidence is needed.

## Evidence and pending numerical qualification

Inspect `report.json`, each response, both server logs, `mixed-samples.jsonl`
and `samples.jsonl`/`restart-samples.jsonl`. The mixed samples retain task
attribution and progress around refusal. Background samples retain `/slots`,
`/metrics`, host available RAM/swap, process RSS/high-water RSS/swap and available
Linux DRM VRAM/GTT readings. Missing DRM/monitor readings leave device peaks
unmeasured; process RSS does not establish them on unified memory.

Review each request's `usage.gufo` and timings: TTFT, prefill/decode time,
throughput, cache work and refusals. Set acceptable peak-memory/swap limits
before the run. Sustained swap growth or unexpected refusals fail capacity
qualification. Per-request performance comparison requires matched histories,
toolchain and cache state, using both 5% and 10 ms gates; noisy or unmatched
timings remain inconclusive. Never average a failure into a pass.

Synthetic images and RAM/disk continuation equality do not replace the
independent image/position numerical oracle. Hardware numerical qualification
remains pending even if this capacity harness passes. When capacity permits,
select the image, layered-prefill, long-context and cache checks described in
[functional testing](../../../../tests/functional/README.md), with local
weights and the same MTP settings. Build the focused native targets on HIP with:

```sh
cmake --preset gpu-test
cmake --build --preset gpu-test --parallel 4 --target \
  qwen38_flash_next_continuation_adapter_test qwen38_flash_next_session_test
```

Run numerical checks and the standard benchmark only after the owned server
exits, in an owner-approved window; they are separate qualifications, not an
automatic sequence of model loads after a refusal. No `--suite all`, downloads
or service changes are part of this procedure.

## Policy compatibility

An external Python launcher must forward `--vision-memory-policy on-demand`
to the native `serve llm` command and omit `--vision-memory-bytes`. The launcher
is not changed by this patch. To restore conservative admission, remove both
options or pass `--vision-memory-policy conservative` without a positive byte
pool. Legacy `--vision-memory-bytes 0` also remains conservative.

A positive `--vision-memory-bytes N` without the new policy retains the fixed
pool and its startup/backing guarantee. Explicit on-demand rejects any byte
option, including zero; explicit conservative rejects a positive byte option.
Keep the exact result directory and disposable cache until review finishes,
then remove only that directory when you choose to discard the evidence.
