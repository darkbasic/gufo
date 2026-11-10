# Vision memory accounting

The serving option `--vision-memory-bytes N` makes Flash-Next image admission
request-aware within one bounded model-wide budget. Zero preserves conservative
admission. This is an internal account of issue #920; usage is on the
[model page](../../../models/qwen3.8-flash-next.md), and hardware qualification
has a [separate procedure](VISION-MEMORY-VALIDATION.md).
The [recorded ZBook admission audit](VISION-MEMORY-ADMISSION.md) supersedes the
historical capacity estimate below: the actual target still has a
4,981,220,688-byte deficit, with no justified savings identified by that audit.

## Root cause and estimates

`Model::SessionBytes()` in `src/models/qwen38_flash_next/engine.cpp` adds
`context * (hidden_size * sizeof(float) + 3 * sizeof(int32_t)) + 64` when the
projector is present. `QwenFlashNextTextRunner::ResourceClaim()` supplies that
claim to `ValidateRunner()` in `src/cli/serve/text_model_runner.cpp`, which
requires all execution sessions to fit before allocating them. DeviceInput's
positions and embeddings are lazy, so a text request never materializes that
full vision term.

For 262,144 tokens and width 2560 the allowance is 2,687,500,352 bytes. The
issue's four full claims total 40,858,436,944 bytes, exceeding its reported
40,088,076,288-byte capacity. Keeping the 64-byte stable descriptor allowance,
four text claims instead total 30,108,435,792 bytes. A 4 GiB shared vision pool,
the full 4 GiB RAM cache and 512 MiB runtime margin bring that estimate to
39,235,241,296 bytes. This historical arithmetic does not reproduce the later
host/device minimum. Actual startup also requires the current host/cgroup
ceiling and all deferred scratch; the recorded deployment fails that check.

The conservative `SessionBytes()` remains unchanged. Its other consumers
include MTP copy-width selection, native continuation probes and allocation
bound tests. The serving opt-in and its copy-width selector use the distinct
`TextSessionBytes()` estimate. Checkpoint component sizes and persistent image
layout/identity representations keep their existing contracts.

## Allocation ownership

| Storage | Owner and lifetime | Admission |
| --- | --- | --- |
| Target weights, shared Q8 MTP weights, BF16 projector weights | Loaded model; projector uploaded before admission | Already resident before capacity query |
| KV/index rows, recurrent/PLE state, MTP hidden/draft rows, stable RoPE descriptor | Each execution session | Complete text base claim; opt-in materializes its per-session rollback allowance before readiness |
| Wide-copy rollback, batch logits, checkpoint arena and snapshot spares | Shared executor; some allocations deferred | Existing deferred-scratch allowance retained |
| Resized RGB history | Prepared request and attached session; CPU image cache may share pixels | Bounded preparation reservation before decode/resize; retained pixel claim follows the prepared context |
| GPU image embeddings | DeviceInput, one per prepared history image; lazy encoding | All history embeddings included before Configure allocates |
| GPU positions plus host positional upload vector | DeviceInput; positions retain capacity across image reuse | Both copies included; text transition drains users and releases positions |
| Encoder pixels, hidden/work/normalized/QKV/attention/score/probability/output arrays and pinned upload | One serialized Encode call | Largest image's full temporary working set included |
| Retained private snapshots | Snapshot allocator | Growth protects the vision pool and deferred work against both host and HIP capacity; cache backing is already committed |
| RAM checkpoint backing and disk transfer staging | Existing component cache | Full explicit grant is committed before readiness or startup fails; disk-only backing is separately reserved |

The fixed 512 MiB margin remains outside text, vision and cache budgets. It
includes the existing 128 MiB CPU preprocessing cache, bounded staging/metadata
and opaque driver/library overhead. It is a conservative policy margin, not a
measurement of a particular driver's working set.

## Reservations and reuse

`vision/admission.cpp` owns checked arithmetic and a dedicated instance of the
existing `cache::ResourceLedger`. The pool's fixed capacity is set before any
execution slots or cache backing are created. Reservations include outstanding
promises, so two request threads cannot independently spend one free-memory
observation. CPU preparation reserves 1 GiB temporary work and 48 MiB per image;
after preparation the transient part releases and the pixel part follows its
shared request owner. Unsupported providers retain their prior policy.

DeviceInput reserves before allocating or replacing image buffers. Its claim
includes old physical buffers that can coexist with the replacement, every
history embedding and the maximum sequential encoder working set. Same-image
reuse keeps the existing claim. Position restoration from RAM or disk enters
the same admission path, with the matching image attachment installed first.
Checkpoints retain image identity/layout, not another copy of GPU embeddings;
missing embeddings are recreated under the attached request's claim.

Claims retain their highest admitted value while image storage remains. An
allocation/copy failure may leave partial buffers, so failure does not release
that claim early. A successful text transition rewrites the stable descriptor,
synchronizes users, frees embeddings/positions and then releases the claim.
Destruction follows the same ownership order. Cancelled requests and restored
shorter layouts cannot make retained buffers disappear from accounting.

Admission exhaustion is a normal HTTP 503 `resource_exhausted`, including
requests that asked for streaming. Preparation and immediate slot admission
fail before SSE starts. If the scheduler has already opened a queued stream,
a later slot refusal uses its existing terminal stream-error path instead.
It does not mark the GPU unhealthy. Logs
report policy, base/shared claims, outstanding reservations, capacities, cache
grants and refusal reasons without request content.

## Limits of the policy

The budget is a logical reservation, not a physical preallocation. External
processes can still consume memory after startup; allocator failures remain
possible and follow the normal cleanup path. Free-memory queries never reset
the outstanding vision ledger. Requests are refused immediately when the pool
cannot admit them; they are not held indefinitely waiting for an idle session
to release retained images.

The opt-in uses the additive `ComponentCacheHooks::require_full_backing` hook;
its default false preserves other providers' lazy backing behavior. The API
change record requires an `ext-api` PR label. Checkpoint bytes and compatibility
identities are unchanged.

The preparation quota and retained high-water scratch make this deliberately
conservative. A small later image may be refused while other slots retain a
larger image history. Full-context text sessions do not reserve context-sized
embeddings, but four worst-case simultaneous image requests need not fit.
The hardware procedure must confirm actual capacity, cache commitment,
per-request performance and numerical image/position correctness.
