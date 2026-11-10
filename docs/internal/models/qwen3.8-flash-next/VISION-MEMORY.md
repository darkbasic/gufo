# Vision memory accounting

`--vision-memory-policy on-demand` admits Flash-Next text sessions without an
unused image-execution allowance. Actual image work reserves its additional
memory before allocating. Conservative admission remains the default, and
positive `--vision-memory-bytes N` retains its existing fixed-pool guarantee.
This is the implementation account for #920; see the
[model page](../../../models/qwen3.8-flash-next.md) for usage and the
[owner procedure](VISION-MEMORY-VALIDATION.md) for pending GPU qualification.

## Startup and the recorded constraint

`Model::SessionBytes()` adds
`context * (hidden_size * sizeof(float) + 3 * sizeof(int32_t)) + 64` with a
projector. At 262,144 tokens and width 2560, that is 2,687,500,352 bytes per
session, although text execution needs only the stable descriptor. Conservative
admission retains this estimate and its existing consumers. Both opt-in modes
use `TextSessionBytes()` for serving and MTP copy-width selection, preserving
the text/MTP allocations and the 64-byte descriptor.

On-demand startup protects the complete text/MTP claims, deferred scratch,
configured RAM-cache allowance and existing 512 MiB runtime margin. It adds no
image-pool deduction. Loaded target, MTP and projector weights and shared
executor allocations are already consumed in the collected capacity and are
never added back. The remaining capacity is the minimum of the device admission
budget and the host ceiling, including the existing host/CMA/cgroup calculation
and separate 4 GiB OS reserve. One post-fingerprinting collection feeds MTP
copy-width selection, admission and the structured diagnostic.

The [November 10, 2026 fixed-pool observation](VISION-MEMORY-ADMISSION.md)
selected 34,672,033,792 bytes. Removing its unused 4,294,967,296-byte vision
deduction changes the remaining requirement from 39,653,254,480 to
35,358,287,184 bytes. **The recorded on-demand configuration still refuses by
686,253,392 bytes.** The fixed-pool regression still refuses by 4,981,220,688
bytes. No redundant model allocation was identified, and this change does not
claim a physical saving or four-session fit. Lazy cache backing changes when
memory is allocated, not its protected allowance or peak requirement.

## Ownership and admission

| Storage | Owner and lifetime | On-demand accounting |
| --- | --- | --- |
| Target, shared Q8 MTP and BF16 projector weights; loaded shared executor buffers | Loaded model | Already reflected in the startup capacity |
| KV/index rows, recurrent/PLE state, MTP hidden/draft/rollback rows, stable vision descriptor | Execution session | Complete protected text claim; rollback is materialized before readiness |
| Deferred wide rollback, batch logits, checkpoint arena and bounded mutable-state spares | Shared executor | Existing deferred allowance protected once |
| RAM checkpoint backing and disk transfer staging | Component cache | Full configured cache grant protected; upstream lazy backing and bounded staging retained |
| Prepared RGB image history | Prepared request and attached session | Preparation bound acquired before decode/resize, reduced to actual retained RGB capacity afterward |
| Image embeddings and GPU/host positional arrays | DeviceInput and its attached history | Full relevant history, retained capacity and replacement peak reserved before allocation |
| Encoder arrays and pinned upload | One serialized encoding operation | Largest sequential image working set, released only after work and transient storage end |
| Optional device snapshot copies beyond protected spares | Snapshot allocator | Compete atomically with image work for the remaining capacity; retained spare blocks keep tickets |

The cache keeps its complete 4 GiB retention grant. On-demand mode does not set
`ComponentCacheHooks::require_full_backing`: upstream warmup requests up to
2 GiB of backing, then grows within the grant. Uncommitted cache capacity is
never lent to images. Fixed-pool mode retains its existing full-backing
requirement; unsupported providers keep their prior behavior.

## One aggregate capacity bound

`AdmissionBudget` uses the existing concurrency-safe `cache::ResourceLedger`.
On-demand mode records the model's initial remaining capacity and protects all
text/cache/scratch/margin obligations in that ledger for the model lifetime.
Images and optional device copies acquire tickets from the remainder. The
bounded mutable-state spare subledger assigns storage within the already
protected scratch allowance; it does not deduct that allowance twice.
An epoch funds one execution/cache pool and at most its configured session
count. Reusing the same on-demand runner for another pool is refused; failed
initialization requires a new runner and capacity observation.

The initial bound never grows after a fresh free-memory reading. Committing
protected session/cache buffers consumes their allowance rather than creating
new image capacity. Runtime host/device observations can refuse allocation
under external pressure, but cannot replenish the ledger. On an APU the host
and device limits constrain the same capacity; they are not independent pools.
The configured cache allowance, concurrent image tickets and snapshot tickets
therefore cannot independently promise the same bytes.

The existing 128 MiB CPU preprocessing cache can retain shared pixels after
their prepared owner releases its ticket. Its bound remains inside the
unchanged 512 MiB runtime margin, alongside bounded staging/metadata and opaque
driver/library overhead; that margin is not lent to images.

Preparation first reserves the bounded 1 GiB decode/resize workspace and up to
48 MiB RGB capacity per image. Completion releases transient work and shrinks
the on-demand pixel ticket to the actual retained vector capacity. DeviceInput
includes all historical embeddings, both positional arrays, encoder workspace
and buffers that coexist during replacement. Stable descriptors belong to the
text base. Reused layouts and RAM/disk restoration enter the same path.
Checkpoints retain image identity/layout; missing embeddings are recreated
under the current attached history's reservation.
Prepared image pointers retain their pixel ticket directly, so copying or
assigning a prompt, or retaining an individual image, cannot release it early.

On-demand leases shrink only after encoding/upload work and transient storage
finish, retaining the charge for every live buffer. Failed allocation or copy
can leave partial storage, so failure keeps its admitted bound. Text transition,
reset and destruction synchronize users and free storage before releasing its
ticket. Cancellation and a shorter restored layout do not imply that retained
capacity has disappeared. Fixed-pool leases retain their earlier high-water
behavior for compatibility.

## Refusals, compatibility and limits

Insufficient capacity produces the existing `resource_exhausted` response;
preparation and immediate slot refusals return HTTP 503 before streaming.
Already-open queued streams use the scheduler's terminal error path. Ordinary
capacity refusal does not mark the GPU unhealthy or prevent later text work.
No indefinite wait is introduced. Diagnostics report policy, limits, protected
and outstanding claims, cache grants and refusal reason without request content.

The initial bound is a logical guarantee between Gufo owners, not physical
preallocation or protection against another process consuming memory. Existing
allocation-failure cleanup remains necessary. Memory released by unrelated
processes after startup does not enlarge image capacity; a later request may
therefore refuse despite a larger current free reading. Even small images can
refuse when the remaining bound is too small for preparation or retained peers.

`ResourceReservation::Shrink(size_t)` is additive and requires the `ext-api`
label and its permanent change record. Numerical operators, image positions,
checkpoint payloads and compatibility identities are unchanged. This admission
behavior does not establish the owner's four-session physical-memory fit,
numerical image correctness or performance; those require the actual GPU.
