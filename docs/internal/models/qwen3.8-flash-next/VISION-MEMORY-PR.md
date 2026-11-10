# PR draft: on-demand vision memory admission

Title: `fix(qwen): reserve vision execution memory on demand`

Addresses #920. Required label: `ext-api` for the additive
`ResourceReservation::Shrink(size_t)` contract.

Flash-Next's conservative admission charges each execution session for a
context-sized image payload. Its fixed-pool opt-in instead deducts the complete
configured image pool before any request. Add `--vision-memory-policy on-demand`
to admit text sessions using their actual text/MTP state and stable vision
descriptors while preserving projector availability. The default remains
conservative; positive `--vision-memory-bytes N` retains the fixed-pool behavior,
zero remains conservative, and ambiguous combinations are rejected.

On-demand startup protects all text, deferred scratch, cache and safety
obligations against one host/device minimum collected after model loading and
fingerprinting. Images and optional device snapshot copies atomically reserve
only their additional demand from the remaining capacity. Loaded allocations
are already consumed, and fresh free-memory observations never replenish this
bound. Prepared history, embeddings, positional storage, encoder work and
replacement peaks remain charged through their actual lifetimes. Successful
completion releases transient reservations; retained or partially allocated
storage remains covered across reuse, cancellation, failure and restoration.

Preserve the full configured RAM-cache grant with upstream lazy backing. Its
uncommitted capacity stays protected from image work; the new policy does not
require eager allocation of all 4 GiB. Fixed-pool backing guarantees, unsupported
providers, conservative estimate consumers, checkpoint compatibility and
numerical execution remain unchanged. Capacity refusals use existing resource
errors and stream-error behavior, leaving subsequent text requests usable.

The change removes the unused fixed image deduction, not physical allocations.
At the recorded ZBook capacity of 34,672,033,792 bytes, on-demand still requires
35,358,287,184 bytes and refuses by **686,253,392 bytes**. The original fixed-pool
case still refuses by 4,981,220,688 bytes. The
[audit](VISION-MEMORY-ADMISSION.md) preserves those measurements and the
unchanged session-component bounds. Context, sessions, MTP, cache budgets,
auxiliary services and safety reserves are not reduced. No four-session fit is
claimed.

Validation exercises production accounting, allocation lifecycle, optional
snapshot admission, CLI metadata and cache/scheduler contracts using synthetic
CPU fixtures. It retains aggregate allocation coverage and the mixed-workload
harness, adapting cache checks to lazy backing and checking occupied cache
storage before mixed execution. See the [current VM results](VISION-MEMORY-TESTS.md#on-demand-follow-up)
for completed commands; historical test counts are separately dated.

The real HIP release build, numerical image correctness and physical fit remain
pending. The [owner procedure](VISION-MEMORY-VALIDATION.md) uses one full-workload
attempt after production chat is unloaded, with required auxiliary services
retained, a separate port and a disposable disk-backed cache. It stops on startup
refusal and retains evidence instead of requiring another model load. No model
download, inference-host access, deployment-launcher change, commit or push is
part of this patch.
