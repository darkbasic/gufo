# Vision memory accounting

`--vision-memory-policy on-demand` admits Flash-Next text sessions without an
allowance for hypothetical images. Image work acquires its additional capacity
before allocation. Conservative admission remains the default; positive
`--vision-memory-bytes N` preserves the legacy fixed-pool policy. See the
[model page](../../../models/qwen3.8-flash-next.md) for usage and the
[owner procedure](VISION-MEMORY-VALIDATION.md) for pending GPU qualification.

## Startup and cache contracts

`Model::SessionBytes()` adds
`context * (hidden_size * sizeof(float) + 3 * sizeof(int32_t)) + 64` when a
projector is loaded. At 262,144 tokens and width 2560, that is 2,687,500,352 bytes
per session. Text execution needs the stable descriptor but no context-sized
image payload. Conservative admission and other estimate consumers remain
unchanged. The opt-in serving paths use `TextSessionBytes()`, preserving all
text/MTP state and the 64-byte descriptor.

| Stage | Upstream 0.33.0 (`77dff828`) | Previous on-demand (`0825cdd`) | Corrected on-demand |
| --- | --- | --- | --- |
| Text gate | Device admission minus deferred execution work; conservative per-session image term | Text, full configured cache, deferred work and runtime margin must fit host/device minimum | Upstream device gate with text/MTP estimate; no hypothetical image term |
| RAM-cache sizing | Host budget and post-slot recheck; reduced grant reported | Complete configured maximum required and protected at startup | Upstream sizing/recheck/trim reporting |
| Per-session MTP rollback | Lazy allocation within admitted state bound | Forced complete allocation before readiness | Upstream lazy allocation; pending execution allowance retained |
| Backing | Bounded warmup, lazy growth, host/device pressure checks | Lazy backing within a permanently protected maximum | Upstream lazy backing with shared arrival-time coordination |
| Image capacity | Conservative startup allowance | Fixed remainder of the combined startup epoch | Current capacity and outstanding obligations checked before image work |

The text gate uses `QueryAdmissionMemory()` and deducts deferred execution work
once. Its existing platform rules remain: Linux may use HIP free bytes, while
Windows also applies its device/host admission constraints. The diagnostic's
host ceiling is not an additional text acceptance condition. Actual text-state
insufficiency and later allocation failures still fail startup.

`ResourceClaim()` again reports the ordinary host snapshot budget and ceiling.
The serving cache rechecks them after slots allocate, preserves its metadata
and staging accounting, and reports any reduced grant. On-demand does not
require an exact full cache grant or full backing commitment. Optional future
cache growth is not reserved at text startup; each growth segment must pass
upstream pressure checks and acquire runtime capacity before committing pages.
Conservative and legacy fixed-pool hooks retain their previous behavior.

On-demand also restores upstream lazy MTP rollback allocation. The measured
target's production memory plan assigns 153,953,280 bytes per session to those
rows, or 615,813,120 bytes across four sessions. Deferring their initial
allocation changes startup commitment, not the peak claim or MTP capability.
The runtime coordinator keeps the unmaterialized rows pending; each successful
row allocation replaces that pending amount with committed ownership. Trimming
rows restores their pending obligation before freeing them. The fixed-pool
mode retains its earlier eager rollback and retained floor.

The [latest owner-reported attempt](VISION-MEMORY-ADMISSION.md#ready-engine-with-cache-limits-at-83c0c7a2)
reached readiness with four sessions, full context, MTP and vision. It still
failed the owner's target: the RAM grant was 484,536,320 bytes below 4 GiB and
backing stopped at 259,522,560 bytes. The earlier combined startup refusal is
preserved separately in that evidence page. Text eligibility, completed startup,
cache grant, planned payload/metadata, committed backing and image capacity are
separate results.

## Optional copied-token width

Copied-token verification can use 8–64 rows. It is separate from the neural
MTP predictor's seven-token draft limit. On-demand selection tries wider rows
only when the device text gate fits and current host/device headroom can also
accommodate the configured cache preference and existing runtime margin.
Otherwise it chooses ordinary eight-row verification. This preference never
refuses startup: the unchanged device gate subsequently checks the selected
text/deferred requirements, and upstream cache sizing still decides the grant.
It creates no image pool or unconditional cache backing promise.

The selector uses production per-width session and deferred-work formulas.
Its chosen bound also controls execution and retained hidden storage, so lower
pending charges cannot leave an unchecked wider allocation enabled. Concurrent
requests, cancellation and retained asynchronous users keep the existing
execution-commitment and reservation ownership. Affordable wider execution
remains available without another CLI option.

For the latest recorded inputs, the selector chooses eight rows. Compared with
64 rows this removes 1,903,942,656 bytes of accepted deferred work and only
9,175,040 initially committed hidden bytes across four sessions. The latter
cannot close the independent 484,536,320-byte logical-grant shortfall. These
are formula differences, not measured physical savings; the next GPU attempt
must establish grant, backing and remaining image capacity.

Cache-initialization observations bracket resource construction and slot
allocation. They report host/device capacity, explicit committed bytes,
unmaterialized execution and transfer-stream count at each phase. The upstream
64 stream/event pairs remain unchanged; the unattributed host-availability
residual is not assigned to streams, hardware queues or allocator overhead.
The backing event reports the real `BackingPlan()` payload and metadata beside
its grant and committed pages; do not compare payload bytes with an exact
4 GiB requirement.

## Arrival-time accounting

The shared `AdmissionBudget` coordinates image work, optional device snapshot
copies and cache backing growth through the existing resource ledger. Before
accepting new work it samples the platform-aware device admission budget and
host ceiling, taking their minimum and retaining the existing 512 MiB runtime
image margin. On the APU these observations constrain the same physical memory;
they are not two independently spendable pools.

Reservations are serialized. Pending obligations are captured before the
memory observation so concurrent conversion to committed ownership can only
make the check conservative. Already committed allocations are reflected in
free memory; they remain attached to their owners but are not subtracted again
or credited back as newly free capacity. Accepted text/deferred execution work
that has not materialized remains unavailable to images and optional growth.
Neither the configured cache maximum nor a combined text/cache/margin sum is
an on-demand startup acceptance condition.

| Storage | Owner and accounting |
| --- | --- |
| Target, shared Q8 MTP and BF16 projector weights; loaded shared executor buffers | Already consumed in the observed available memory |
| KV/index rows, recurrent/PLE state, MTP hidden/draft/rollback rows, stable vision descriptor | Text session; accepted unmaterialized execution obligations remain protected |
| Deferred batch logits, wide-copy rollback, checkpoint arena and bounded mutable-state spares | Shared executor; pending allowance diminishes only after actual commitment |
| RAM checkpoint backing | Component cache; each segment reserves before allocation/registration and retains its charge until freed |
| Prepared RGB image history | Prepared request/image pointers; preparation bound acquired before decode/resize, then reduced to retained RGB capacity |
| Image embeddings and GPU/host positional arrays | DeviceInput and attached history; retained capacity and replacement peak reserved before allocation |
| Encoder arrays and pinned upload | Encoding operation; largest sequential working set released only after asynchronous users and storage end |
| Optional device snapshot copies | Snapshot allocator; shared capacity, with retained spare blocks keeping ownership charges |

Upstream cache reserve checks still run; internal reservations do not exempt
backing growth from external pressure. A cache grant is its retention limit,
not a promise to allocate every byte. Backing refusal keeps its warning and
normal cache fallback. The existing 128 MiB CPU preprocessing cache stays
inside the unchanged runtime margin alongside bounded staging/metadata and
opaque driver/library overhead.

## Ownership and refusals

Preparation reserves the bounded 1 GiB decode/resize workspace and up to
48 MiB RGB capacity per image before preparing or counting tokens, including
entry points that run before execution-slot acquisition. Completion releases
transient work and shrinks the pixel ticket to actual retained vector capacity.
Pixel pointers retain the ticket through prompt copies, assignments and
individual image retention.

DeviceInput accounts for all relevant history, embeddings, both positional
arrays, encoder workspace and old/new buffers coexisting during replacement.
Stable descriptors belong to the text base. Reused layouts and RAM/disk
restoration use the same admission path. Checkpoints retain image identity and
layout; missing embeddings are recreated under the attached history's claim.

Leases shrink only after work and transient storage end. Partial allocation
failure retains coverage for surviving buffers and remaining work. Reset,
text transitions and destruction synchronize users and free storage before
releasing charges. Cancellation or a shorter restored layout cannot release
retained capacity. Fixed-pool leases keep their prior high-water behavior.

Insufficient image capacity uses the existing structured `resource_exhausted`
response and normal `usage.gufo` reporting where applicable. Preparation and
immediate slot refusals return HTTP 503 before streaming; an already-open
queued stream uses the scheduler's terminal error path. Ordinary exhaustion
does not mark the GPU unhealthy or fabricate success; later text requests can
continue. No indefinite wait or per-token memory polling is introduced.

Internal coordination cannot prevent another process consuming memory after
an observation. Allocation-failure cleanup remains necessary, and the bounds
are not exhaustive measurements of opaque driver/BLAS storage. Diagnostics
separate text eligibility, effective cache grant/backing, current runtime
capacity, pending/committed ownership and refusal reasons without content or
credentials. Numerical operators, image positions and checkpoint identities
are unchanged. CPU tests establish these accounting contracts, not GPU
compilation, four-session physical fit, numerical quality or performance.
