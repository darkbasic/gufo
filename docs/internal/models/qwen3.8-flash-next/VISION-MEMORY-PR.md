# PR draft: on-demand vision memory admission

Title: `fix(qwen): admit vision work on arrival without changing text cache policy`

Addresses #920. Required label: `ext-api` for the additive resource ownership
and cache backing admission hooks; retain their permanent change records.

Flash-Next's default admission charges every execution session for a
context-sized image payload. `--vision-memory-policy on-demand` keeps the
projector available and uses the text/MTP estimate for ordinary traffic, then
reserves image preparation and execution capacity on arrival. Default
conservative admission, legacy `--vision-memory-bytes`, option validation and
unsupported providers retain their behavior.

Correct the earlier on-demand implementation's stronger startup contract.
Text eligibility uses the upstream platform-aware device/state gate and
required deferred execution work, without requiring the full cache maximum
and additional runtime margin beneath a host/device minimum. Restore upstream
cache sizing, the post-slot host recheck, reduced-grant reporting, lazy backing
and pressure checks. A configured cache maximum is not an unconditional startup
promise; the owner's four-session/full-4-GiB deployment check remains separate.
Restore upstream lazy MTP rollback for on-demand sessions, with its admitted
unmaterialized bytes protected until allocation. The target memory plan defers
615,813,120 bytes of initial commitment across four sessions; this is not a
reduction in peak memory or MTP capability. Legacy fixed-pool behavior stays
unchanged.

Runtime image reservations still coordinate concurrent preparation, complete
image history, retained pixels/embeddings/positions, encoding and replacement
peaks, restored layouts, optional snapshots and cache backing growth. Fresh
observations constrain pending obligations; committed storage is already in
free-memory measurements and is not charged twice. Accepted execution work
that has not materialized stays protected. Charges follow ownership until
buffers and asynchronous users are released. Allocation failures and ordinary
capacity exhaustion preserve the existing resource-error and recovery paths.

The latest owner-run candidate reached readiness with four full-context
sessions, seven-token MTP and vision. The launcher correctly rejected its
3,810,430,976-byte RAM grant; physical backing separately stopped at
259,522,560 bytes. Automatic 64-row copied-token verification had passed the
text gate while adding optional work to the runtime budget.

Make that width choice pressure-aware: wider rows require device eligibility
and host/device headroom for the configured cache preference plus existing
runtime margin. Failure to fund wider acceleration selects ordinary eight-row
verification, never a new combined startup refusal. Neural MTP's draft limit
and every deployment setting remain unchanged. The chosen width bounds actual
allocation and pending accounting together.

At the recorded inputs, 64→8 removes 1,903,942,656 deferred bytes and 9,175,040
initially committed hidden bytes across four sessions. Narrowing alone cannot
close the independent 484,536,320-byte cache-grant gap. Add phase observations
around cache-resource construction and slot allocation, with actual backing
plan/commitment fields, to localize the remaining unattributed interval. Keep
the upstream 64 transfer stream/event pairs; no byte cost or queue equivalence
is assumed. Independent allocator/transfer optimizations stay outside this PR.
See the [accounting comparison](VISION-MEMORY.md) and
[recorded evidence](VISION-MEMORY-ADMISSION.md).

CPU regressions use production admission, resource hooks and ownership with
synthetic observations and small allocations. They execute the production width-selection path for constrained and affordable
wide plans, distinguish startup stages,
retain concurrency/history/replacement/failure coverage and preserve strict
mixed-workload qualification. The harness records text eligibility, completed
startup/readiness and cache grant/backing separately, including failed
attempts; an engine cache trim or backing-growth limit fails the owner's
full-budget qualification while retaining functional results. See
[validation results](VISION-MEMORY-TESTS.md) for actual
commands and outcomes.

The real HIP release build, GPU numerical correctness and physical workload
fit remain pending. The [owner procedure](VISION-MEMORY-VALIDATION.md) makes one
candidate attempt after production chat is unloaded, retaining auxiliary
services, unchanged settings, a separate port and disposable disk-backed cache.
It stops on startup refusal and preserves that log. The independent GGUF
identity and disk-publication fixes remain intact; no launcher changes,
weights, inference-host access, commits or pushes are part of this correction.
