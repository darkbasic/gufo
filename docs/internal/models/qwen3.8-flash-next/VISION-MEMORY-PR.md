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

The recorded same-phase observation is preserved unchanged. Its previous
combined policy refused by **730,301,776 bytes**; the corrected Linux text gate
passes with **9,971,190,448 bytes** of headroom. That eligibility change removes
an acceptance condition; lazy rollback changes allocation timing. Full-claim
host arithmetic still leaves about 3.820 GiB before the existing 4 GiB host
reserve, and hardware may trim
the cache or refuse backing/images. No four-session/full-cache fit is claimed.
See the [accounting comparison](VISION-MEMORY.md) and
[recorded evidence](VISION-MEMORY-ADMISSION.md).

CPU regressions use production admission, resource hooks and ownership with
synthetic observations and small allocations. They distinguish startup stages,
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
