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

The latest owner-run r6 candidate confirms on-demand admission, ordinary
eight-row copied-token verification, four full-context sessions, seven-token
MTP and vision. The engine reached readiness; the launcher correctly rejected
its 3,851,272,192-byte RAM grant. Backing crossed the ordinary 2 GiB preparation
target at 2,229,338,112 bytes, while the later runtime observation had only
9,113,408 bytes of headroom. These are distinct results, not complete workload
qualification.

The optional-width correction remains: wider rows require device eligibility
and host/device headroom for the cache preference plus existing runtime margin.
Failure selects ordinary verification, never a combined startup refusal or a
change to neural MTP's draft limit. The selected bound controls allocation and
pending accounting together.

The new phase observations localize 512 MiB of HIP consumption to cache-resource
construction before slots/backing. This phase creates the upstream 64 native
stream/event pairs, but does not establish their individual cost or a safe pool
reduction. A separately reviewable [model-free native probe](../../transfer-resource-probe.md)
now investigates that overhead. No allocator/transfer optimization or claimed
memory saving belongs to this #920 correction without that evidence and a
lease/progress proof. See the [accounting comparison](VISION-MEMORY.md) and
[recorded evidence](VISION-MEMORY-ADMISSION.md).

CPU regressions use production admission, resource hooks and ownership with
synthetic observations and small allocations. They execute the production
width-selection path for constrained and affordable wide plans, distinguish
startup stages,
retain concurrency/history/replacement/failure coverage and preserve strict
mixed-workload qualification. The harness records text eligibility, completed
startup/readiness and cache grant/backing separately, including failed
attempts; an engine cache trim or backing-growth limit fails the owner's
full-budget qualification while retaining functional results. See
[validation results](VISION-MEMORY-TESTS.md) for actual
commands and outcomes.

The new probe's real HIP compilation and execution, GPU numerical correctness
and complete physical workload fit remain pending. No production HIP source is
changed by this measurement follow-up; CPU tests compile the existing transfer
pool against simulated HIP only. The [owner procedure](VISION-MEMORY-VALIDATION.md) runs the
model-free probe first with production chat unloaded. Another full-model attempt
requires a justified changed candidate or a specific unresolved measurement;
it retains auxiliary services, unchanged settings, a separate port and disposable
disk-backed cache, and stops on admission/cache refusal while preserving the log.
The independent GGUF identity fix remains untouched. Keep the deployment's
separate disk-publication fix in its composed source stack; it is absent from
this checkout and this investigation does not reapply it. No launcher changes,
weights, inference-host access, commits or pushes are part of this correction.
