# PR draft: safe request-aware vision admission

Title: `fix(qwen): admit vision memory from a shared request budget`

Addresses #920; the recorded four-session deployment remains over budget.

Required PR label: `ext-api` (additive component-cache backing and memory-snapshot interfaces).

Flash-Next reserves a context-sized image embedding/position allowance for every
execution session whenever its vision projector is loaded. At 262,144 tokens
and width 2560, this adds 2,687,500,352 bytes per session, even though text
requests allocate only the stable descriptor. Four reported conservative
claims total 40,858,436,944 bytes against 40,088,076,288 available, so startup
rejects four sessions before seeing any image request.

Add provider-owned `--vision-memory-bytes N`. Zero, including the default,
preserves conservative admission. A positive value reserves one bounded pool
beside complete text/MTP state, deferred shared work, the full RAM-cache grant
and the existing runtime margin. The opt-in commits its full checkpoint backing
before readiness and refuses startup if it cannot honor that grant.
`SessionBytes()` keeps its original worst-case
meaning for snapshots and non-serving callers; the opt-in serving path uses a
separate base estimate including the stable descriptor.

Image preparation first reserves a bounded CPU decode/resize allowance; the
prepared request retains its pixel reservation. Image requests reserve checked
memory bounds for image history,
embeddings, position arrays, encoder working buffers, pinned upload and
replacement peaks. The shared resource ledger atomically accounts outstanding
reservations; refreshing free-memory observations cannot spend the same pool
twice. Charges follow retained buffers across reuse, cancellation, failure and
checkpoint restoration, and release only after the corresponding storage is
freed. Preparation and immediate admission refusals return HTTP 503
`resource_exhausted` before streaming; already-open queued streams use the
scheduler's terminal error path. Subsequent text requests remain usable.
Vision stays loaded; numerical
operators and persistent checkpoint representations are unchanged. The additive
`ComponentCacheHooks::require_full_backing` contract defaults to false for
existing providers and has its own extension API change record.

The later ZBook event selects 34,672,033,792 bytes against a complete remaining
requirement of 39,653,254,480 bytes. The new deterministic regression requires
the unchanged 4,981,220,688-byte refusal. The audit found no justified savings;
even making the whole vision pool lazy would leave 686,253,392 bytes missing.
See the [admission audit](VISION-MEMORY-ADMISSION.md) for the exact components
and ownership findings. No four-session fit is claimed.

Collect the host/device inputs once after fingerprinting and share them between
copy-width selection and final admission. One event exposes both limits, their
CMA/cgroup/reserve calculation, loaded allocation ownership, uncommitted claims
and the deficit. Production host and per-session calculations are shared with
CPU tests, preserving the previous totals and safety reserves.

Validation uses production host accounting and DeviceInput lifecycle code with
simulated HIP allocations, plus shared scheduler/cache and provider-option
regressions. Coverage includes default/opt-in behavior, exact boundaries,
overflow, simultaneous reservations, history, retained growth, reset,
cancellation, allocation failures and RAM/disk restoration. The standalone
accounting tests pass under GCC and ASan/UBSan. The complete CPU build succeeds;
the initial native PR gate passed all 114 tests in 57.79 seconds. Those results
predate this follow-up. The final admission follow-up passed nine focused CPU
regressions and 21 harness tests; see its [commands and limits](VISION-MEMORY-ADMISSION.md#validation-scope).
The [initial VM test report](VISION-MEMORY-TESTS.md) retains the earlier evidence.

The [owner-run procedure](VISION-MEMORY-VALIDATION.md) retains separate-port
four-session execution evidence, full cache grants, image success/refusal,
RAM/disk reuse, per-request timings and memory/swap samples. Neither hardware
capacity nor numerical image correctness is claimed from the development VM;
both require the actual GPU qualification. No weights were downloaded and no
inference host, production cache or deployment launcher was accessed.
