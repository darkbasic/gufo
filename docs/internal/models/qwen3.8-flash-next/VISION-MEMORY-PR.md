# PR draft: safe request-aware vision admission

Title: `fix(qwen): admit vision memory from a shared request budget`

Fixes #920.

Required PR label: `ext-api` (additive component-cache backing hook).

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

With the issue's estimates, four base claims including the descriptor total
30,108,435,792 bytes. Adding a shared 4 GiB vision pool, the unchanged 4 GiB RAM
cache and 512 MiB margin totals 39,235,241,296 bytes, below the reported capacity.
This establishes admission eligibility only; actual GPU capacity is unverified.

Validation uses production host accounting and DeviceInput lifecycle code with
simulated HIP allocations, plus shared scheduler/cache and provider-option
regressions. Coverage includes default/opt-in behavior, exact boundaries,
overflow, simultaneous reservations, history, retained growth, reset,
cancellation, allocation failures and RAM/disk restoration. The standalone
accounting tests pass under GCC and ASan/UBSan. The complete CPU build succeeds;
the native PR gate passes all 114 tests in 57.79 seconds. Six CPU tests of the
hardware harness also pass. See the [VM test report](VISION-MEMORY-TESTS.md)
for commands, limits and paths not compiled with HIP.

The [owner-run procedure](VISION-MEMORY-VALIDATION.md) retains separate-port
four-session execution evidence, full cache grants, image success/refusal,
RAM/disk reuse, per-request timings and memory/swap samples. Neither hardware
capacity nor numerical image correctness is claimed from the development VM;
both require the actual GPU qualification. No weights were downloaded and no
inference host, production cache or deployment launcher was accessed.
