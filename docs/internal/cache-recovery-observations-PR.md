# perf(cache): reuse dependency observations during disk recovery

Populated-cache recovery stats shared payloads once per reference after already
enumerating the payload directories. Reuse one observation per namespace/file ID
in the existing temporary validation map, then validate geometry and every
conflicting owner after excluding incomplete manifests. Keep restore checksums,
publication, cache grants and lazy backing unchanged.

Add total-reference/distinct-file diagnostics, recovery-phase instrumentation,
synthetic create/recover benchmark modes and regression coverage for shared
dependencies, malformed claims, reopen, restore and allocation failure. Handle
pathname allocation exhaustion at the platform `StatAt` boundary without
terminating or rejecting valid owners. Label `ext-api` for the additive stats.

Against identical warm synthetic files with 1,137,400 references, dependency
stats fall to 122,650 and ordinary C++ allocation count falls by 4,059,000.
Instrumented recovery takes about 0.83 s versus 2.03–2.21 s. Retained heap is
unchanged, peak requested heap increases about 0.94 MiB, and RSS shows no meaningful
reduction. This does not establish a fix for the owner's cache-grant deficit.

Validation: nine native CPU regressions pass; production disk code also compiles
without test hooks. API/infrastructure checks pass. Changed C++ files pass
formatting; two untouched baseline format errors remain. HIP execution, Windows
compilation, ZFS attribution and full-workload capacity remain unverified.

This is independently reviewable from #920 vision admission and the GGUF identity
fix. See [evidence and the model-free owner procedure](cache-recovery-observations.md).
