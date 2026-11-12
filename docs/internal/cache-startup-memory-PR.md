# Populated cache startup: PR description

Suggested title: `fix(cache): account for owned manifests during startup`

Required label: `ext-api`.

Recovered checkpoint manifests allocate storage before the post-slot host
observation, and that storage is already included in the total cache RAM budget.
The previous grant cap nevertheless charged the whole budget against the
remaining host ceiling. Compare the budget's remaining allocation obligation
with remaining headroom by reconstructing only initialized, retained manifest
vectors. Keep staging, pinned-slot treatment, logical ledger limits, lazy backing
and all existing reserves enforced. Ledger reservations and converted overhead
estimates receive no physical-memory credit.

Reduce cold recovery's temporary allocation by retaining one canonical borrowed
reference per unique payload, comparing fixed-size geometry, and rescanning every
owner after conflicts are known. Validate all owners before removing rejected entries;
reject every conflicting or duplicate owner regardless of enumeration order.
Payload checksums remain a streaming-restore requirement, and cold startup
continues reading manifests and statting dependencies rather than reading payloads.

This is a shared cache-startup correction, independently reviewable from #920,
native transfer-resource optimization and GGUF identity. It changes neither
configured budgets nor the launcher's full-cache requirement and does not discard
valid checkpoints to improve admission. The deployment must retain its existing
independent disk-publication fix when composing the candidate; the owner's
composed r8 source was not built or tested in this VM.

The owner observed a 302,198,784-byte cache-grant shortfall with populated r8
persistence. The observed 828,944,384-byte host decline is not claimed as C++
metadata or a leak. Actual owned bytes and temporary allocation savings must be
distinguished from filesystem effects and from the remaining workload capacity.

See [the invariant, evidence and populated restart procedure](cache-startup-memory.md).
CPU regressions use production disk/serving paths and synthetic checkpoints;
real HIP execution and populated-cache workload qualification remain pending.

Matched synthetic recovery profiles reduced validation's requested C++ heap peak
from 1,850,704 to 792,848 bytes without sharing, and from 191,824 to 55,568 bytes
with shared chunks. Logical live charges stayed unchanged. These scoped byte
measurements are not a prediction of the owner's resident-memory saving; timings
are single instrumented observations, with all runs retained.

The final CPU PR gate passed **116/116 tests in 50.26 seconds**. The initial gate
passed 115/116 tests; a new overly broad heap-failure test
aborted before validation. After restricting actual heap-failure injection to
the production validation scope while retaining whole-constructor ledger faults,
the affected disk and serving tests passed 2/2 in 1.45 seconds, followed by the
successful full gate. The linked notes record exact commands, the initial failure
and pending hardware qualification.
