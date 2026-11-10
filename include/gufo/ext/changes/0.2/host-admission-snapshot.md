# Host admission snapshot

Add `gufo::ext::HostSnapshotMemory`, `PlanHostSnapshotMemory` and
`QueryHostSnapshotMemory` through the existing host-budget forwarding header.
The snapshot exposes raw host availability, free CMA and its actual exclusion,
the minimum finite cgroup headroom and selected host limiter, the unchanged
4 GiB OS reserve, and the resulting explicit and automatic cache limits.

The pure planner and live query share the existing calculation; the scalar
host-budget functions delegate to the live snapshot. Query fields belong to
one sampling pass, not an atomic kernel observation. Committed allocations
are already reflected in free memory; callers remain responsible for their
uncommitted reservations. Existing function signatures and reserve policy
remain unchanged. This is additive source compatibility after rebuilding.

Add `gufo::hip::QueryAdmissionMemoryUsingHost` through the experimental
admission-query forwarding header. It reuses a caller's raw host observation
for the integrated-device budget calculation; the existing no-argument query
keeps its signature and behavior.
