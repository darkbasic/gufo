# Session-sized cache transfer resources

Add a source-compatible `TextModelRunner::CreateComponentCacheResourcesForSessions`
factory receiving the actual cache session count. Its default delegates to the
existing two-argument provider. Serving calls it before constructing slots or
granting cache backing, so native resources remain visible in memory snapshots.

`ComponentCacheHooks::transfer_leases` optionally separates eager logical
event/lease capacity from native stream count. Zero retains exclusive pairs.
`PlanComponentTransfers` sizes shared resources for the bounded component-cache
consumers; `TransferPool` adds an eager shared-stream constructor with independent
completion fences, retained ownership and lane quarantine. There is no native
resource creation during acquisition. Rebuild extensions; no binary ABI is promised.
