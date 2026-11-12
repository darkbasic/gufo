# Bounded cache recovery observations

The platform adds `MemoryObservation` and `QueryMemoryObservation()` for bounded,
best-effort process/host memory diagnostics. Missing or unsupported counters stay
optional; collected categories overlap and are not admission inputs.

`DiskStore` adds a constructor overload taking a synchronous, constructor-only
`DiskRecoveryObserver`. Four fixed callbacks distinguish directory scanning,
manifest decoding and temporary validation ownership. Observer exceptions do not
change recovery. `DiskStartupStats::directory_entry_stats` counts successful
metadata observations from startup enumeration, separately from dependency checks.
Existing constructors, cache formats, publication and memory grants are unchanged.
