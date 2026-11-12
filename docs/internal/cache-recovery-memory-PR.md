# chore(cache): observe memory across populated recovery

The owner sees about 986 MiB less host availability between cache-resource
construction and slot creation. Existing logical ledger and dependency counters
cannot distinguish process residency from global filesystem/kernel changes.

Add eight fixed startup observations through the platform abstraction, including
RSS/high-water RSS, host availability/cache/slab, optional ARC size and collection
timing. Separate directory-enumeration stats from dependency stats and observe
temporary validation ownership before and after destruction. Missing values stay
unavailable; diagnostics never affect admission or recovery acceptance.

No justified resident-memory saving was found. Cache grants, reserves, lazy
backing, publication/restore checks, on-demand vision and deployment settings
remain unchanged. The 494,959,220-byte grant deficit is unresolved.

Validation: eight native CPU tests pass, including deterministic parser/reader
fixtures and production recovery/serving observation boundaries. API label:
`ext-api` (additive revision 0.3.6). GPU execution, Windows compilation and actual
workstation memory attribution remain pending. See the
[audit, commands and single owner attempt](cache-recovery-memory.md).
