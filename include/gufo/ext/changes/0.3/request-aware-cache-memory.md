# Request-aware vision and cache memory contracts

Carry the branch's additive memory contracts into the active 0.3 series while
retaining its historical 0.2 records. Host admission exposes a consistent
snapshot; prepared text requests retain image reservations before preparation
and execution. Providers can coordinate cache-backing growth and optional
snapshot allocations with outstanding execution obligations.

The session-aware component-resource factory defaults to the existing provider
factory. Supporting providers can create bounded shared native transfer streams
with independent completion ownership. `DiskStore::OwnedManifestBytes()` reports
initialized retained vectors for remaining-allocation startup accounting.

These contracts retain the required text scheduling profile introduced in 0.3.
Rebuild extensions; no binary ABI compatibility is promised.
