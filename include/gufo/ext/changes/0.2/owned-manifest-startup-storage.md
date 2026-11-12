# Owned manifest storage at cache startup

Add `DiskStore::OwnedManifestBytes()` for quiescent startup accounting. It
reports initialized retained manifest vector storage already included in the
cache RAM allowance, excluding reservations, estimated overhead, spare
capacity and temporary validation storage. It is not an RSS measurement.

The shared serving cache compares remaining allocation obligations with the
post-slot host headroom, preserving the total cache allowance and staging
charge. Extensions must rebuild; binary ABI compatibility is not promised.

Test builds also expose a current-thread reference-validation scope hook for
allocation-failure injection; this hook is absent from shipped builds.
