# Shrink pending resource reservations

Add `gufo::cache::ResourceReservation::Shrink(std::size_t bytes)` to the
forwarded resource-ledger contract. It reduces a standalone pending reservation
without allocating or converting it into a committed charge. The remaining
reservation keeps its identity and category; zero empties the handle.

Shrinking an empty handle throws `std::logic_error`. Growing a reservation or
shrinking a pool assignment, backing allocation or non-pending allocation
throws `std::invalid_argument`. Equal size is permitted. The operation updates
the ledger under its existing lock and does not invoke allocation-failure
injection. Callers must finish asynchronous work and free any storage or
obligation covered by the released portion before shrinking.

This is an additive source API change. Existing reservation, conversion,
backing-pool and persistence-pin behavior remains unchanged.
