# Shared admission for physical cache growth

Add the trailing optional `reserve_segment` callback to
`gufo::hip::ComponentCacheHooks` and `gufo::hip::BackingPoolConfig`. It reserves
only a segment about to allocate. Successful registration converts its ticket
to a charge retained until the physical pages are released; failed allocation
or pressure checks release pages before the reservation. Capacity exhaustion
declines growth through the ordinary cache pressure path.

The empty callback preserves existing cache grants, lazy backing and pressure
checks. It does not reserve an uncommitted configured cache maximum. Existing
aggregate initializers retain their behavior after rebuilding.
