# Distinct disk recovery observations

`DiskStartupStats` adds `dependency_references` and `distinct_dependency_files`.
`dependency_stats` continues to count actual dependency stat calls, now shared
by references to the same namespace and file identity during one cold recovery.
Directory enumeration remains a separate source of metadata checks. These
counters are not resident-memory measurements; no checkpoint format changes.

The platform `StatAt` no-throw boundary reports pathname allocation exhaustion
as `not_enough_memory`; disk recovery propagates it as allocation failure rather
than rejecting otherwise valid checkpoint owners. Rebuild extensions against
the additive source API revision.
