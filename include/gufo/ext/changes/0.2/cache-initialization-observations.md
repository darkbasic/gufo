# Component-cache initialization observations

Add the trailing optional `observe_initialization` callback to
`gufo::cache::ComponentCacheResources`. Serving invokes it at `pre_slots`,
`post_slot` and `post_slots`, with the number of successfully created slots,
before the post-slot cache-budget recheck and backing commitment. The observer
must not throw or affect admission decisions or execution storage; an empty
callback preserves
existing providers without extra observations. Rebuilding retains existing
aggregate initializers and cache sizing behavior.
