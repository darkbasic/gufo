# Required cache backing at readiness

Add the trailing, default-false `require_full_backing` field to
`gufo::hip::ComponentCacheHooks`, available through the component-cache resource
forwarding header. When true, `commit_backing` commits every payload page in the
adapter's backing plan and refuses initialization if the full payload cannot be
backed. Metadata remains separate from committed payload bytes. The default
keeps bounded startup preparation and lazy growth.

This is additive source compatibility after rebuilding; existing aggregate
initializers keep their behavior.
