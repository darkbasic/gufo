# Cache transfer resources: PR description

Suggested title: `perf(cache): share native transfer streams in Flash-Next`

Required label: `ext-api`.

Flash-Next's cache pool constructs 64 native HIP streams before slots and cache
backing. The owner's matching-runtime probe measured an 8 MiB count-dependent
allocation per stream, returned on destruction. Events and first use added no
observable HIP allocation in those cases. The r6 four-session build reached
readiness but its logical RAM-cache grant was 443,695,104 bytes below the
requested 4 GiB. Reuse a bounded set of native FIFO streams while keeping a
separate event and ownership record for every logical transfer lease.

The session-aware Flash-Next provider eagerly creates `N + 1` native streams and
`18 * N + 1` logical leases: one permanent preservation lease, a 16-transfer
foreground batch and one conservative in-flight idle allowance per slot, plus
one serialized disk-staging transfer. Four sessions use five native streams and
73 event leases. Existing batch width and supported concurrent typed callers
retain their logical capacity. Acquisition creates no native resource and does
not wait for another lease to release a stream.

Healthy shared-mode completion waits for its own system-release event, preserving
host visibility without waiting for unrelated later work. Leases and completions
retain their event and source ownership; error paths drain, and pool-observed
native failures quarantine affected lanes. Flash-Next's temporary key-copy path
uses the same lease fence before releasing its buffer. Existing unsized providers and
the old factory entry point retain exclusive-pool behavior through an additive
source API fallback.

The measured stream-count term predicts a conditional **472 MiB** reduction for
four sessions, only **51,232,768 bytes** more than the recorded logical grant
gap. Actual footprint of the changed event policy and throughput remain to be
measured. Vision admission, requested cache settings, lazy backing, safety
reserves, model/MTP settings and the launcher's grant check are unchanged.

This is separately reviewable from #920 and the independent GGUF identity fix.
See the [resource proof, owner evidence and validation procedure](cache-transfer-resources.md).

The new probe evidence was supplied by the owner, with embedding and WebUI
stopped. It supports a native resource optimization but does not establish the
full daily workload's capacity or throughput. GPU execution of the changed
candidate and its workload qualification remain pending under the linked owner
procedure, with production chat unloaded, auxiliaries running and disposable
disk-backed cache. A full grant alone does not qualify backing growth, image
admission, numerical correctness or unchanged throughput.

Production planning, provider dispatch, concurrent capture/restore and transfer
ownership have CPU regressions using small buffers and controlled completions.
They retain existing backing, cancellation, persistence and failure coverage.
The native CPU PR gate passed **116/116 tests in 52.20 seconds**; the focused
serving transfer-resource cases also passed. After the final cached-native-handle
cleanup correction, the two affected regressions passed again in **1.38 seconds**.
Exact commands and scope are in the linked validation notes. No pinned Nix gate
was available in this VM.
The API diff gate passed at 0.2.26; documentation, changed-file formatting and
diff checks passed. Whole-tree formatting retains two unrelated baseline
violations, documented in the validation notes.
Real HIP compilation of the changed pool/resource factory and Flash-Next
runner/continuation adapter remains pending; simulated HIP cannot establish
hardware compilation, numerical correctness or memory fit.
