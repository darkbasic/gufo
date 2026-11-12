# Populated archive metadata: the remaining recovery constraint

The r12 evidence localizes the dominant host-memory decline to initial filesystem
discovery. The current archive and quota contracts still require actual metadata
for every distinct managed payload. This audit found no small safe change that
avoids that working set through startup, reclamation and later writes. Production
behavior and the r12 observations remain unchanged; new regressions document the
constraint rather than claim a memory saving.

## Owner evidence

The owner ran the branch containing `17fd70dbe46fde695bd0c85008aa35396e2a4f9e`.
During `before_disk_recovery` to `after_directories`, about 1.014 seconds:

| Observation, bytes | Before | After |
| --- | ---: | ---: |
| Host available | 38,907,834,368 | 38,071,574,528 |
| Process RSS | 1,404,215,296 | 1,404,252,160 |
| Host Slab | 952,897,536 | 1,885,995,008 |
| ZFS ARC size | 1,420,813,832 | 2,152,642,976 |

Host availability declined 797.520 MiB while process RSS grew only 36 KiB.
Slab and ARC are overlapping global observations: their increases cannot be
added together or assigned exclusively to Gufo allocations. This phase includes
layout opening/syncing and directory traversal, before any manifest decoding.
It reported 184,363 directory-entry stats, zero dependency stats and zero
manifest bytes read.

Final recovery accepted 100 manifests with 184,263 distinct dependencies and
1,137,375 references. Directory-entry stats reached 184,463; dependency stats
were 184,263. The extra 100 directory observations are the second manifest scan.
The initial count is consistent with 100 manifests plus those distinct payloads.
Removing repeated checks alone does not eliminate their first metadata access.

Across `post_cache_resources` to `pre_slots`, host availability fell 948.586 MiB,
RSS rose 81.895 MiB and HIP free stayed at 40,447,311,872 bytes. Four sessions
loaded successfully. The launcher rejected a 3,843,851,660-byte RAM-cache grant,
**451,115,636 bytes below** the 4,294,967,296-byte request. This is a cache-grant
constraint, not an OOM or text-session admission failure.

## Why the current archive still needs these visits

[DiskStore](../../src/cache/disk.cpp) stores ID-only filenames and raw payload
files. Manifests describe expected payload lengths and checksums; they are not
an inventory of actual namespace usage. No independent trusted aggregate records
unreferenced files, temporary partial writes or changed payload lengths.
Filesystem allocated-space totals also differ from this quota's logical file
lengths, particularly for sparse or compressed files.

The initial four `Scan()` calls enumerate manifests, chunks, private and temporary
files. [ListDirectory](../../src/core/platform/fs_posix.cpp) performs a no-follow
metadata call for every directory entry before the cache filters its name/type.
`managed_bytes` sums actual sizes of managed regular files, once per namespace
entry, including referenced, temporary and orphaned files. Sharing one payload
between many manifests does not charge that file repeatedly.

After decoding, `ValidateReferences()` independently checks each distinct
dependency's actual type, single-link status and size. It excludes incomplete
owners before checking shared geometry, private aliases and duplicate checkpoint
IDs. That ordering matters: an owner missing another dependency must not poison
an otherwise valid checkpoint merely because their manifest claims conflict.

Even a name-only initial traversal would therefore leave all 184,263 distinct
payload checks before startup completes. Deferring both passes would leave
global usage and availability unresolved. Manifest declarations cannot safely
stand in for actual sizes: an orphan has no manifest, a temporary file can be
partially written, and an existing payload can be larger than its declaration.

The exclusive `LOCK` prevents concurrent cooperating Gufo owners. It does not
attest changes made while Gufo was stopped, or make a saved directory timestamp
detect in-place child-file changes. A persistent summary needs a trustworthy
mutation/crash protocol and explicit external-change handling. Falling back to
the same full traversal must remain visible; a summary alone is insufficient.

## Later work prevents a startup-only shortcut

The [shared serving constructor](../../src/cli/serve/component_text_cache.cpp)
always starts `ReclaimOrphans()` after grant/backing setup. `Reclaim()` first
scans manifests, removes rejected manifests and syncs their directory before
deleting any unreferenced payload. It then scans payload/private/tmp namespaces,
syncs deletions, and scans all four namespaces again to reconcile actual usage.
Thus moving the first traversal past the grant would recreate its working set
after backing and session allocations have consumed more headroom.

`Publish()` requires reconciled durability state and checks actual managed bytes
against the configured disk quota before and after resolving reused dependencies.
It verifies reused payload contents and counts successful writes. Failed writes
retain their orphan charges; uncertain manifest publication requires recovery.
`Retire()` removes and syncs the manifest before deleting unshared payloads.
Deletion capacity cannot fund new publication until durability is established.
The lowered-budget serving path waits for reclamation and then evicts eligible
checkpoints as necessary; it cannot safely make those decisions from incomplete
usage data.

`ReadPayload()` verifies the selected opened file, type/link/size, checksum and
stable identity, quarantining its owners on failure. Streamed restore settles
submitted copies and invalidates the destination before fallback. These checks
protect selected restored state, but do not discover other orphan/tmp usage or
preserve all startup conflict/incomplete-owner semantics by themselves. Later
restores and publication also touch their selected payload files.

| Possible shortcut | Remaining correctness or memory constraint |
| --- | --- |
| Name-only enumeration | No actual sizes; dependency validation and reclamation still visit the same distinct files |
| Manifest-derived usage | Omits orphan/tmp data and cannot detect enlarged files; unsafe publication headroom |
| Reserve the whole quota as unknown | Safely blocks writes, but cannot resume ordinary publication until reconciliation; that later scan still needs the working set |
| Small batches or delayed worker | Bounds userspace work per batch, not kernel metadata residency after handles close |
| Trust a saved directory stamp | Does not attest in-place child changes or crash-time partial writes |
| Page-cache advice | The existing platform hint does not promise inode/dentry/ARC metadata eviction |

No guard or validation is relaxed in this change. A reliable filesystem usage or
mutation facility, an explicitly trusted durable inventory, or a packed archive
could change the constraint. Each needs separate design and crash/external-change
semantics; no format migration or new trust policy is introduced here.

## Focused regressions

`ManagedUsageRequiresPayloadMetadata()` exercises the real `DiskStore` with tiny
synthetic data. A valid published checkpoint plus a 64 KiB sparse orphan and
17-byte temporary file exceeds actual quota even though manifest-derived usage
would permit the next write. Publication is refused before reading payloads.
Normal reclamation removes only the orphan/tmp files; publication then fits the
exact quota, and restart restores all three synthetic payloads correctly.

A second case grows an existing payload while the store is closed. The parent
directory's mtime remains unchanged without sleeps or forced timestamps.
Recovery nevertheless counts the actual larger file, rejects the incomplete
checkpoint and reads no payload contents. This demonstrates why a directory
timestamp cannot validate an actual-byte inventory.

The native CPU build passed; **5/5 selected tests passed in 19.82 seconds**.
`cache_disk_test`, including the new cases, took 0.57 seconds. These elapsed
results are not a controlled estimate of additional test overhead.

The existing tests retain missing/truncated/corrupt-file cases, symlink/hard-link
checks, all-owner conflict and duplicate-ID rejection, shared-file accounting,
lowered budgets, interrupted publication and manifest-before-payload deletion
durability. The r12 observations are exercised by the shared cache test.

Actual VM commands, using the already configured CPU build:

```sh
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset cpu-test --target cache_disk_test \
  cache_disk_publication_test cache_tiered_test cache_streaming_test \
  component_text_cache_test --parallel 4
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  ctest --test-dir build/cpu-test \
  -R '^(cache_disk_test|cache_disk_publication_test|cache_tiered_test|cache_streaming_test|component_text_cache_test)$' \
  --output-on-failure --parallel 4 --timeout 180
clang-format --dry-run --Werror tests/cache/disk_test.cpp
python3 tools/ci/check-docs.py
python3 tools/ci/check-extension-api.py --base HEAD
python3 tools/ci/generate-infrastructure.py --check
python3 tools/ci/check-format.py
git diff --check
```

Changed-file formatting, documentation, API (unchanged at 0.3.6), infrastructure
and diff checks pass. The whole formatting check still reports the two untouched
baseline violations in `audio_tts_api.cpp:473` and `openai_chat_test.cpp:6750`.

No inference GPU, weights, ZFS memory measurement or owner-machine access was
used. No recovery benchmark was repeated. CPU assertions establish quota and
recovery behavior, not filesystem memory savings or workload capacity.

## Later owner validation, gated on a substantive candidate

**This tests/documentation-only revision does not justify another model load.**
There is no proposed production saving to measure now. Preserve the current
cache and backup, and use the following single validation only after a reviewed
source change addresses the distinct-file working set throughout the lifecycle.

For that changed build, start the Nix daemon and retain the requested job setting:

```sh
sudo systemctl start nix-daemon.service
nix build .#release --jobs 32
```

Use the existing kit's candidate composition and launch workflow; no external
launcher modification or invented command is required. In one maintenance
window, unload production chat first, retain EmbeddingGemma and Open WebUI, and
use the same populated active cache. Do not clear, migrate or replace it, rewrite
the backup, reboot for an identical attempt, flush caches or change kernel/ZFS/
swap settings.

Keep four sessions, context 262144, the same model/quantization/projector, shared
MTP with seven drafts, on-demand vision, 4 GiB requested RAM cache, 50 GiB disk
budget, lazy backing and all reserves. Preserve the launcher's requested-cache
check. Save all r12 phase observations and actual entry/dependency counts.

If startup or the grant is refused, stop and inspect that attempt's log. If it is
accepted, continue the same attempt beyond readiness: restore a known persisted
conversation, exercise concurrent text work, create/publish further checkpoints,
and exercise RAM backing growth and another restore. Retain memory observations
during reclamation, restore and growth, plus actual managed disk bytes, effective
grant, planned payload and committed backing. A startup-only reduction followed
by the same later metadata peak is not success. Exercise an image request and
subsequent text recovery after any clean resource refusal.

Qualification still requires a full requested cache grant and acceptable later
growth/runtime behavior. Until a safe production correction exists and passes
that test, the **451,115,636-byte grant deficit remains unresolved**.
