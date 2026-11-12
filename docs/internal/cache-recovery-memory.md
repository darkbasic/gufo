# Memory observations across populated cache recovery

This follow-up adds bounded production observations to the shared cache-startup
path. It does not establish a memory saving or change text admission, cache
grants, reserves, vision admission or lazy backing.

## Established evidence and remaining constraint

The owner tested Gufo 0.34.0 with `temp11`, based on
`14255c94aca9b80bc2e253efefba4c6719de8935`. Four full-context sessions loaded with
MTP and seven draft tokens under on-demand vision admission. The launcher refused
the **3,800,008,076-byte grant**, which is **494,959,220 bytes below** the requested
4,294,967,296 bytes.

Recovery accepted 100 manifests, read 75,143,800 manifest bytes, and observed
1,137,375 references naming 184,263 distinct payloads. It performed 184,263
dependency stats and took 2,316,649 microseconds. The preceding implementation
performed 1,137,375 dependency stats and took 3,334,759 microseconds. This owner
evidence supports reduced recovery work, not a measured reduction in startup
host-memory footprint. The earlier synthetic benchmark was not repeated.

From `post_cache_resources` to `pre_slots`, host availability fell from
38,911,946,752 to 37,878,001,664 bytes: **1,033,945,088 bytes**. HIP free stayed at
40,447,307,776 bytes. Owned manifest storage was 76,293,516 bytes; logical
live/peak RAM charges were 76,356,316 / 223,078,719 bytes. Logical charges are
neither RSS nor physical allocation measurements.

After slots, the existing grant arithmetic already includes manifest ownership:

```text
8019730432 - 4294967296 + 76293516 - 1048576 = 3800008076
```

No second manifest credit is justified. Nothing in this patch weakens that guard
or promises that the full cache backing and occasional images fit.

## Audited interval

After resource construction, serving copies the compatibility identity,
initializes the empty prefix index and retention facade, charges their metadata,
hashes the identity, obtains a nonce and registers the component inventory.
Registration creates an empty inventory tree; it does not import checkpoints.

`DiskStore` then acquires directory ownership, opens/creates its layout and syncs
it and the ancestors. It scans manifests, chunks, private and temporary files,
reserves the entry table, and scans manifests a second time. Each manifest is
opened/statted, read into a temporary byte vector, planned, decoded and
structurally validated. Accepted vectors and availability/pin ownership remain.
Serialized read buffers and per-manifest validation sets die before the next
manifest is processed.

The shared-reference pass owns one bounded map and compact owner arrays. It
checks distinct dependencies, excludes incomplete owners, validates geometry,
private aliases and checkpoint IDs, and rejects every conflicting owner. These
temporary containers and their ledger reservation die before final directory
syncs and sorting finish recovery. The recovery log is formatted before
`pre_slots`.

Slots, backing pages, persisted catalog/index population, `StreamedStore`
staging, orphan reclamation workers and the persistence queue occur **after**
`pre_slots`. They cannot explain this interval. The source audit found no
justified approximately 1 GiB retained buffer to remove. Avoiding the second
manifest enumeration would remove roughly 100 entry stats here, not justify a
capacity claim; it would also require retaining a separate discovery list.

## Production observations

With disk persistence enabled, successful startup emits exactly eight
`schema=component-cache-v1 event=recovery_memory` records:

| Phase | Completed work / storage still owned |
| --- | --- |
| `post_cache_resources` | Resource factory returned; identity/index initialization has not started |
| `before_disk_recovery` | Serving identity, inventory and facade setup complete |
| `after_directories` | Four initial scans complete and enumeration handles closed |
| `after_manifests` | Second manifest scan complete; decoded owners retained, per-manifest temporaries destroyed |
| `validation_live` | Dependency and conflict validation complete; temporary map, arrays and reservation still live |
| `after_validation` | Those temporary containers and reservation destroyed |
| `after_disk_recovery` | Final directory syncs/sort and constructor complete |
| `pre_slots` | Recovery summary logged; no session slots or backing allocated yet |

The first shared-path record is just after the runner resource factory returns;
the runner's existing HIP observation occurs inside that factory. They are
adjacent observations, not one atomic reading. Compare fields within the new
event sequence, and retain the existing HIP logs as separate evidence.

Each record reports process RSS and lifetime high-water RSS; host MemAvailable,
Cached, Slab and SReclaimable; optional ZFS ARC size; directory-entry and
dependency counts; manifest bytes read; and logical live/peak/reserved metadata
charges. `directory_entry_stats` counts successful metadata observations reported
by startup enumeration, including ignored/nonregular entries and the second
manifest scan. It excludes dependency checks, failed enumeration attempts and
stats of opened manifest files. It is not a count of distinct payloads.

`sample_begin_us` is relative to shared cache construction; `since_previous_sample_us`
measures between sample starts, so it includes the preceding observation/logging
cost. `collection_us` measures OS collection itself. The first interval includes
resource construction. Existing `recovery_us` now includes the four internal
diagnostic callbacks, so it is not an uninstrumented timing comparison.

The platform abstraction reads Linux `/proc/self/status`, `/proc/meminfo` and,
when accessible, `/proc/spl/kstat/zfs/arcstats`. The ARC parser accepts the
unsigned `size` counter; its type is defined by the
[OpenZFS kstat interface](https://github.com/openzfs/zfs/blob/master/include/os/linux/spl/sys/kstat.h).
One reusable 64 KiB buffer bounds each file read, with one extra byte to detect
overflow. Missing, unsupported, malformed, duplicate or overlong values are
`unavailable`, never an invented zero. Windows exposes working-set/current peak
and available physical memory; Linux-specific categories remain unavailable.

These are sequential, approximate observations, explicitly marked
`snapshot_atomic=false categories_overlap=true`. SReclaimable overlaps Slab;
RSS, kernel caches and ARC are not disjoint allocation buckets. Do not add their
deltas together or subtract logical reservations from RSS. High-water RSS covers
the process lifetime, including earlier model loading; it is not a resettable
phase peak. Freeing temporary containers does not guarantee pages return to the
OS. An RSS change is also not a direct measurement of initialized C++ heap.

Collection occurs only at these fixed startup boundaries, with no per-file or
per-token polling and no global allocation instrumentation. The synchronous
disk observer is not retained; observer exceptions cannot change recovery
acceptance, ownership or durability. There is no new policy option.

## Native CPU validation

The existing CPU build was reused. **8/8 tests passed in 19.92 seconds**, including
production recovery/publication, restore, conflict/missing-file cases, synthetic
backing, shared serving startup, platform reader/parser and GGUF identity.
New tests check four disk callbacks/eight serving records independent of file
count, live-versus-released validation charges, ignored directory entries,
observer failures and no callbacks during later publication/restore/destruction.
Parser and actual bounded-reader fixtures check unavailable versus zero,
malformed/duplicate input, overflow, exact limits, missing and unreadable files.

Actual VM commands (the isolated dependency prefix was already configured):

```sh
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset cpu-test --target memory_observation_test \
  cache_disk_test cache_disk_publication_test component_text_cache_test \
  cache_tiered_test cache_streaming_test cache_full_backing_test \
  gguf_identity_test --parallel 4
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  ctest --test-dir build/cpu-test \
  -R '^(memory_observation_test|cache_disk_test|cache_disk_publication_test|component_text_cache_test|cache_tiered_test|cache_streaming_test|cache_full_backing_test|gguf_identity_test)$' \
  --output-on-failure --parallel 4 --timeout 180
c++ -std=c++20 -O2 -Wall -Wextra -pthread -I. -Iinclude \
  -c src/cache/disk.cpp -o /tmp/gufo-recovery-observation-production.o
python3 tools/ci/check-extension-api.py --write
python3 tools/ci/check-extension-api.py --base HEAD --labels '["ext-api"]'
python3 tools/ci/generate-infrastructure.py --check
python3 tools/ci/check-docs.py
python3 tools/ci/check-format.py
```

Production disk code also compiles without test hooks. API 0.3.6,
infrastructure and documentation checks pass. The whole formatting check still
reports only the pre-existing `audio_tts_api.cpp:473` and
`openai_chat_test.cpp:6750` violations. Changed C++ files pass separately.
Nix and HIP are unavailable in this VM. No HIP translation unit changes; the
Windows branch of `os_info.cpp` remains uncompiled here. These CPU tests establish
neither a physical-memory saving nor GPU/workload qualification.

## One owner attempt with new evidence

Any owner build starts by enabling the daemon and retains the existing 32-way
setting. For the repository release build:

```sh
sudo systemctl start nix-daemon.service
nix build .#release --cores 32
```

For the existing workstation kit, incorporate this source change into its normal
candidate composition and keep its existing 32-way build setting and requested
session/cache checks. No new launcher command or deployment change is required.
Include the additive extension record when preparing the source: Nix consumes
tracked files. This VM has not staged, committed or pushed the change.

Use one owner-controlled maintenance window: unload production chat, retain
EmbeddingGemma and Open WebUI, and use the existing **populated** active cache.
Do not delete, replace or migrate it, and do not delete or rewrite the backup.
Keep four sessions, context 262144, MTP with seven drafts, the same model and
projector, on-demand vision, 4 GiB RAM request, 50 GiB disk budget, lazy backing
and all reserves/settings unchanged.

Save the complete startup log from this changed candidate. If the launcher
refuses the grant, stop after that attempt and inspect the saved log; do not
repeat an identical model load or substitute an empty-cache test. The new
evidence is where current/high-water RSS and global filesystem/kernel counters
change relative to scans, decoding, live validation and destruction:

- A host decline during scanning without comparable process growth points
  investigation outside retained process storage; cache/slab/ARC observations
  can narrow it, without proving exclusive attribution.
- RSS growth across decoding or live validation identifies process residency
  during that work. Persistent RSS after logical charges release is evidence
  to investigate allocator/native retention, not by itself a leak.
- Missing ARC visibility stays unavailable. An unexplained residual remains
  unexplained; do not assign it to ARC, duplicate reservations or later slots.

Keep grant, planned/committed backing and runtime image capacity separate. The
capacity problem remains unresolved until this new evidence supports a safe
correction and subsequent hardware qualification.
