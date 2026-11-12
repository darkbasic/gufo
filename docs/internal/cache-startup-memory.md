# Populated cache startup: remaining allocations and recovery storage

This correction belongs to shared checkpoint-cache startup, independently of
vision admission and GGUF identity. It preserves configured budgets, lazy
backing, host/device reserves, existing checkpoint formats and the launcher's
effective-cache check. It does not make clearing persistence a startup policy.

## Owner evidence and provenance

The owner supplied two r8 attempts on November 11, 2026, using the same build
`local-0330-vision920-cache-68a06d8135c0` and composed source
`1e46016c50bedf8fe4337c82db06f19373ee91eb`. That deployed composition need not
exist in this checkout. The implementation starts from `a0e0a5ea` and retains
the preceding cache-transfer and on-demand vision work.

| Startup persistence | Automatic bytes | Maximum bytes | Granted bytes |
| --- | ---: | ---: | ---: |
| Populated | 4,144,392,192 | 3,992,768,512 | 3,992,768,512 |
| Empty, after owner backup | 4,584,632,320 | 4,873,248,768 | 4,294,967,296 |

The populated grant was **302,198,784 bytes below** the requested 4 GiB. Its
post-resource to pre-slot host-availability decrease was **828,944,384 bytes**,
while HIP free memory did not change. This difference is not a measurement of
C++ metadata, a leak, or an allocation that can be credited to free memory.
Filesystem cache, allocator retention and allocations outside Gufo's ledger
can affect the observation.

The old persistence had 94 manifests totaling 74,135,862 bytes, 96,624 chunk
files totaling 22,460,427,264 bytes and 12,169 private files totaling
12,833,854,012 bytes, with no temporary files. Serialized manifest bytes do not
give the retained decoded storage size or the unknown reference distribution.

After the empty-cache startup, the owner worked for hours with three concurrent
agents, sampled about 5.3, 4.1, 3.6, 3.5 and 3.7 GiB of MemAvailable, and saw zero
swap usage. That is useful operational evidence, not qualification of populated
restart, full cache occupancy, leak freedom or every four-request/image peak.

Disk recovery and the post-slot grant calculation already existed in upstream
0.33.0 (`77dff828be02a498e9951694f31b024e7ac87119`). Before this patch,
`disk.cpp` and `disk_catalog.cpp` were byte-identical to that base. The independent
disk-publication fix is part of the owner's external composition, not this
checkout's tested source. This patch leaves the publication protocol/format and
GGUF identity correction unchanged; the composed candidate must retain the
owner's independent fixes.

## Accounting invariant

DiskStore reads and owns decoded manifest vectors before the post-slot host
observation. Those vectors are charged inside the total RAM retention budget.
The previous grant cap compared that entire budget with remaining host headroom,
charging this already-owned part twice. Pinned execution slots already had a
separate reconstruction because their storage sits outside the cache budget.

Let `H` be the remaining host ceiling after its existing reserve, `R` the total
cache grant, `O` the initialized manifest-vector storage already owned, and `S`
the future disk staging allowance. The remaining obligation is:

```text
max(0, R - O) + S <= H
```

The grant calculation reconstructs only `O` when applying the total-budget cap.
The automatic half-available sizing similarly reconstructs half of the existing
owned storage. Existing pinned-slot treatment, initial ledger limits and staging
deduction remain. The final ledger must still fit every charge and reservation;
an explicitly undersized total budget or insufficient host headroom still fails
or trims under the existing policy.

`DiskStore::OwnedManifestBytes()` counts initialized input, token, component and
chunk-vector elements retained by accepted manifests. It excludes spare vector
capacity, ledger-only estimates, pending reservations, map/owner overhead,
temporary decoding/validation storage, checkpoint payloads and later
catalog/index allocations. The method is an ownership-based accounting bound,
not an RSS or physical-residency measurement. It does not add ledger RAM, ARC,
historical peaks or the observed host decline to free memory.

The HIP backing provider still receives the total budget and uses the model's
`BackingPlan()` to leave room for metadata/headroom. Registered payload growth
retains its ordinary ledger and host/device pressure checks. Neither full future
cache growth nor image capacity becomes a startup guarantee.

The `disk_recovered` diagnostic reports manifest/dependency counts, manifest
bytes read, rejected entries, owned manifest bytes, logical live/peak RAM charges
and recovery time. These logical counters are not heap/RSS measurements.
The `ram_admission` diagnostic records the same-stage host ceiling, owned
manifest bytes, pinned slots, staging, ledger RAM, proposed grant and remaining
additional bytes. `configured` reports the effective total grant. Keep these
separate from the backing plan and actually committed pages.

## Cold recovery validation

Recovery continues to validate all manifests and dependency geometry before
building a trusted catalog. It must not use publication-only validation against
an assumed-valid catalog.

The compact pass keeps one borrowed canonical reference and conflict flag per
unique payload identity, comparing fixed-size geometry without allocating a
geometry vector for each reference. A second full scan marks every owner of a
conflicting payload before discarding any entry, and separately handles duplicate
checkpoint IDs. This removes per-payload owner vectors without making enumeration
order decide which conflicting checkpoint survives or allocating one large
reference array when many manifests share the same payloads.

Retained manifest storage, temporary validation allocation and logical ledger
charges remain separate quantities. A reduction in temporary heap allocation
does not by itself prove that the allocator returns those pages to the OS before
the host recheck. Startup reads manifests and stats dependencies; streaming
restore still checks payload contents before making a destination executable.

## CPU measurement

Matched instrumented production `DiskStore` builds compared `a0e0a5ea` with the
candidate using GCC 15.3.0 on the development VM. Each fixture contained 16
manifests, 512 chunk references and four private files per manifest: 8,256 total
references and 302,080 manifest bytes. One shape had 8,192 unique chunk files;
the other shared all 512 chunks across the 16 manifests. Both had 64 private
files. Neither shape reproduces the owner's unknown reference distribution.
Fixture creation and recovery run in the same process, so filesystem metadata
is warm; this is not a cold OS-cache benchmark.

| Quantity | Unique chunks: before | Unique chunks: after | Shared chunks: before | Shared chunks: after |
| --- | ---: | ---: | ---: | ---: |
| Logical live RAM bytes | 314,112 | 314,112 | 314,112 | 314,112 |
| Logical peak RAM bytes | 2,759,424 | 1,313,360 | 2,759,424 | 1,313,360 |
| Validation C++ heap peak bytes | 1,850,704 | 792,848 | 191,824 | 55,568 |
| Validation C++ allocations | 33,059 | 8,260 | 19,747 | 580 |
| Validation time, ms | 2.589 | 1.967 | 0.775 | 0.619 |
| Whole recovery time, ms | 27.002 | 24.697 | 10.824 | 10.802 |
| Process maximum RSS, KiB | 7,172 | 5,512 | 5,000 | 4,944 |

Validation heap measurements count ordinary C++ requested bytes between entry
and exit of the production function, not allocator usable bytes, all native
allocations or RSS. Temporary live bytes returned to zero. Whole-recovery C++
live bytes stayed 322,137 in both shapes and revisions. Maximum RSS includes
fixture creation, the whole process and profiler allocation headers; it is not
an attributable resident-memory saving at the later host admission snapshot.
These were single instrumented runs per shape/revision, not timing qualification.
The table uses the final harness with non-inlined replacement allocation
functions. Preliminary runs had identical allocation counts and byte peaks;
their timings differed, including slightly slower whole recovery for the shared
candidate. Do not compare timings across harness revisions or discard that
earlier observation as proof of performance improvement.
The [recorded measurements and raw logs](cache-startup-memory-results.json)
retain the observations and their instrumentation limits.
The measured validation heap-peak reductions are 1,057,856 bytes for unique
chunks and 136,256 bytes for shared chunks. Retained storage is unchanged: its
startup credit corrects the remaining-obligation calculation rather than freeing
those buffers.

The optional `cache_disk_startup_bench` target builds under the CPU preset and
does not run in the normal regression gate. Its arguments are manifest count,
chunks per manifest, private files per manifest and shared chunk count. It
creates and removes an isolated temporary directory with sparse payload files;
startup does not read their contents. It writes manifests and filesystem
metadata, not tens of GiB of payload data.

```sh
cmake --preset cpu-test -DGUFO_BUILD_TOOLS=ON
cmake --build --preset cpu-test --target cache_disk_startup_bench --parallel 4
/usr/bin/time -v build/cpu-test/tests/cache/cache_disk_startup_bench 16 512 4 0
/usr/bin/time -v build/cpu-test/tests/cache/cache_disk_startup_bench 16 512 4 512
```

The repository-built target was also run with no arguments: eight manifests,
544 references, 22,016 manifest bytes, and 2.07534 ms of recovery. Logical
live/peak RAM was 28,416/94,440 bytes; ordinary C++ live/peak heap was
33,625/64,545 bytes; whole-process maximum RSS was 4,600 KiB. It exited zero.
The oversized arguments `1000 60000 255 0` returned exit 2 at the aggregate bound.
These are fixture/CLI checks, not a second matched timing comparison.

```sh
/usr/bin/time -v build/cpu-test/tests/cache/cache_disk_startup_bench
build/cpu-test/tests/cache/cache_disk_startup_bench 1000 60000 255 0
```

The ordinary target reports whole recovery. The scoped table used the following
matched temporary copies of the production translation unit, with entry/exit
callbacks only. It does not change the checkout or substitute another validator:

```sh
python3 - <<'PY'
from pathlib import Path
import subprocess
before = subprocess.check_output(
    ['git', 'show', 'a0e0a5ea:src/cache/disk.cpp'], text=True)
after = Path('src/cache/disk.cpp').read_text()
openings = ('  void ValidateReferences(bool reject = true) {\n',
            '  void ValidateReferences() {\n')
scope = ('    GufoDiskValidationBegin();\n'
         '    struct ValidationMeasure { ~ValidationMeasure() { '
         'GufoDiskValidationEnd(); } } measure;\n')
declarations = ('extern "C" void GufoDiskValidationBegin();\n'
                'extern "C" void GufoDiskValidationEnd();\n')
for name, source in [('before', before), ('after', after)]:
    opening = next(value for value in openings if source.count(value) == 1)
    Path(f'/tmp/gufo-disk-recovery-{name}-measured.final.cpp').write_text(
        declarations + source.replace(opening, opening + scope))
PY
for variant in before after; do
  c++ -std=c++20 -O2 -g -UNDEBUG -DGUFO_CACHE_TESTING -pthread -I. -Iinclude \
    tests/cache/disk_startup_bench.cpp \
    /tmp/gufo-disk-recovery-${variant}-measured.final.cpp -Wl,--start-group \
    build/cpu-test/src/cache/libgufo_cache.a \
    build/cpu-test/src/core/platform/libgufo_platform.a -Wl,--end-group -ldl \
    -o /tmp/gufo-disk-recovery-${variant}-bench.final
  /usr/bin/time -v /tmp/gufo-disk-recovery-${variant}-bench.final 16 512 4 0
  /usr/bin/time -v /tmp/gufo-disk-recovery-${variant}-bench.final 16 512 4 512
done
```

An explicitly selected larger shape, `94 1028 129 0`, creates 96,632 chunk files
and 12,126 private files. These approximate file counts only, not the owner's
manifest sizes or sharing. Do not interpret a successful benchmark as proof of
populated-cache hardware fit, restore correctness or the full daily workload.

## VM regressions and limits

Production-path CPU tests cover empty and populated startup, explicit and
automatic grants, exact fit and one-byte pressure, separate staging/pinned-slot
accounting, no credit for pending reservations or converted estimates, restart
and restored synthetic state, all-owner conflicts, duplicate checkpoint IDs,
missing/truncated/corrupt dependencies and private aliases. Failure cases check
charge and directory-ownership release without discarding valid persistence.

The final full CPU PR gate passed **116/116 tests in 50.26 seconds**.
The initial gate passed **115/116 tests in 55.36 seconds**. The new broad
heap-failure sweep aborted in the pre-existing no-throw `Directory::Open`
pathname copy (`fs_posix.cpp:109`), before reaching cache reference validation.
The corrected sweep targets actual heap
allocations inside `ValidateReferences()`; whole-constructor ledger reserve and
convert failure sweeps remain. It does not claim arbitrary startup heap failures
are safe in the unrelated platform code. After that test-scope correction,
`cache_disk_test` and `component_text_cache_test` passed **2/2 in 1.45 seconds**.

The VM used its available compiler and an isolated dependency prefix, not Nix
or HIP. The actual configure, initial gate and final focused commands were:

```sh
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --preset cpu-test -DCMAKE_PREFIX_PATH=/tmp/gufo-cpu-deps/usr \
  -DZLIB_LIBRARY_RELEASE=/usr/lib/x86_64-linux-gnu/libz.so.1 \
  -DGUFO_BUILD_TOOLS=ON
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset cpu-test --target cache_disk_startup_bench --parallel 4
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset pr --parallel 4
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  PKG_CONFIG_SYSROOT_DIR=/tmp/gufo-cpu-deps \
  PKG_CONFIG_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu/pkgconfig \
  cmake --build --preset cpu-test --target cache_disk_test \
  component_text_cache_test --parallel 4
env LD_LIBRARY_PATH=/tmp/gufo-cpu-deps/usr/lib/x86_64-linux-gnu \
  ctest --test-dir build/cpu-test \
  -R '^(cache_disk_test|component_text_cache_test)$' \
  --output-on-failure --timeout 60
```

The same PR-gate command was rerun after the scope correction. The initial gate
log is `/tmp/gufo-cache-startup-pr.log`, the corrected focused result is
`/tmp/gufo-cache-startup-final-tests.log`, and the successful final full gate is
`/tmp/gufo-cache-startup-pr-final.log`. The API diff gate passed at
**0.2.27 / 130 surface paths**, with the `ext-api` label. Whole-tree formatting
still reports two unchanged baseline violations in
`src/cli/serve/audio_tts_api.cpp:473` and `tests/cli/openai_chat_test.cpp:6633`.
Formatting of all six changed C++ files passed.

The changed shared cache sources compile through the native CPU targets. No HIP
translation unit changed in this correction, and no real HIP execution occurred.
A capable environment still needs the normal release build and owner validation:

```sh
cmake --preset release
cmake --build --preset release --parallel 4
```

## Owner validation with populated persistence

Use one owner-controlled maintenance window and the normal kit workflow. Unload
production chat before starting the changed candidate. Keep EmbeddingGemma and
Open WebUI running so the daily budget is represented. Do not alter queue/kernel
settings, swap, reserves or the launcher's session/cache checks.

Preserve both the owner's populated backup and the currently active cache.
Prepare a **disposable disk-backed copy** of populated persistence using ordinary
copy tools; do not share writable files through hard links or point the candidate
at either original directory. Use the kit's existing candidate source selection
and a separate unused port. Do not invent launcher options or clear caches.

Keep four sessions, context 262,144, maximum output 32,000, the same model and
quantization, shared Q8_0 MTP with seven draft tokens, the loaded BF16 projector,
`--vision-memory-policy on-demand`, requested RAM 4,294,967,296 bytes and disk
53,687,091,200 bytes.

1. Start the changed candidate against the populated copy and retain its build
   identity and complete log. Record recovery counts, same-phase host/HIP
   observations, `owned_manifest_bytes`, the proposed/effective grant, planned
   backing payload/metadata and actually committed backing. Require the full
   requested grant; readiness alone does not qualify it. After any startup or
   grant refusal, stop and inspect that attempt's saved log without retrying
   with an empty cache.
2. Use the existing cache workload to capture a known synthetic conversation,
   verify RAM reuse, and wait for durable disk publication. Check text progress
   with distinct task IDs. Preserve later backing-growth or image refusals as
   distinct outcomes rather than interpreting the logical grant as full payload
   capacity.
3. Restart the candidate once against the **same populated disposable copy**.
   Repeat the known history and verify the disk hit, expected cached-token count
   and restored continuation results. This restart collects the missing
   populated recovery/restore evidence; it is not an unchanged empty-cache probe.
4. Exercise an image-bearing request while text work progresses. Record image
   admission or its structured resource refusal and subsequent text recovery.
   Retain peak host/device memory, swap, TTFT and throughput observations.

Real HIP execution, populated-cache hardware qualification and the full daily
workload remain pending until the owner runs this procedure. Holding the recorded
post-slot host ceiling unchanged, the accounting correction alone would need
`owned_manifest_bytes >= 302198784` to close the old grant gap. The owner's actual
owned amount and the effects of smaller temporary allocations on its later host
observation remain unmeasured. Do not assume either correction funds the full
grant, later backing growth or the rest of the requested workload.
