# GGUF identity cache

Gufo hashes every byte of every GGUF shard, including metadata and tensor
padding. It reuses a cached full-file digest only when the record establishes
that the file's timestamps were outside the ambiguity window before scanning.

## Why matching metadata can be insufficient

The cache key contains the open file's device, inode, size, modification time
and change time. A same-size rewrite can preserve all five fields when the
filesystem assigns both writes the same timestamp; restoring modification
time does not necessarily leave a distinct change time. Comparing the key
before and after lookup cannot detect that collision. This is the same class
of ambiguity described by [Racy Git](https://git-scm.com/docs/racy-git).

The reported ZFS failure had changed on-disk bytes and unchanged key fields.
The regression therefore needs a new full-content digest, not timestamp
manipulation, retries, or a different sample of the file.

## Publication and reuse

A cacheable scan must start with both timestamps positive and strictly more
than 2 seconds behind the wall clock. The scan reads the full open file and
checks its metadata again. A backward clock observation prevents publication;
if detected after a cache hit, Gufo discards that record and hashes the file.
If a formerly qualified record must be retired but cannot be removed, lookup
fails explicitly. Restore cache access and discard that record before retrying;
clock recovery alone does not repair a failed retirement.

Records contain the `gguf-sha256-cache-v2` format tag, scan-start time and full
SHA-256 digest. Lookup validates the timestamp relationship against that
stored scan-start time, as well as the current clock. The stored proof does
not change as the record ages. A record made inside the ambiguity window
cannot become trustworthy merely because a later lookup happens much later.

For a recent file, Gufo returns the freshly computed identity without
publishing a reusable digest. If hashing takes long enough for the file to
leave the ambiguity window, that scan still cannot qualify retroactively.
A later lookup must hash the full content again before publishing. Subsequent
lookups of a stable file read the small cache record and inspect metadata;
they do not scan the model again.

Gufo retains the existing open-file checks, owner/permission checks and
no-follow cache opens. It also checks the cache record's metadata after
reading it. Publication uses an exclusive temporary file and atomic rename.
Observed model metadata changes during hashing or lookup cause an error.
The private test hooks override model metadata and clock observations only;
cache security checks and content hashing still use real files.

## Existing cache compatibility

The directory remains `$XDG_CACHE_HOME/gufo/gguf-sha256-v1` (or the platform's
usual per-user cache base). The content identity scheme also remains
`gguf-sha256-v1`: identical bytes produce the same identity, so correctly
identified checkpoint data does not need a format migration.

Old 65-byte digest records contain no scan-start proof and are never reused,
even if their files or records are old. The first lookup recomputes the full
digest and replaces an eligible record with the new format. Budget for one
full scan per stable artifact after upgrading; recent files can require an
additional later scan. Old Gufo versions likewise cannot read the new record
format, so alternating versions may repeat this work. No cache deletion or
model download is required.

## Clock and filesystem limits

The 2-second window assumes a coherent local filesystem with a meaningful
change timestamp, with timestamp granularity plus update lag strictly below
2 seconds. File timestamps and the process wall clock must use the same time
base, without unobserved backward jumps across writing, scanning and reuse.
Keep model artifacts immutable while they are being loaded or used.

This is a bounded timestamp policy, not proof that identical metadata implies
identical content. Arbitrary metadata-preserving changes, clock rollback and
recovery between observations, restored filesystem snapshots, or incoherent
remote timestamps can violate the assumptions. Detecting such changes needs
another full-content read or a reliable filesystem change signal. The Windows
platform's last-write-time fallback on FAT/exFAT is not a meaningful independent
change timestamp and falls outside this reuse guarantee.

## CPU and owner validation

`gguf_identity_test` uses synthetic GGUF files with controlled model metadata
and clocks, while reading and hashing real bytes. It covers key collisions,
reopening, suspicious and legacy records that age, boundary/future timestamps,
changes during hashing and lookup, malformed records, permissions and symlinks.
Counters distinguish full scans from stable cache hits with zero hashed bytes.
The original real-filesystem assertion still requires a correct identity after
a same-size edit with restored modification time; its immediate record-count
expectation is separate because recent files may not publish a record.

As of November 9, 2026, the VM results on the patch based on `67ca4760` are:

| Check | Result |
| --- | --- |
| Focused CPU target build | Passed with GCC 15.3; HIP disabled |
| `gguf_identity_test` | Passed; 0.12 seconds in the final CTest run |
| Identity, reader, filesystem, mapped-file and atomic-publication regressions | 5/5 passed; 0.15 seconds total |
| Targeted ASan/UBSan with leak detection | Passed; 0.202 seconds, no diagnostics |
| Changed C++ formatting, documentation, extension API and diff checks | Passed; extension API remains 0.2.21 |

The targeted sanitizer build instruments the identity implementation and its
test; reader/platform dependencies use the prebuilt CPU libraries. Its exact
command and output are retained in `/tmp/gufo-gguf-identity-sanitizers.log`.
This is not whole-tree sanitizer coverage. The shared formatting check with
the available clang-format 22.1.8 reports two unchanged files,
`src/cli/serve/audio_tts_api.cpp` and `tests/cli/openai_chat_test.cpp`; they were
left untouched. No vision files or commits changed in this patch.

The official `77dff828be02a498e9951694f31b024e7ac87119` object is unavailable
in this checkout, so the reported base comparison was not independently
repeated. The inspected current implementation has the reported metadata-key
behavior. No ZFS reproduction, Nix run, inference-host access or model loading
was performed.

To repeat the focused VM checks with the documented build dependencies:

```sh
cmake --build --preset cpu-test --parallel 4 --target \
  gguf_identity_test gguf_reader_test platform_fs_test \
  platform_mapped_file_test platform_publish_test
ctest --preset cpu-test --output-on-failure -R \
  '^(gguf_identity_test|gguf_reader_test|platform_fs_test|platform_mapped_file_test|platform_publish_test)$'
```

The development VM cannot establish the reported ZFS behavior or run the
owner's pinned Nix environment. On the ZBook, run the focused CPU regression
with synthetic fixtures; no GPU, weights, inference service or kernel changes
are needed:

```sh
nix develop -c cmake --preset cpu-test
nix develop -c cmake --build --preset cpu-test --parallel 4 --target gguf_identity_test
nix develop -c ctest --preset cpu-test -R '^gguf_identity_test$' --output-on-failure
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
nix flake check
```

Before the flake check, ensure the reviewed checkout includes the new internal
header and this note: flakes omit untracked files. Stage only this patch's
paths if needed, preserving unrelated work. Retain the exact revision,
toolchain, filesystem and test output; do not retry until success or interpret
timing-sensitive failure frequencies as statistical estimates.
