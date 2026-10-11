# Repository storage cleanup — October 11, 2026

The audit removed disposable generated files and an unused duplicate video. It
preserved original recordings, reviewed annotations, training inputs, models,
verification evidence, the current desktop application and its rollback package.

About **12.9 GB of free space** was available on C: at the final measurement;
the initial measurement was zero. Audited removed files total **12,895,148,835
logical bytes (12.01 GiB)**. These figures differ because filesystem allocation,
hardlinks and other activity affect free space. The final repository inventory
is **62,442,983,700 logical bytes (58.15 GiB)**. The first complete inventory was
69.72 GiB, taken after 0.42 GiB of Next caches had already been removed; the
corresponding reconstructed starting size is approximately 70.14 GiB.

Protected Git/configuration metadata and reparse targets were excluded from the
inventory and deletions. Logical size counts hardlinked paths separately and is
not a measure of allocated disk space. The final inventory has no read errors.

## Removed

| Category | Logical bytes | Decision evidence |
|---|---:|---|
| Four obsolete desktop packages | 6,286,085,964 | `artifacts/repo_cleanup_2026_10_11/releases_removed.json` |
| Unused `frontend/public/game_1.mp4` | 5,572,649,632 | Exact size/SHA equality with the preserved canonical recording; `public_video_duplicate.json`, `public_video_removed.json` |
| Next build caches | 451,768,311 | `frontend/.next/cache` and `frontend/.next-desktop/cache`; actual production output retained |
| 168 test renderer/browser cache directories | 583,806,764 | `artifact_caches_removed.json`; Local Storage, IndexedDB and review drafts excluded |
| 51 project Python bytecode files | 838,164 | `python_bytecode_removed.json`; Python source/dependency runtimes excluded |

Deleted packages: `TT Trajectory Engine`, `TT Trajectory Engine before precision`,
`TT Trajectory Engine training review`, and `TT Trajectory Engine training review
v2`. The launcher now checks the retained `paired training v2`, then `paired
training v1`, then the existing development Electron fallback.

All removal targets were resolved inside the repository, checked for reparse
points and checked against tracked paths. Deletion used native PowerShell literal
paths. No camera was opened and no running project process was terminated.

## Relocated source compatibility

All **29 original recordings (32,166,783,016 bytes)** were SHA-256 fingerprinted
at their new locations under `python_tools/dataset/raw_videos`. The folder counts
are training 14, validation 10 and seal 5. Folder placement does not establish an
independent session split. The same original iPhone session remains training
material; previously confirmed mixed-match compilations remain development
material. No sealed labels or model results were inspected.

Eleven existing imported identities were reconnected through
`training_data/live/source_locations.json`, with its previous version saved as
`relocated_recordings_audit.source_locations.before.json`. Annotation files and
dataset assignments were not edited. Four additional public dataset originals
were not automatically classified as compilations from their folder name.

The inventory CLI now accepts an explicit recordings root and a hash-bound
compilation manifest. A shared Python resolver updates 23 legacy tools to find
known relocated public sources from arbitrary working directories, retains flat
path compatibility and rejects ambiguous copies. It never recursively searches
the sealed folder. The native simulation fallback now finds the nested game_1
recording before the historical flat path.

## Preservation and verification

`artifacts/repo_cleanup_2026_10_11/preservation_final.json` reports zero mismatches:

- 29 original videos and all 95,546 dataset files have unchanged size/mtime
  relative to the audited baseline. The 32 GB videos were not redundantly hashed
  a second time after cleanup.
- 750 reviewed images, 42 capture/index/annotation files and 114 annotation
  revisions have unchanged SHA-256 hashes.
- Sixteen model, checkpoint, registry, resolver and release hash checks passed.
- Current v2 (1,864 files) and rollback v1 (1,859 files) retain their metadata and
  verified application/backend manifest identities.
- 126 Local Storage/IndexedDB/draft files retain their metadata and have no
  overlap with deleted cache roots. The baseline lacked content hashes for
  those files; this is metadata and deletion-exclusion evidence.

Source compatibility verification passed 38 inventory tests and 11 Python
resolver tests; 25 Python files compiled. The updated native C++ candidate built
successfully and loaded the default relocated video. A one-frame analysis journal
confirms simulation mode, frame 1, 1920×1080 dimensions and camera inactive.
That fixture is deliberately `PARTIAL`, not a complete recording, accuracy or
latency acceptance test. See `native_source_check_verified.json` and the full
`native_source_check_v3.log`. Earlier failed harness attempts are retained as
logs; they used an invalid flag or omitted the required request file.

The optional raw-tensor deduplication helper passed 16 safety tests. Its read-only
run rehashed 342 diagnostic blobs. None met the identical timestamp/metadata
condition for safe hardlink replacement, so no evidence blob was modified. The
611 MB of byte-identical candidates remain preserved. Different timestamps were
not discarded to gain storage.

## Why the remaining large folders stay

The raw videos support annotation, continuity checks, full recording replay and
future retraining. Extracted YOLO images, `detector_improvement/clean_dataset`,
`scene_detector/dataset` and `multigame` are referenced by current preparation or
training workflows. Some game_1 images already share NTFS storage through
hardlinks. Current dependencies, build output, model/checkpoint lineage and
regression evidence remain necessary for implementation and rollback.

## Can videos be deleted after training?

A trained detector can perform live inference using its weights without the raw
training videos. **Do not delete the only original copies now.** The project is
still collecting/reviewing data and has not passed the agreed model release
gates. Existing reviewed images alone do not preserve every source frame,
temporal window, timestamp or future correction opportunity.

For the next large storage saving, archive originals to another drive. Verify
the archive against the source SHA manifest, retain labels/manifests/model
lineage, and update source locations before removing the local copies. Complete
recording playback and reanalysis still require access to the video. Avoid
recompressing an original that existing annotations and hashes identify.

Cleanup did not train or promote a model. Frozen plans hashing the old source
resolver or modified compatibility scripts must be regenerated before fitting;
the independent paired-camera data and reviewed-coverage dependencies remain.
