# Focused annotation review — October 12, 2026

This checklist comes from the post-registration strict audit. It does not change
annotations or decide where the ball is. Keep all other completed reviews.

## Seven duplicate disagreements in Video Project 6.mp4

The same source images have differing saved ball targets in two review batches.
Check the actual ball and its exposure center/extent in each image, then save
consistent reviewed labels for both copies. Do not copy either existing label
without inspecting the image. If it cannot be located reliably, keep it unknown
and excluded. Model predictions are suggestions, not reference labels.

Batch 1 is `phone_452627e98a3c2652_1` (300 images); batch 2 is
`phone_452627e98a3c2652_2` (120 images). The review counter is one-based and
matches the number in `frame_N.jpg` for these batches.

| Original source frame | Batch 1 review counter/file | Batch 2 review counter/file |
|---|---|---|
| 7297 | 236 / `frame_236.jpg` | 103 / `frame_103.jpg` |
| 7298 | 237 / `frame_237.jpg` | 104 / `frame_104.jpg` |
| 7299 | 238 / `frame_238.jpg` | 105 / `frame_105.jpg` |
| 8383 | 271 / `frame_271.jpg` | 112 / `frame_112.jpg` |
| 8384 | 272 / `frame_272.jpg` | 113 / `frame_113.jpg` |
| 8385 | 273 / `frame_273.jpg` | 114 / `frame_114.jpg` |
| 8386 | 274 / `frame_274.jpg` | 115 / `frame_115.jpg` |

These are seven unique frames. Each appears twice in the diagnostic list because
both source-frame identity and decoded-image identity identify the disagreement.

## Two unknown-visibility labels in IMG_7944.MOV

Capture `phone_e23544b0c4c1a850_1` retains saved boxes but explicitly unknown
visibility at review counters 11 and 12, original frames 9974 and 9975.
Inspect each image and choose Visible or Partly visible only when supported.
Use Not in frame only when absence is verified; otherwise leave Hidden / unsure.
Save in Ball only mode to preserve existing people/table review status.

## After review

Regenerate the strict dataset plan before preparing a training export. This
checklist is a snapshot, so it does not mark items resolved automatically.
Resolving these frames cannot replace the missing paired A/B camera sessions.

Evidence: `artifacts/yolo_paired_training_plan_2026_10_12_v2.json`, both Video
Project 6 frame indexes/annotations, and the IMG_7944 frame index/annotations.
