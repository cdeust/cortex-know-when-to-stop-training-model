# BEAM data validation correction

Source: `cdeust/cortex-know-when-to-stop-training-model` at `0687f7a73aa2a98e5bfe8f6495b1c7013c4a775e`, `data/seed/beam.jsonl`.
The failed CI job was https://github.com/cdeust/cortex-know-when-to-stop-training-model/actions/runs/39693459715/job/119545026374.

The original validator reported 103 duplicate pairs because its key truncated each text to 100 characters. Comparing full texts finds 85 repeated pairs: 84 records identical in every field and one pair with contradictory labels. The remaining 18 reports concern distinct full passages, which remain in the dataset.

The corrected dataset contains 19,025 records (9,116 relevant, 9,909 irrelevant), from 19,111 records (9,199 relevant, 9,912 irrelevant). The correction removes 84 redundant records and moves both contradictory records, at original lines 17251 and 17255, into `beam-label-conflicts.jsonl` with source provenance. No label was inferred or changed. The first occurrence of each other record and its original serialized text are retained.

`quarantine/` is outside the training and validation `data/` tree. The conflicting pair needs source review before either record returns to training. Historical model metrics describe the original dataset; this correction does not retrain or reevaluate the model.

Removed exact duplicate source lines:

```text
10, 22, 257, 259, 389, 393, 646, 921, 980, 1035, 1643, 1773, 1868, 2215, 2292, 2563, 2827, 3322, 3403, 3404, 3405, 3408, 3409, 3534, 3880, 3898, 4036, 4481, 4586, 4587, 4815, 4819, 5088, 5091, 5835, 5866, 6238, 6769, 8088, 8129, 8130, 8133, 8314, 8315, 8452, 8453, 8558, 8559, 8763, 8764, 9045, 9049, 9071, 9072, 9073, 9088, 9342, 9366, 9398, 10185, 10404, 10651, 10652, 10784, 11464, 11571, 11572, 11573, 11852, 11853, 11886, 12544, 13869, 14565, 16065, 16261, 16508, 17063, 17167, 18101, 18538, 18544, 18653, 18864
```

The source dataset was checked byte-for-byte against current main `206a6df0142b41549d5c5de46176e33cdd601fb8` before this correction was applied to PR #10. Its SHA-256 is `43e19e191823ce51f3405256e24e5e8628169fbc44584cc93ba81ec17b1ddc44`. Relative to PR head `11d886299a2713e3b02e4cdc8348c735bfb7815b`, this restores 18 distinct pairs removed by prefix comparison and moves the remaining conflicting-label record to quarantine.
