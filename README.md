# MultiPassage-RegIR


# LTR Run Filename Legend

---

## Filename Pattern

```
ltr_k{K}_ns{on|off}_nl{LEAVES}_lr{LR}_mdl{MINLEAF}[_ce{TOPN}_w{WEIGHT}]_test.trec
```

### Example
```
ltr_k100_nsoff_nl63_lr0.07_mdl50_test.trec
ltr_k200_nson_nl127_lr0.05_mdl100_ce50_w02_test.trec
```

---

## Parameter Tokens

| Token | Meaning | Source / Where Set | Notes |
|---|---|---|---|
| `ltr_` | LightGBM Learning-to-Rank run | auto prefix in `run_ablation.sh` | Distinguishes LTR runs from other baselines |
| `k{K}` | **Top-K per input run** used in 4a (feature extraction) to build the union candidate set | `TOPK_LIST` in `run_ablation.sh` (used by `step4a_extract_features.py --k`) | Reflects **training/validation** candidate generation. Apply (4c) may use a fixed k (e.g., 200); the `k` in the filename still refers to the train/val features |
| `ns{on|off}` | Neighbor-semantic features enabled during 4a/4b | `NEIGHBOR_SIM_LIST` in `run_ablation.sh` (passed to 4a `--neighbor-sim`) | `nson` = neighbor features **ON** at train time; `nsoff` = **OFF**. If you do not mirror neighbor features in 4c, `nson`-trained models can underperform at apply time |
| `nl{LEAVES}` | LightGBM `num_leaves` (tree width) | `NUM_LEAVES_LIST` in `run_ablation.sh` (passed to 4b `--num-leaves`) | Larger = more expressive leaves per tree |
| `lr{LR}` | LightGBM learning rate | `LEARNING_RATE_LIST` in `run_ablation.sh` (passed to 4b `--learning-rate`) | Typical values: 0.05–0.10 |
| `mdl{MINLEAF}` | LightGBM `min_data_in_leaf` | `MIN_DATA_LEAF_LIST` in `run_ablation.sh` (passed to 4b `--min-data-in-leaf`) | Regularization knob; higher = more conservative trees |
| `_ce{TOPN}` | (Optional) Cross-Encoder second pass: re-score top-N candidates AFTER LTR | `CE_TOPN_LIST` in `run_ablation.sh` (passed to 4c `--ce-topn`) | Only present if a CE model is used in 4c |
| `_w{WEIGHT}` | (Optional) Weight for CE fusion | `CE_WEIGHT_LIST` in `run_ablation.sh` (passed to 4c `--ce-weight`) | Final score = `(1 - w) * LTR + w * CE` |
| `_test.trec` | TREC-format run file for the **test** split | `step4c_apply_ltr.py --out-run` | This is the file you evaluate with `pytrec_eval` / `trec_eval` |

---

## Older / Baseline Names You May See

| Filename | Meaning |
|---|---|
| `ltr_lgbm_allfeat_test.trec` | Older all-features LTR run (legacy naming; no hyperparam tag) |
| `ltr_lgbm_allfeat_tuned_mdl100_test.trec` | Same as above with `min_data_in_leaf=100` |
| `..._ce_test.trec` | Same as above but with CE fusion applied in 4c |

---

## Mapping to Code & Artifacts

- **Model file** (saved by 4b):  
  `outputs/ltr/ltr_k{K}_ns{on|off}_nl{LEAVES}_lr{LR}_mdl{MINLEAF}.txt`

- **Per-model feature list** (saved by 4b):  
  `outputs/ltr/feature_list_ltr_k{K}_ns{on|off}_nl{LEAVES}_lr{LR}_mdl{MINLEAF}.txt`  
  Used by 4c to ensure the exact feature column order.

- **Run file** (written by 4c):  
  `outputs/runs/ltr_k{K}_ns{on|off}_nl{LEAVES}_lr{LR}_mdl{MINLEAF}[_ce{TOPN}_w{WEIGHT}]_test.trec`

---

## Notes & Tips

1. **Train/Test Feature Parity:**  
   Ensure `TEST_RUNS` includes the same base runs as train/val (e.g., `bm25`, `ft_e5`, `rrf`) so 4c has the same features the model was trained on.

2. **IDF Parity:**  
   4c should compute `idf_overlap_sum` using the same formula as 4a (you added this patch) to avoid subtle drift.

3. **Neighbor Features:**  
   If models are trained with `ns=on`, enable the same neighbor feature computation in 4c (pass `--neighbor-sim` and encoder settings). If not mirrored, prefer `nsoff` for clean comparisons.

4. **Cross-Encoder Weighting:**  
   Start small (e.g., `w=0.05–0.2`). Validate that CE improves nDCG@10; if it degrades, reduce `w` or disable CE.

5. **k in Filename vs Apply k:**  
   The `k{K}` token reflects **training/validation** candidate generation. The apply stage can use its own `--k`; document that separately if needed.

---

## Quick Example Walk-through

- **Filename:** `ltr_k200_nsoff_nl127_lr0.05_mdl100_ce50_w02_test.trec`  
  - 4a used union top-200 per run; neighbor features OFF.
  - 4b trained LightGBM with `num_leaves=127`, `lr=0.05`, `min_data_in_leaf=100`.
  - 4c rescored the **top-50** LTR docs with a CE, fused with **w=0.2**.
  - Output is a TREC run over the test split.

