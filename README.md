
# MultiPassage-RegulatoryRAG

End-to-end retrieval (BM25 + dense + RRF) and optional Learning-to-Rank (LightGBM) for multi-passage regulatory QA.  
Everything runs from clean setup ➜ indexing ➜ retrieval ➜ fusion ➜ feature extraction ➜ LTR ➜ evaluation @10.

----------

## What’s here

-   **Prep**
    
    -   `step0_fix_json_ids.py` — normalizes IDs in QA JSON (train/val/test)
        
    -   `step1_build_graph.py` — builds a regulatory graph (Documents, Passages, NamedEntities, DefinedTerms, Cross-refs)
        
-   **Retrieval**
    
    -   BM25 via **Pyserini/Lucene**
        
    -   Dense retrieval via **SentenceTransformers + Faiss**
        
    -   Hybrid fusion (**RRF** or weighted)
        
-   **LTR *
    
    -   Feature extraction (+ doc & graph features)
        
    -   LightGBM LambdaMART training and inference
        
-   **Evaluation**
    
    -   Clean, pytrec_eval-based **@10** metrics (Recall@10, MAP@10, nDCG@10)
        

----------

## Repo structure

``` 
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ .gitignore
├─ scripts/
│  ├─ setup_env.sh # one-shot conda env + pip installs (no env.yml) │  ├─ teardown_env.sh # remove env if you want a fresh start │  └─ run_end2end_test.sh # full test split pipeline (id-fix→index→retrieve→eval@10) ├─ data/
│  ├─ QADataset/ # put raw ObliQA_MultiPassage_{train,val,test}.json here │  └─ Documents/
│     ├─ <document_jsons>.json # regulatory docs referenced by doc_manifest.py │     └─ CrossReferenceData.csv # cross-ref edges (SourceID, TargetID) ├─ outputs/ # created at runtime (gitignored) │  ├─ QADataset_fixed/
│  ├─ collections/
│  ├─ indexes/
│  ├─ runs/
│  ├─ features/
│  ├─ ltr/
│  └─ eval/
├─ src/
│  ├─ doc_manifest.py # points to documents under data/Documents/ │  ├─ prep/
│  ├─ retrieval/
│  ├─ features/
│  ├─ ltr/
│  └─ eval/
└─ docs/
   └─ results_basics.md # simple table of @10 metrics (optional) 
   ``` 


## Quickstart (macOS/Linux)

> No `requirements.txt` or `environment.yml` needed — the env is fully scripted.

```bash
# 1) from repo root bash scripts/setup_env.sh
conda activate regulatoryrag 
# 2) Java for Pyserini (macOS examples)  
export JAVA_HOME="$("/usr/libexec/java_home" -v 17)" export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib" export PATH="$JAVA_HOME/bin:$PATH" 
# 3) (optional) stability on macOS/Conda for dense encoding  
export OMP_NUM_THREADS=1 export MKL_NUM_THREADS=1 export TOKENIZERS_PARALLELISM=false  
# 4) run the end-to-end TEST pipeline 
bash scripts/run_end2end_test.sh`
``` 

**Expected last lines (example on 447 test queries):**

```bash
All test runs (evaluated on  447 queries): 
Method 			Recall@10 	MAP@10 	nDCG@10  
-------------------------------------------
rrf_bm25+dense 	0.5885  	0.4539  0.5711 
bm25 			0.5585  	0.4540  0.5703 
dense 			0.5072  	0.3747  0.4804 
[OK] Wrote summary TSV -> outputs/eval/test_at10_summary.tsv
✅ Done. See outputs/eval/test_at10_summary.tsv
```  


## Data you need

Place the raw QA and document JSONs under `data/`:

``` 
data/
├─ QADataset/
│  ├─ ObliQA_MultiPassage_train.json
│  ├─ ObliQA_MultiPassage_val.json
│  └─ ObliQA_MultiPassage_test.json
└─ Documents/
   ├─ <regulatory_document_1>.json
   ├─ <regulatory_document_2>.json
   ├─ ...
   └─ CrossReferenceData.csv` 
 ```
    
-   `src/doc_manifest.py` tells the graph builder where the document JSON files are.
    
-   `CrossReferenceData.csv` should have passage-UID columns: `SourceID,TargetID`.


----------

## End-to-end (test split)

The script **does everything**:

`bash scripts/run_end2end_test.sh` 

It runs the following steps:

1.  **Fix QA IDs** → `outputs/QADataset_fixed/ObliQA_MultiPassage_test.json`
2.  **Build regulatory graph** with Documents/Passages/NE/DT/CITES/… → `data/graph.gpickle`
3.  **Export JsonCollection** & **build Lucene BM25 index**
4.  **BM25 retrieval** → `outputs/runs/bm25_test.trec`
5.  **Dense retrieval** (SentenceTransformers + Faiss) → `outputs/runs/dense_test.trec`
6.  **RRF fusion** → `outputs/runs/rrf_bm25+dense_test.trec`
7.  **qrels from JSON** (all listed test passages are positive)
8.  **(Minimal) feature extraction** for LTR (saved for later if you want it)
9.  **Evaluate @10** with `pytrec_eval` → TSV in `outputs/eval/test_at10_summary.tsv`
    


## Learning-to-Rank (train/val/test)

When you’re ready to go beyond the demo:
```bash
`# 1) Build qrels for train/val 
python -m src.eval.json_passages_to_qrels \
  --split-json outputs/QADataset_fixed/ObliQA_MultiPassage_train.json \
  --out-qrels data/qrels_train.txt

python -m src.eval.json_passages_to_qrels \
  --split-json outputs/QADataset_fixed/ObliQA_MultiPassage_val.json \
  --out-qrels data/qrels_val.txt 
  
  # 2) Make BM25/Dense/RRF runs for train/val (analogous to test)  #    ... then extract features: 
python -m src.features.step4a_extract_features \
  --bm25-run outputs/runs/bm25_train.trec \
  --dense-run outputs/runs/dense_train.trec \
  --rrf-run   outputs/runs/rrf_bm25+dense_train.trec \
  --graph data/graph.gpickle \
  --doclen-map data/collections/bm25_jsonl/docids.tsv \
  --out-csv outputs/features/features_train.csv \
  --feature-list outputs/ltr/feature_list.txt

python -m src.features.step4a_extract_features \
  --bm25-run outputs/runs/bm25_val.trec \
  --dense-run outputs/runs/dense_val.trec \
  --rrf-run   outputs/runs/rrf_bm25+dense_val.trec \
  --graph data/graph.gpickle \
  --doclen-map data/collections/bm25_jsonl/docids.tsv \
  --out-csv outputs/features/features_val.csv \
  --feature-list outputs/ltr/feature_list.txt 
  
  # 3) Train LTR 
 python -m src.ltr.step4b_train_ltr \
  --train-csv outputs/features/features_train.csv \
  --val-csv   outputs/features/features_val.csv \
  --qrels     data/qrels_train.txt \
  --val-qrels data/qrels_val.txt \
  --feature-list outputs/ltr/feature_list.txt \
  --out-model outputs/ltr/ltr_model.txt 
  
  # 4) Apply LTR to test 
 python -m src.ltr.step4c_apply_ltr \
  --model outputs/ltr/ltr_model.txt \
  --feature-csv outputs/features/features_test.csv \
  --feature-list outputs/ltr/feature_list.txt \
  --out outputs/runs/ltr_test.trec 
  
  # 5) Re-evaluate @10 including LTR 
 python -m src.eval.eval_test_at10 \
  --qa-test outputs/QADataset_fixed/ObliQA_MultiPassage_test.json \
  --runs-dir outputs/runs \
  --pattern '*_test.trec' \
  --out-tsv outputs/eval/test_at10_summary.tsv` 
 ```
----------

## Configuration knobs

-   **BM25**: `step2b_retrieve_bm25` supports `--bm25-k1` and `--bm25-b`.
-   **Dense**: `step2c_retrieve_dense` supports `--batch`, `--model` (SentenceTransformer), etc.
-   **Fusion**: `step2d_hybrid_fuse` supports `--method rrf|weighted`, `--alpha`, `--norm`, `--k`, `--rrf-c`.
-   **Features**: list saved to `outputs/ltr/feature_list.txt` (used by LTR).
-   **LTR**: LightGBM params are set inside `step4b_train_ltr.py` (tweak there if needed).
    

----------

## Java & environment notes

-   **Java (JDK 17 or 21)** required for Pyserini/Lucene.
    
    -   macOS (Temurin 17):
        ```bash
        `export JAVA_HOME="$("/usr/libexec/java_home" -v 17)" 
        export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib" 
        export PATH="$JAVA_HOME/bin:$PATH" ```
-   **Conda env** is created by `scripts/setup_env.sh` (no `requirements.txt` / `environment.yml` needed).
-   We install `pyserini==0.22.0` **without deps** and bring in only what we actually use, avoiding heavy/fragile packages.
    

----------

## Troubleshooting

-   **`ModuleNotFoundError: pyserini`**  
    You likely aren’t in the conda env:
    
    `conda activate regulatoryrag
    python -c "import pyserini; print('ok')"` 
    
-   **`Unable to find libjvm` / `dlopen ... libjvm.dylib`**  
    Export Java vars in the current shell (see Java section), then retry.
    
-   **`compiled by a more recent Java ... class file 65.0`**  
    Your Java is older than the Pyserini Lucene classes. Use JDK **21** (or keep 17 but ensure `pyserini==0.22.0` is installed in this env):
    
    `export JAVA_HOME="$("/usr/libexec/java_home" -v 21)" export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib" export PATH="$JAVA_HOME/bin:$PATH"` 
    
-   **Segfaults / very slow dense encoding on macOS/Conda**  
    Use the stability flags:
    
    `export OMP_NUM_THREADS=1 export MKL_NUM_THREADS=1 export TOKENIZERS_PARALLELISM=false` 
    
-   **Permission denied on scripts**  
    `chmod +x scripts/*.sh`
    

If something else pops, check the last error lines and file/line number — most steps are clearly logged.

----------

## Results & reproducibility

-   The end-to-end test script prints a small table and writes `outputs/eval/test_at10_summary.tsv`.
    
-   For fully reproducible baselines, keep the conda env, Java version, and script defaults unchanged.
    

----------

## Citing

```bibtex
@misc{
}
```
    

----------

## License

-   See **`LICENSE`** (MIT).
    

----------

## Acknowledgements

-   **Pyserini/Lucene**, **Faiss**, **SentenceTransformers**, **LightGBM**, **pytrec_eval** — thank you to the authors/maintainers.
    

----------

## Contact

Issues / questions → open a GitHub Issue on this repo.
