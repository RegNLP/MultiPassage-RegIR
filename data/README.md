# Data

This folder contains the datasets and graph resources used in **ObliQA-MP**.

## Files
- `QADataset/`
  - `ObliQA_MultiPassage_train.json`
  - `ObliQA_MultiPassage_val.json`
  - `ObliQA_MultiPassage_test.json`  
  These contain the train/validation/test splits of the ObliQA-MP dataset.

- `graph.gpickle`  
  Regulatory knowledge graph derived from the underlying documents, used for graph-based retrieval experiments.

## Access
The dataset is provided here for research purposes.  


## Notes
- Embedding collections, indexes, and raw documents are excluded from this repository due to size and licensing restrictions.
- These can be reproduced using the scripts in `src/prep/` and `src/retrieval/`.
