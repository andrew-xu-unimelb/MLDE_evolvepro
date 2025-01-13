### Worklog:

**Admin / organisation**

- Organise files - TODO
- Upload repo to DB's repo

**Research areas**

- How to best optimise computing resources for creating embeddings
    - [TODO] Flash attention - only possible on CUDA (GPU) enabled devices 
    - [Done] Creating & tracking a log of all submited jobs on Milton

- How to optimise first round variants 
    - Currently, EvolvePro's method is to randomly pick a list of mutations for first round
    - In their source code, they also included a way to cluster sequences by medoids, and selecting sequences from clusters that the most diverse (i.e. furthest euclidean distance) 
        - I think there's room for improvement here. 
        - [Done] Use base source code to run dsm grid search (only on Jones embeddings)
        - [Done] Visualise K-medoid clustering for PCA embeddings
        - [Done] Read [intrinsic dimension reduction](https://huggingface.co/blog/AmelieSchreiber/intrinsic-dimension-of-proteins) article for ESM2 
        - [Done] Implement UMAP, then perform PCA and compare the two embeddings.
    - Optimise the cost effectiveness of number of first round variants
        - [TODO] Create a cost effectivess chart of number of first round variants to final round activity:round ratio
    - [TODO] use ESM3 to create embeddings
    - [TODO] use ESMC to create embeddings

- Surrogate objective functions
    - [TODO] Structure prediction with OmegaFold
    - [TODO] Thermostability prediction with TemStaPro
    - [TODO] Aggregation propensity with Aggrescan3D
    - [TODO] Electrostatic potential and hydrophobicity with PEP-Patch


- Visualise final evolution proteins

**Lab stuff**
- Read transfection steps