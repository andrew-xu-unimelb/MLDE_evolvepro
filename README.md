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
        - [TODO] Add explanation of intuition
        - [TODO] Use base source code to run dsm grid search (only on Jones embeddings)
        - [TODO] Visualise K-medoid clustering for PCA embeddings
        - [TODO] Read [intrinsic dimension reduction](https://huggingface.co/blog/AmelieSchreiber/intrinsic-dimension-of-proteins) article for ESM2 
        - [TTODO] Implement inrinsic dimension reduction, then perform PCA and compare the two embeddings.

**Lab stuff**
- Read transfection steps