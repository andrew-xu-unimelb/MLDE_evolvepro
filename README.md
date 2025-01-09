### TODO:

- How to best optimise computing resources for creating embeddings
    - Flash attention - only possible on CUDA (GPU) enabled devices - TODO
    - Creating & tracking a log of all submited jobs on Milton - Done

- How to optimise first round variants 
    - Currently, EvolvePro's method is to randomly pick a list of mutations for first round
    - In their source code, they also included a way to cluster sequences by medoids, and selecting sequences from clusters that the most diverse (i.e. furthest euclidean distance) 
        - I think there's room for improvement here. 
        - Add explanation - TODO
