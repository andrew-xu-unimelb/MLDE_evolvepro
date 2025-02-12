#### WEHI MLDE Summer Project


##### Setup
- Make virtual environment
```python
pip install -r requirements.txt
```


##### File structure
 - This repo is a franken-merge between [EvolvePro's repo](https://github.com/mat10d/EvolvePro) and [local Bayesian optimisation's repo](https://github.com/uber-research/TuRBO). I wanted to keep everything in one place, but as a consequence there's a bunch of code together. Here's a reference to all the files which below to each repo:
    - EvolvePro: colab, evolvepro, scripts
    - Local bayesian optimisation: turbo
    - My own scripts: MLDE_scripts
    - My own execution of scripts: MLDE_examples
- The folder 'data' contains relevant data from both EvolvePro and local Bayesian optimisation. When you create embeddings using ESM2 (or derivatives), EvolvePro's code generates quite a bit of data, so I tend to keep all of the generated data under .data/output and kept it out of the github repo. All of the data that's not excluded by gitignore came originally from EvolvePro's repo.
- Below is a description of all of the scripts I've made:
    - NOTE: because I've moved some files here from the base directory, some of the import calls in relation to other scripts may be broken. You'll have to set your own absolute os path to run them.
    - MLDE_examples/local_embedding_generation.ipynb: example code for how to generate embeddings based on EvolvePro's code locally (i.e. without Milton)
    - MLDE_scripts/dms_embedding.py: used to generate all embeddings for single AA mutants.
    - MLDE_scripts/embedding.sh: the bash script which uses dms_embedding_script.py to run in Milton
    - MLDE_scripts/dimension_reduction.py: visualisation tools for both PCA and UMAP
    - MLDE_examples/dimension_reduction_visualisation.ipynb: example code for running dimension_reduction.py
    - MLDE_scripts/esm2_model_download.py: script for downloading different ESM2 model sizes
    - MLDE_scripts/model_simulation_pipeline.py: my own custom script for running simulations with different model types, I made certain custimisations with regards to what metrics it calculates per iteration
    - MLDE_examples/random_forest_tests.ipynb: example code for executing the model_simulation_pipeline.py with RandomForests as an example, I also included methods to run it multiple times in a row to get average performance
    - MLDE_bayesian_optimisation.ipynb: my trial run using bayesian optimisation using local optimisers