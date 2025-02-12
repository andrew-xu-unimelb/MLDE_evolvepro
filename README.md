# WEHI MLDE Summer Project


## Setup
- Make virtual environment, then simply:
```python
pip install -r requirements.txt
```


## **Repository Overview**

This repository is a **franken-merge** between two main projects:  
[EvolvePro](https://github.com/mat10d/EvolvePro) and [Local Bayesian Optimisation](https://github.com/uber-research/TuRBO).  
To keep everything in one place, their codes have been merged, leading to a mix of scripts from different sources.

### **File & Folder Structure**
| Folder/File | Description |
|------------|-------------|
| **EvolvePro** | `colab`, `evolvepro`, `scripts` |
| **Local Bayesian Optimisation** | `turbo` |
| **My Own Scripts** | `MLDE_scripts` |
| **My Execution of Scripts** | `MLDE_examples` |

#### **Data Management**
- The **data/** folder contains relevant data from both **EvolvePro** and **Local Bayesian Optimisation**.
- When generating embeddings using **ESM2 (or derivatives)**, EvolvePro's code produces large datasets.  
  → These are stored under `.data/output` and **excluded from GitHub**.  
- Any **data not excluded by `.gitignore`** originates from the **EvolvePro** repository.

---

### **Custom Scripts & Examples**
⚠️ **Note**:  
Since some files have been **moved** from their original locations, some **import paths may be broken**.  
You'll need to **set absolute OS paths** to run them correctly.

#### **MLDE Examples (`MLDE_examples/`)**
| File | Description |
|------|------------|
| `local_embedding_generation.ipynb` | Example of generating embeddings locally (without Milton) using EvolvePro's code. |
| `dimension_reduction_visualisation.ipynb` | Example of running `dimension_reduction.py` for PCA & UMAP visualization. |
| `random_forest_tests.ipynb` | Example execution of `model_simulation_pipeline.py` with Random Forests, including performance averaging over multiple runs. |
| `bayesian_optimisation.ipynb` | Trial run using Bayesian Optimisation with local optimisers. |

#### **MLDE Scripts (`MLDE_scripts/`)**
| File | Description |
|------|------------|
| `dms_embedding.py` | Generates embeddings for **single amino acid mutants**. |
| `embedding.sh` | Bash script to execute `dms_embedding.py` on **Milton**. |
| `dimension_reduction.py` | Provides **PCA & UMAP** visualization tools. |
| `esm2_model_download.py` | Downloads different **ESM2 model sizes**. |
| `model_simulation_pipeline.py` | Custom script for running **simulations with different models**. Includes additional custom metrics per iteration. |

---
