#!/bin/bash
# Configuration values for SLURM job submission.
# One leading hash ahead of the word SBATCH is not a comment, but two are.
#SBATCH --time=48:00:00 
##SBATCH -x node[110]
#SBATCH --job-name=esm2_15B_full_grid_search
#SBATCH -n 1 
#SBATCH -N 1   
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1  
##SBATCH --constraint=high-capacity    
#SBATCH --mem=20gb  
#SBATCH --output /orcd/archive/abugoot/001/Projects/Matteo/Github/directed_evolution/grid_search/out/esm2_15B_full_grid_search-%j.out 

source ~/.bashrc
conda activate embeddings

names=("brenan")
datasets=()

for name in "${names[@]}"; do
    datasets+=("${name}_esm2_t48_15B_UR50D")
done

experiment_name="esm2_15B_full_grid_search"
num_simulations=2
num_iterations=3
measured_var=("fitness" "fitness_scaled")
learning_strategies=("random" "top5bottom5")
num_mutants_per_round=16
num_final_round_mutants=16
first_round_strategies=("random" "diverse_medoids")
embedding_types=("embeddings" "embeddings_norm" )
regression_types=("ridge" "lasso")
file_type="csvs"
embedding_type_pt="average"

# Function to run grid search for a given dataset
function run_grid_search() {
    dataset_name=$1

    echo "Running ${dataset_name} dataset:" > out/${dataset_name}-${embedding_type_pt}-esm2_15B_10_rounds.out
    python3 -u grid_search.py \
        --dataset_name ${dataset_name} \
        --experiment_name ${experiment_name} \
        --base_path ../extract/esm/results_means \
        --num_simulations ${num_simulations} \
        --num_iterations ${num_iterations} \
        --measured_var ${measured_var[*]} \
        --learning_strategies ${learning_strategies[*]} \
        --num_mutants_per_round ${num_mutants_per_round} \
        --num_final_round_mutants ${num_final_round_mutants} \
        --first_round_strategies ${first_round_strategies[*]} \
        --embedding_types ${embedding_types[*]} \
        --regression_types ${regression_types[*]} \
        --file_type ${file_type} \
        --embeddings_type_pt ${embedding_type_pt} \
        >> out/${dataset_name}-${embedding_type_pt}-esm2_15B_10_rounds.out
    echo "Done running ${dataset_name} dataset:" >> out/${dataset_name}-${embedding_type_pt}-esm2_15B_10_rounds.out
}

# Loop over datasets
for dataset in "${datasets[@]}"; do
    run_grid_search ${dataset}
done