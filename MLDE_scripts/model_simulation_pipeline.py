import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
import random


def model_simulation(
        embeddings,
        labels,
        output_dir,
        predict_all = False,
        activity = "activity",
        cycles = 10,
        num_per_cycle = 10,
        model = RandomForestRegressor(n_estimators=100, criterion='friedman_mse', max_depth=None, min_samples_split=2,
                                        min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0,
                                        max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                        n_jobs=None, random_state=1, verbose=0, warm_start=False, ccp_alpha=0.0,
                                        max_samples=None),
        random_seed = None,
        selection_method = None,
    ):
    """
    Run a simulation loop that trains any model on selected variants,
    predicts activity on unseen variants, and selects new variants for the next cycle.
    
    Parameters:
        embeddings (pd.DataFrame): DataFrame with variant embeddings. The index should be variant names.
        labels (pd.DataFrame): DataFrame with labels. Must include a 'variant' column and an activity column.
        output_dir (str): Directory to save the output CSV files.
        predict_all (bool): Whether to predict for all mutations (True) or only unseen ones (False) per iteration.
        activity (str): The column name for the activity label.
        cycles (int): Number of cycles (iterations) to run.
        num_per_cycle (int): Number of variants to select per cycle.
        model (object): A scikit-learn model (default is a RandomForestRegressor).
        random_seed (int): Seed for reproducibility.
        selection_method (str): Method for variant selection. Options include "limit_AA", or None.
    
    Returns:
        final_predictions_df (pd.DataFrame): DataFrame with predictions for each cycle.
        metrics_df (pd.DataFrame): DataFrame with computed metrics for each cycle.
        final_cycle_predictions_df (pd.DataFrame): Predictions from the final cycle only.
    """    
    # ------------------------- Setup and Initialization -------------------------
    # set random seed for both initialisation and model attributes
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)  # Add this for NumPy
        if hasattr(model, 'random_state'):
            model.set_params(random_state=random_seed)  # Update model's seed

    os.makedirs(output_dir, exist_ok=True)

    # initialise lists to store predictions and metrics for each iteration
    predictions_list = []
    metrics_list = []

    # dataframe to store variantes selected over iterations
    iteration_old = None

    # if selection method limits amino acid (AA) positions, keep trach of positions already selected
    selected_positions = set()

    # set the index of labels DataFrame to the variant names for easier lookup
    labels.set_index('variant', drop=False, inplace=True)

    # ------------------------------ Simulation Loop ------------------------------
    for iter in range(0, cycles+1):
        if iter == 0:
            # -------------------- Initial Cycle --------------------
            # Select random mutants (excluding WT) and add the WT variant
            non_wt_variants = labels.loc[labels['variant'] != 'WT', 'variant'].tolist()
            # first iteration: randomly select mutants and add WT
            selected_variants = np.random.choice(
                non_wt_variants,
                size=num_per_cycle, # number of mutants per round
                replace=False
            ).tolist()
            selected_variants.append('WT') # add WT back
            iteration_old = pd.DataFrame({'iteration': iter, 'variant': selected_variants})

            # For the "limit_AA" selection method, extract and track positions (e.g., "A123")
            if selection_method == "limit_AA":
                selected_positions.update(set(pd.Series(selected_variants).str.extract(r'([A-Za-z]\d+)')[0].dropna()))

            # save initial metrics (most values are not applicable for the first cycle)
            metrics_iter0 = {
                'iteration': iter,
                'next_iter_variants': ' '.join(selected_variants),
                'train_error': None,
                'test_error': None,
                'median_activity': None,
                'top_activity': None,
                'activity_binary_percentage': None,
                'top_variant': None,
                'top_final_round_variants': None
            }
            metrics_list.append(metrics_iter0)

        else:
            # -------------------- Subsequent Cycles --------------------
            # get variants selected so far
            selected_so_far = iteration_old['variant'].tolist()
            # select embeddings and labels based on variant names
            selected_mask = embeddings.index.isin(selected_so_far)

            # split data into training and prediction sets
            X_train = embeddings[selected_mask]
            y_train = labels.loc[selected_mask, activity]

            X_predict = embeddings[~selected_mask]
            y_test_actual = labels.loc[~selected_mask, activity]

            # train random forest model
            model.fit(X_train, y_train)

            # make predictions for unseen mutations - used for next step calculations
            y_train_predict = model.predict(X_train)
            y_test_predict = model.predict(X_predict)
            
            # Create DataFrames with predictions and actual values for train and test sets 
            df_train = pd.DataFrame({
                'variant': X_train.index,
                'y_pred': y_train_predict,
                'y_actual': y_train,
                'y_actual_binary': labels.loc[X_train.index, 'activity_binary']
            })
            df_test = pd.DataFrame({
                'variant': X_predict.index,
                'y_pred': y_test_predict,
                'y_actual': y_test_actual,
                'y_actual_binary': labels.loc[X_predict.index, 'activity_binary']
            })

            # combine train and test and sort by predicted activity
            df_all = pd.concat([df_train, df_test], ignore_index=True)
            df_sorted_all = df_all.sort_values('y_pred', ascending=False).reset_index(drop=True)

            # compute metrics
            top_k = num_per_cycle
            top_chunk = df_sorted_all.iloc[:top_k]
            #top_chunk = df_train.iloc[:top_k]
            median_activity = top_chunk['y_actual'].median()
            top_activity = top_chunk['y_actual'].max()
            top_variant = top_chunk.loc[top_chunk['y_actual'].idxmax(), 'variant']
            top_final_round_variants = ' '.join(top_chunk['variant'].tolist())
            activity_binary_percentage = top_chunk['y_actual_binary'].sum() / top_k
            #activity_binary_percentage = top_chunk['y_actual_binary'].mean()
            
            # Prepare a DataFrame of predictions for the unseen variants for the current cycle
            df_predictions = pd.DataFrame({
                'iteration': iter,
                'variant': X_predict.index,
                'predicted_activity': y_test_predict,
            })
        
           # Save predictions: either for all variants or only for unseen variants
            if predict_all:
                all_predictions = model.predict(embeddings)
                all_predictions_df = pd.DataFrame({
                    'iteration': iter,
                    'variant': embeddings.index.tolist(),
                    'predicted_activity': all_predictions,
                })
                predictions_list.append(all_predictions_df)
            else:
                predictions_list.append(df_predictions)


            # ------------------- Selection of Next Variants -------------------
            if selection_method == "limit_AA":
                # For each variant, extract its position (e.g., "A123") and filter out positions already used
                df_predictions['position'] = df_predictions['variant'].str.extract(r'([A-Za-z]\d+)')[0]
                df_predictions = df_predictions[~df_predictions['position'].isin(selected_positions)]

                # For each remaining position, keep the variant with the highest predicted activity
                top_variants = df_predictions.loc[df_predictions.groupby('position')['predicted_activity'].idxmax()]
                # Then select the top variants across positions
                top_variants = top_variants.sort_values('predicted_activity', ascending=False).head(num_per_cycle)
                selected_variants = top_variants['variant'].tolist()
            
                # Update the set of positions to avoid re-selecting the same positions
                selected_positions.update(set(top_variants['position']))

            else:
                # Default selection: pick the top variants based on predicted activity
                selected_variants = df_predictions.nlargest(num_per_cycle, 'predicted_activity')['variant'].tolist()
            
            # update iteration_old with the selected variants
            new_iteration = pd.DataFrame({'iteration': iter, 'variant': selected_variants})
            iteration_old = pd.concat([iteration_old, new_iteration], ignore_index=True)

            # ------------------- Compute and Save Metrics -------------------
            train_error = mean_squared_error(y_train, y_train_predict)
            test_error = mean_squared_error(y_test_actual, y_test_predict)

            next_iter_variants = ' '.join(selected_variants)

            metrics = {
                'iteration': iter,
                'next_iter_variants': next_iter_variants,
                'train_error': train_error,
                'test_error': test_error,
                'median_activity': median_activity,
                'top_activity': top_activity,
                'activity_binary_percentage': activity_binary_percentage,
                'top_variant': top_variant,
                'top_final_round_variants': top_final_round_variants
            }

            metrics_list.append(metrics)
        

    # --------------------- Compile and Save Results ---------------------
    # Concatenate all predictions collected over cycles
    final_predictions_df = pd.concat(predictions_list, ignore_index=True)

    # Extract predictions from the final cycle (dropping the iteration column)
    final_cycle_predictions_df = final_predictions_df[final_predictions_df["iteration"]==iter].drop(columns=["iteration"])

    # Create a DataFrame from the metrics collected during the simulation
    metrics_df = pd.DataFrame(metrics_list)

    # save outputs
    final_predictions_df.to_csv(os.path.join(output_dir, "predicted_activities.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    final_cycle_predictions_df.to_csv(os.path.join(output_dir, "final_cycle_predictions.csv"), index=False)

    return final_predictions_df, metrics_df, final_cycle_predictions_df 