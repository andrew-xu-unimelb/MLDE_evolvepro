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
        predict_all = True,
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

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)  # Add this for NumPy
        if hasattr(model, 'random_state'):
            model.set_params(random_state=random_seed)  # Update model's seed

    os.makedirs(output_dir, exist_ok=True)
    predictions_list = []
    metrics_list = []
    iteration_old = None
    selected_positions = set()

    labels.set_index('variant', drop=False, inplace=True)
    for iter in range(0, cycles+1):
        if iter == 0:
            non_wt_variants = labels.loc[labels['variant'] != 'WT', 'variant'].tolist()
            # first iteration: randomly select mutants and add WT
            selected_variants = np.random.choice(
                non_wt_variants,
                size=num_per_cycle, # number of mutants per round
                replace=False
            ).tolist()
            selected_variants.append('WT') # add WT back
            iteration_old = pd.DataFrame({'iteration': iter, 'variant': selected_variants})

            # Track selected positions
            if selection_method == "limit_AA":
                selected_positions.update(set(pd.Series(selected_variants).str.extract(r'([A-Za-z]\d+)')[0].dropna()))

            # add metrics data
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
            # subsequent iterations: train Random Forest and predict on unselected variants
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
            activity_binary_percentage = top_chunk['y_actual_binary'].mean()
            
            # decide how to pick the next round's variants
            df_predictions = pd.DataFrame({
                'iteration': iter,
                'variant': X_predict.index,
                'predicted_activity': y_test_predict,
            })
        

            if predict_all:
            # predict all mutations instead of just unseen ones, and save to output dataframe
                all_predictions = model.predict(embeddings)
                all_predictions_df = pd.DataFrame({
                    'iteration': iter,
                    'variant': embeddings.index.tolist(),
                    'predicted_activity': all_predictions,
                })
                # save predictions of all mutations
                predictions_list.append(all_predictions_df)
            else:
                # save predictions of only unseen mutations
                predictions_list.append(df_predictions)


            # select the top variants for next iteration
            if selection_method == "limit_AA":
                # alternative approach - limit the number of choices per AA position to 1
                df_predictions['position'] = df_predictions['variant'].str.extract(r'([A-Za-z]\d+)')[0]
                df_predictions = df_predictions[~df_predictions['position'].isin(selected_positions)]
                #top_variants = df_predictions.sort_values('predicted_activity', ascending=False).head(num_per_cycle)
                top_variants = df_predictions.loc[df_predictions.groupby('position')['predicted_activity'].idxmax()]
                top_variants = top_variants.sort_values('predicted_activity', ascending=False).head(num_per_cycle)
                selected_variants = top_variants['variant'].tolist()
                selected_positions.update(set(top_variants['position']))
            elif selection_method == "top_features":
                # get the feature importances from the trained model
                importances = model.feature_importances_
                # sort the features by importance in descending order
                feature_names = X_train.columns.tolist()
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }) 
                # Sort features by importance in descending order
                sorted_features = feature_importance.sort_values('importance', ascending=False)['feature']

                # Select only the top features (if needed, otherwise keep all)
                top_features = sorted_features[:num_per_cycle]  

                # Normalize the selected features in X_predict for fair comparison
                X_predict_normalized = X_predict[top_features].copy()
                X_predict_normalized = (X_predict_normalized - X_predict_normalized.min()) / (X_predict_normalized.max() - X_predict_normalized.min())
                
                # Compute a combined score for each variant by summing across features (equal weighting)
                X_predict_normalized['total_score'] = X_predict_normalized.sum(axis=1)  # or use mean() for average

                # Select top variants with highest aggregated score
                selected_variants = X_predict_normalized.nlargest(num_per_cycle, 'total_score').index.tolist()

                #sorted_features = feature_importance.sort_values('importance', ascending=False)['feature']
                ## for each of these top features, find the variant in X_predict (unselected variants) that has the highest value in the feature
                #selected_variants = []
                #seen = set()
                #for feature in sorted_features:
                #    if len(selected_variants) >= num_per_cycle:
                #        break
                #    top_variant = X_predict[str(feature)].idxmax()
                #    if top_variant not in seen:
                #        selected_variants.append(top_variant)
                #        seen.add(top_variant)
                #selected_variants = selected_variants[:num_per_cycle]
                ## collect these variants as the selected ones for next iteration
            else:
                selected_variants = df_predictions.nlargest(num_per_cycle, 'predicted_activity')['variant'].tolist()
            
            # update iteration_old with the selected variants
            new_iteration = pd.DataFrame({'iteration': iter, 'variant': selected_variants})
            iteration_old = pd.concat([iteration_old, new_iteration], ignore_index=True)

            # calculate metrics
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
        

    # compile results
    final_predictions_df = pd.concat(predictions_list, ignore_index=True)
    final_cycle_predictions_df = final_predictions_df[final_predictions_df["iteration"]==iter].drop(columns=["iteration"])
    metrics_df = pd.DataFrame(metrics_list)
    #if limit_AA_selection:
    #    final_predictions_df = final_predictions_df.drop(columns=['position'])
    #    final_cycle_predictions_df = final_cycle_predictions_df.drop(columns=['position'])

    # save outputs
    final_predictions_df.to_csv(os.path.join(output_dir, "predicted_activities.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    final_cycle_predictions_df.to_csv(os.path.join(output_dir, "final_cycle_predictions.csv"), index=False)
    return final_predictions_df, metrics_df, final_cycle_predictions_df 