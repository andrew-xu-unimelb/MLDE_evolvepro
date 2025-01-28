import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


class Create_PCA:
    def __init__(self, embedding_path, labels_path = None, label_name = None, n_dims = 10):
        self.embedding = pd.read_csv(embedding_path)
        self.labels = pd.read_csv(labels_path)
        self.label_name = label_name
        self.n_dims = n_dims
    
    def _create(self):
        data = self.embedding.iloc[:, 1:] # remove labels
        pca = PCA(n_components=self.n_dims)
        pca_embeddings = pca.fit_transform(data)
        pca_df = pd.DataFrame(pca_embeddings, columns=[f'PC{i+1}' for i in range(pca_embeddings.shape[1])])
        return pca_df
    
    def visualise_base(self):
        data = self._create()
        return sns.pairplot(data)

    def visualise_labels(self, palette='viridis'):
        pca_df = self._create()
        data = self.embedding
        data['activity'] = 0.0
        for index, row in data.iterrows():
            mutation = row.iloc[0]
            activity = self.labels.loc[self.labels['variant'] == mutation, self.label_name].iloc[0]
            data.loc[index, 'activity'] = activity

        pca_df['activity'] = data['activity']
        plot = sns.pairplot(
            pca_df, hue='activity',
            vars=pca_df.columns.drop('activity'),
            palette=palette
        )
        return plot
        


        
    

    
    
