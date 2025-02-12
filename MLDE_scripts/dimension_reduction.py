import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import ipywidgets as widgets
from ipywidgets import interact

class Create_PCA:
    def __init__(self, embedding_path, labels_path = None, label_name = None, n_dims = 10):
        self.embedding = pd.read_csv(embedding_path)
        # making sure labels are sorted based on variants corresponding to each entry in embedding table
        labels = pd.read_csv(labels_path)
        sorted_labels = labels.set_index('variant')
        self.labels = sorted_labels.loc[self.embedding.iloc[:, 0]].reset_index()
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

        pca_df['activity'] = self.labels['activity']
        plot = sns.pairplot(
            pca_df, hue='activity',
            vars=pca_df.columns.drop('activity'),
            palette=palette
        )
        return plot
    
    def visualise_labels_3d(self, palette='viridis', x_dim = 0, y_dim = 1):
        pca_df = self._create()
        # setting dimensions to plot
        x = pca_df.iloc[:, x_dim].values
        y = pca_df.iloc[:, y_dim].values
        z = self.labels['activity'].values

        # interpolating values between discrete points
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 50),
            np.linspace(y.min(), y.max(), 50)
        )
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        # plotting
        sns.set_theme(style='whitegrid')
        def _update_plot(elev=40, azim=45):
            plt.close('all')
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            # plot surface
            sc = ax.plot_surface(grid_x, grid_y, grid_z, cmap=palette, edgecolor='none', alpha=0.8)
            ax.view_init(elev=elev, azim=azim)
            fig.colorbar(sc, ax=ax, shrink=0.5)
            return fig

        interactive_plot = interact(
            _update_plot,
            elev=widgets.IntSlider(min=0, max=90, step=5, value=30),
            azim=widgets.IntSlider(min=0, max=360, step=10, value=45)
        )

        return interactive_plot
        


        
class Create_UMAP:
    def __init__(self, embedding_path, labels_path = None, label_name = None, n_components = 2, n_neighbours = 16, min_dist = 0.1):
        self.embedding = pd.read_csv(embedding_path)
        # making sure labels are sorted based on variants corresponding to each entry in embedding table
        labels = pd.read_csv(labels_path)
        sorted_labels = labels.set_index('variant')
        self.labels = sorted_labels.loc[self.embedding.iloc[:, 0]].reset_index()
        self.label_name = label_name
        self.n_components = n_components
        self.n_neighbours = n_neighbours
        self.min_dist = min_dist
    
    def _create(self):
        data = self.embedding.iloc[:, 1:] # remove labels
        dr = umap.UMAP(n_components=self.n_components, n_neighbors=self.n_neighbours, min_dist=self.min_dist)
        umap_embeddings = dr.fit_transform(data)
        umap_df = pd.DataFrame(umap_embeddings, columns=[f'UMAP{i+1}' for i in range(umap_embeddings.shape[1])]) 
        return umap_df
    
    def visualise_base(self):
        data = self._create()
        return sns.pairplot(data)

    def visualise_labels(self, palette='viridis'):
        umap_df = self._create()

        umap_df['activity'] = self.labels['activity']
        plot = sns.pairplot(
            umap_df, hue='activity',
            vars=umap_df.columns.drop('activity'),
            palette=palette
        )
        return plot
    
    def visualise_labels_3d(self, palette='viridis', x_dim = 0, y_dim = 1):
        umap_df = self._create()
        # setting dimensions to plot
        x = umap_df.iloc[:, x_dim].values
        y = umap_df.iloc[:, y_dim].values
        z = self.labels['activity'].values

        # interpolating values between discrete points
        grid_x, grid_y = np.meshgrid(
            np.linspace(x.min(), x.max(), 50),
            np.linspace(y.min(), y.max(), 50)
        )
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

        # plotting
        sns.set_theme(style='whitegrid')
        def _update_plot(elev=40, azim=45):
            plt.close('all')
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            # plot surface
            sc = ax.plot_surface(grid_x, grid_y, grid_z, cmap=palette, edgecolor='none', alpha=0.8)
            ax.view_init(elev=elev, azim=azim)
            fig.colorbar(sc, ax=ax, shrink=0.5)
            return fig

        interactive_plot = interact(
            _update_plot,
            elev=widgets.IntSlider(min=0, max=90, step=5, value=30),
            azim=widgets.IntSlider(min=0, max=360, step=10, value=45)
        )

        return interactive_plot
    

    
    
