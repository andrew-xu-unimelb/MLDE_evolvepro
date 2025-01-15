import torch
from pathlib import Path
import os
from esm.pretrained import load_hub_workaround


def download_esm2_15B_with_regression(model_name="esm2_t48_15B_UR50D", download_dir="/path/to/custom/directory"):
    """
    Downloads the ESM2 15B model and its contact regression weights to a specified directory.
    
    Args:
        model_name (str): The name of the model to download.
        download_dir (str): The directory where the model and weights will be downloaded.
    
    Returns:
        model_data (dict): The downloaded model state dictionary.
        regression_data (dict): The downloaded regression state dictionary.
    """
    # Ensure download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Set custom torch hub directory for caching
    torch.hub.set_dir(download_dir)
    
    # URLs for model and regression weights
    model_url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    regression_url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    
    # Download model
    model_path = Path(download_dir) / f"{model_name}.pt"
    model_data = load_hub_workaround(model_url)
    print(f"Model saved at: {model_path}")
    
    # Download regression weights
    regression_path = Path(download_dir) / f"{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(regression_url)
    print(f"Regression weights saved at: {regression_path}")
    
    return model_data, regression_data
    #return regression_data

# Example usage
cache_directory = "/./vast/projects/G000448_Protein_Design/model_weights"
model_data, regression_data = download_esm2_15B_with_regression(model_name="esm2_t36_3B_UR50D", download_dir=cache_directory)
#regression_data = download_esm2_15B_with_regression(model_name="esm2_t48_15B_UR50D", download_dir=cache_directory)
