import numpy as np
import matplotlib.pyplot as plt
import torch
from NewArchitectures import NewArchitectures

def comparaison_prediction(prediction, target):
    # Ensure inputs are numpy arrays and detached from computation graph
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Target image
    im0 = axs[0].imshow(target[0, 0], cmap='viridis')
    axs[0].set_title("Image cible")
    plt.colorbar(im0, ax=axs[0])
    
    # Prediction image (using target's min and max for consistent color scaling)
    im1 = axs[1].imshow(prediction[0, 0], cmap='viridis', 
                        vmin=target[0, 0].min(), vmax=target[0, 0].max())
    axs[1].set_title("Prédiction")
    plt.colorbar(im1, ax=axs[1])
    
    # Error image
    erreur = np.abs(target[0, 0] - prediction[0, 0])
    im2 = axs[2].imshow(erreur, cmap='viridis')
    axs[2].set_title("Erreur")
    plt.colorbar(im2, ax=axs[2])
    
    # Save and print mean error
    plt.savefig('comparison_plot.png')
    print(f"Mean Absolute Error: {np.mean(erreur)}")
    plt.close(fig)  # Close the figure to free up memory

def main():
    # Load the model from checkpoint
    model = NewArchitectures.load_from_checkpoint("checkpoints/dem_sar/version_0/checkpoints/last.ckpt")
    model.eval()
    dl= model.test_dataloader()
    # Get a batch of data
    inputs, targets, folder = list(dl)[0]
    print(np.shape(inputs[0]))
    #
    comparaison_prediction()

if __name__ == "__main__":
    main()