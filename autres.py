import numpy as np
import matplotlib.pyplot as plt
import torch
from NewArchitectures import NewArchitectures
import weightwatcher as ww
import glob,os

def comparaison_prediction(model_path, batch_idx):
    device="cuda"
    model=NewArchitectures.load_from_checkpoint(model_path)
    dl=model.test_dataloader()
    model = model.to(device)
   
   # Iterate through multiple batches
    for batch_idx, batch in enumerate(dl):
        if batch_idx==batch_idx:
            inputs, targets, _ = batch
            inputs = [inp.to(device) for inp in inputs]
            outputs = model(inputs)

            for img_idx in range(len(inputs)):
            # Visualization for each batch
            # Calculate and print L1 loss for the batch
                l1 = np.mean(np.abs(targets[img_idx, 0].cpu().detach().numpy() - outputs[img_idx, 0].cpu().detach().numpy()))
            print(f"Batch {batch_idx} image {img_idx} L1 Loss: {l1}")
            fig, ax = plt.subplots(1, 4, figsize=(15, 5))
            im0 = ax[0].imshow(outputs[img_idx, 0].cpu().detach().numpy(), cmap='viridis')
            ax[0].set_title("Output")
            im1 = ax[1].imshow(targets[img_idx, 0].cpu().detach().numpy(), cmap='viridis')
            ax[1].set_title("Target")
            im2 = ax[2].imshow(np.abs(targets[img_idx, 0].cpu().detach().numpy() - outputs[img_idx, 0].cpu().detach().numpy()))
            ax[2].set_title("Error")
            cbar = fig.colorbar(im1, ax=[ax[1], ax[2]], orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Color Intensity')
            im3 = ax[3].imshow(inputs[-1][img_idx].argmax(dim=0).cpu().detach().numpy(),cmap='tab10')
            ax[3].set_title("Classes")
            plt.show()

            break



def plt_alpha_ww(details, savefig=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Un seul rang, deux colonnes

    # Histogramme sur l'axe de gauche
    details.alpha.plot.hist(bins=20, ax=axes[0], title="")
    axes[0].set_xlabel("Weights Watcher Layer Quality Metrics Alpha")
    axes[0].axvline(x=2, color="r")
    axes[0].axvline(x=6, color="orange")

    # Graphique en barres sur l'axe de droite
    layer_names = details.index  # ou details.layer_id selon la version
    alpha_values = details.alpha
    axes[1].bar(range(len(alpha_values)), alpha_values)
    axes[1].set_xlabel("Couches")
    axes[1].set_ylabel("Alpha (Weight Watcher)")
    axes[1].set_title("Qualité des couches par alpha")
    axes[1].axhline(y=2, color="r", linestyle="--", label="Surentraîné (α < 2)")
    axes[1].axhline(y=6, color="orange", linestyle="--", label="Sous-entraîné (α > 6)")
    axes[1].set_xticks(range(len(alpha_values)))
    axes[1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[1].legend()

    plt.tight_layout()

    # Enregistrement du graphique si un chemin est fourni
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')  # bbox_inches='tight' évite les découpages indésirables

    plt.show()

    # Identifier les couches problématiques
    problematic_layers = details[(details.alpha < 2) | (details.alpha > 6)]

    print("Couches surentraînées (α < 2):")
    for idx, row in problematic_layers[problematic_layers.alpha < 2].iterrows():
        print(f"  {idx}: α = {row.alpha:.3f}")

    print("\nCouches sous-entraînées (α > 6):")
    for idx, row in problematic_layers[problematic_layers.alpha > 6].iterrows():
        print(f"  {idx}: α = {row.alpha:.3f}")

def visualize_weights(model_path,savefig=None):
    model = NewArchitectures.load_from_checkpoint(model_path)
        
    # Créer le WeightWatcher
    watcher = ww.WeightWatcher(model=model)
    
    # Analyser le modèle
    print(f"Analyse de {model_path}")
    details = watcher.analyze()
    
    # Sauvegarder les résultats
    plt_alpha_ww(details, savefig=savefig)

        
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