import numpy as np
import matplotlib.pyplot as plt
import weightwatcher as ww
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import torch
import torch.nn.functional as F


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

def visualize_weights(model,savefig=None):
        
    # Créer le WeightWatcher
    watcher = ww.WeightWatcher(model=model)
    
    # Analyser le modèle

    details = watcher.analyze()
    
    # Sauvegarder les résultats
    plt_alpha_ww(details, savefig=savefig)


def ww_model(model, savefig=None):
    alpha_values=[]
    color=[]
    if model.conf['method'] == 'middle_fusion':
        watcher = ww.WeightWatcher(model=model.fusion_en)
        details = watcher.analyze()
        alpha_values.extend(details.alpha.tolist())
        color.extend(['blue'] * len(details.alpha))
    
    watcher = ww.WeightWatcher(model=model.net)
    details = watcher.analyze()
    alpha_values.extend(details.alpha.tolist())
    color.extend(['green'] * len(details.alpha))

    plt.subplots(figsize=(12, 6))  # Un seul rang, deux colonnes

    # Graphique en barres sur l'axe de droite    
    plt.bar(range(len(alpha_values)), alpha_values, color=color)
    plt.xlabel("Couches")
    plt.ylabel("Alpha (Weight Watcher)")
    plt.title("Qualité des couches par alpha")
    
    plt.xticks(range(len(alpha_values)), rotation=45, ha='right')
    plt.legend(handles=[
        Patch(color='blue', label='Middle Fusion'),
        Patch(color='green', label='Unet'),
        Line2D([0], [0], color="r", linestyle="--", label="over-fitted (α < 2)"),
        Line2D([0], [0], color="orange", linestyle="--", label="Under fitted (α > 6)")
    ])

    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    else:
        plt.show()




def calculate_tensor_gradient(image_tensor):
    # Conserver tous les canaux de l'image
    gray_tensor = image_tensor

    # Définir les noyaux de Sobel pour les gradients en x et y
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_x = sobel_x.to(image_tensor.device)  # Déplacer le noyau sur le même appareil que l'image
    sobel_x = sobel_x.expand(gray_tensor.size(1), 1, -1, -1)  # Adapter pour le nombre de canaux

    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = sobel_y.to(image_tensor.device)  # Déplacer le noyau sur le même appareil que l'image
    sobel_y = sobel_y.expand(gray_tensor.size(1), 1, -1, -1)  # Adapter pour le nombre de canaux

    # Calculer les gradients en x et y pour chaque canal
    grad_x = F.conv2d(gray_tensor, sobel_x, padding=1, groups=gray_tensor.size(1))
    grad_y = F.conv2d(gray_tensor, sobel_y, padding=1, groups=gray_tensor.size(1))

    # Calculer la magnitude du gradient pour chaque canal
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)

    # Normaliser la magnitude du gradient pour chaque canal
    # Nous allons normaliser chaque canal séparément
    min_vals = grad_magnitude.min()
    max_vals = grad_magnitude.max()

    # Éviter la division par zéro
    epsilon = 1e-8
    grad_magnitude_normalized = (grad_magnitude - min_vals) / (max_vals-min_vals + epsilon)

    return grad_magnitude_normalized