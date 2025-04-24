import argparse
import json
import subprocess

# Variables contenant les chemins des fichiers
json_path = "chemin/vers/fichier.json"
checkpoint_path = "chemin/vers/checkpoint"
output_path = "chemin/vers/dossier"

def main(json_file, checkpoint, output_dir, test_mode):
    # Exemple d'utilisation des arguments
    print(f"Fichier JSON: {json_file}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Dossier de sortie: {output_dir}")
    print(f"Mode test: {test_mode}")
    
    # Charger le fichier JSON si nécessaire
    with open(json_file, 'r') as file:
        data = json.load(file)
        print("Contenu JSON chargé:")
        print(data)
    
    # Votre logique principale ici
    command = [
        "python", "NewArchitectures.py",
        "--json", json_file,
        "--checkpoint", checkpoint,
        "--output", output_dir
    ]
    
    # Ajouter l'argument test s'il est activé
    if test_mode:
        command.append("--test")
    
    # Exécuter le script A.py avec les arguments
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        print("Sortie de A.py :")
        print(result.stdout)
        if result.stderr:
            print("Erreurs de A.py :")
            print(result.stderr)
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script principal pour exécuter des tâches spécifiques.")
    
    # Ajouter des arguments
    parser.add_argument("--json", type=str, required=True, help="Chemin vers le fichier JSON.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Chemin vers le checkpoint.")
    parser.add_argument("--output", type=str, required=True, help="Dossier de sortie.")
    parser.add_argument("--test", action="store_true", help="Activer le mode test.")
    
    # Récupérer les arguments
    args = parser.parse_args()
    
    # Appeler la fonction principale avec les arguments
    main(args.json, args.checkpoint, args.output, args.test)
