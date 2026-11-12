import os
import pandas as pd

import core.config as config
dossier_dataset = config.RAW_DATASET_DIR

print("🔍 ANALYSE DES COLONNES DE CHAQUE FICHIER :\n")

# Boucle sur tous les fichiers du dossier
for fichier in os.listdir(dossier_dataset):
    if fichier.endswith('.csv'):
        chemin_complet = os.path.join(dossier_dataset, fichier)
        try:
            # On lit juste les 2 premières lignes pour aller vite
            df_temp = pd.read_csv(chemin_complet, nrows=2)
            print(f"📄 Fichier : {fichier}")
            print(f"   📊 Colonnes trouvées : {list(df_temp.columns)}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Erreur avec le fichier {fichier} : {e}")