# 1. Utiliser une image Python légère officielle
FROM python:3.11-slim

# 2. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 3. Copier d'abord les dépendances pour maximiser le cache Docker
COPY requirements.txt .

# 4. Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copier tout le reste du projet (scripts, modèles .pkl)
COPY . .

# 6. Exposer le port par défaut de Streamlit
EXPOSE 8501

# 7. Lancer l'application au démarrage du conteneur
CMD ["streamlit", "run", "interface.py", "--server.port=8501", "--server.address=0.0.0.0"]