Après avoir cloner le projet utiliser la commande suivante pour installer les dépendance :

  pip install -r requirements.txt

Ensuite démarrer dans trois différents terminal les commande suivante : 

  ollama pull qwen2.5:3b

  ollama serve

  python3 app.py

Créer le modèle CareEasy personnalisé
  ollama create careasy -f Modelfile

Exporter la base de donnée dans un cache JSON 
  python3 export_db.py

Lancer l'IA
  bash start_careasy.sh
