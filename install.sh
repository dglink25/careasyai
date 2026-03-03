set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo -e "${NC}"

echo -e "\n${YELLOW}[1/5] Installation Ollama (LLM local)...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}Ollama déjà installé : $(ollama --version)${NC}"
else
    echo "Téléchargement Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}Ollama installé${NC}"
fi

echo -e "\n${YELLOW}[2/5] Démarrage Ollama...${NC}"
if pgrep -x "ollama" > /dev/null; then
    echo -e "${GREEN}Ollama déjà en cours d'exécution${NC}"
else
    ollama serve &>/tmp/ollama.log &
    sleep 3
    echo -e "${GREEN}Ollama démarré (PID: $!)${NC}"
fi

echo -e "\n${YELLOW}[3/5] Téléchargement du modèle qwen2.5:3b...${NC}"
echo "    (~2 GB — patientez, c'est la seule fois)"
ollama pull qwen2.5:3b
echo -e "${GREEN}Modèle prêt${NC}"

echo -e "\n${YELLOW}[4/5] Installation des packages Python...${NC}"

# Détecter conda ou venv
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Conda détecté : $CONDA_DEFAULT_ENV"
    pip install -r requirements.txt
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Création d'un environnement virtuel..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi
echo -e "${GREEN}Packages installés${NC}"

echo -e "\n${YELLOW}[5/5] Configuration .env...${NC}"
if [ ! -f ".env" ]; then
cat > .env << 'ENVEOF'
# CareEasy AI — Configuration 100% gratuite

# Ollama (LLM local — aucune clé requise)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b

# ChromaDB (base vectorielle locale — aucune clé requise)
CHROMA_DIR=./chroma_db

# Embeddings locaux (aucune clé requise)
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Laravel (votre projet existant)
LARAVEL_API_URL=http://localhost:8000/api
LARAVEL_AI_TOKEN=REMPLACER_PAR_VOTRE_TOKEN_SANCTUM

# YouTube (optionnel — gratuit avec quota)
YOUTUBE_API_KEY=

# Audio
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu

# Flask
FLASK_ENV=development
FLASK_PORT=5000
ENVEOF
    echo -e "${GREEN}Fichier .env créé${NC}"
    echo -e "${YELLOW}   Éditez .env et mettez votre LARAVEL_AI_TOKEN${NC}"
else
    echo -e "${GREEN}.env existe déjà${NC}"
fi

echo -e "\n${GREEN}"
echo "╔══════════════════════════════════════════════════╗"
echo "║   ✅ Installation terminée !                    ║"
echo "║                                                  ║"
echo "║   Pour lancer l'API :                           ║"
echo "║     python3 app.py                              ║"
echo "║                                                  ║"
echo "║   Pour tester :                                 ║"
echo "║     curl -X POST http://localhost:5000/api/v1/chat ║"
echo "║       -H 'Content-Type: application/json'       ║"
echo "║       -d '{\"message\": \"Ma moto ne démarre pas\"}' ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"
