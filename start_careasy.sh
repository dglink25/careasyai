#!/bin/bash
GREEN='\033[0;32m' YELLOW='\033[1;33m' RED='\033[0;31m' NC='\033[0m'

echo -e "${GREEN}=== CareEasy AI — Démarrage ===${NC}"

# 1. Ollama
echo -e "\n${YELLOW}[1/4] Vérification Ollama...${NC}"
if ! pgrep -x ollama > /dev/null; then
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 4
    echo -e "${GREEN}Ollama démarré${NC}"
else
    echo -e "${GREEN}Ollama déjà actif${NC}"
fi

# 2. Créer le modèle personnalisé careasy si absent
echo -e "\n${YELLOW}[2/4] Modèle CareEasy personnalisé...${NC}"
if ollama list 2>/dev/null | grep -q "careasy"; then
    echo -e "${GREEN}Modèle careasy déjà créé${NC}"
else
    echo "Création du modèle careasy (30 secondes)..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/Modelfile" ]; then
        ollama create careasy -f "$SCRIPT_DIR/Modelfile"
        echo -e "${GREEN}Modèle careasy créé${NC}"
    else
        echo -e "${RED}Modelfile absent — utilisation de qwen2.5:3b${NC}"
        # Fallback: utiliser qwen2.5:3b si careasy absent
        sed -i 's/OLLAMA_MODEL=.*/OLLAMA_MODEL=qwen2.5:3b/' .env 2>/dev/null
    fi
fi

# Mettre à jour le .env pour utiliser careasy
if grep -q "OLLAMA_MODEL" .env 2>/dev/null; then
    sed -i 's/OLLAMA_MODEL=.*/OLLAMA_MODEL=careasy/' .env
else
    echo "OLLAMA_MODEL=careasy" >> .env
fi

# 3. Vérifier Laravel
echo -e "\n${YELLOW}[3/4] Vérification Laravel...${NC}"
LARAVEL_URL=$(grep LARAVEL_API_URL .env 2>/dev/null | cut -d= -f2 | sed 's|/api||')
LARAVEL_URL=${LARAVEL_URL:-http://localhost:8000}

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$LARAVEL_URL" --max-time 3 2>/dev/null)
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
    echo -e "${GREEN}Laravel actif sur $LARAVEL_URL${NC}"
else
    echo -e "${RED}Laravel HORS LIGNE (HTTP $HTTP_CODE)${NC}"
    echo ""
    echo "   Pour lancer Laravel, ouvrez un NOUVEAU terminal et tapez :"
    echo "   cd chemin vers projet laravel && php artisan serve"
    echo ""
    echo -e "${YELLOW}   L'IA démarrera quand même mais sans accès à la base de données.${NC}"
    echo "   Exportez d'abord vos entreprises : python3 export_db.py"
fi

# 4. Lancer Flask
echo -e "\n${YELLOW}[4/4] Démarrage API CareEasy...${NC}"
echo -e "${GREEN}API démarrée sur http://localhost:5000${NC}"
echo ""
echo "Test rapide :"
echo "  curl -X POST http://localhost:5000/api/v1/chat \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\": \"Bonjour\", \"lang\": \"fr\"}'"
echo ""
python3 app.py
