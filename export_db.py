"""
export_db.py — Exporte les entreprises de Laravel vers un cache JSON local.
À lancer UNE FOIS après avoir démarré Laravel : python3 export_db.py
Le cache permettra à l'IA de fonctionner même si Laravel est hors ligne.
"""
import json, os, math, sys
import requests
from dotenv import load_dotenv

load_dotenv()

LARAVEL_URL   = os.getenv("LARAVEL_API_URL", "http://localhost:8000/api")
LARAVEL_TOKEN = os.getenv("LARAVEL_AI_TOKEN", "")
CACHE_FILE    = "entreprises_cache.json"

def haversine(lat1, lng1, lat2, lng2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return round(R * 2 * math.asin(math.sqrt(a)), 2)

headers = {"Accept": "application/json"}
if LARAVEL_TOKEN:
    headers["Authorization"] = f"Bearer {LARAVEL_TOKEN}"

print(f"Connexion à {LARAVEL_URL}...")
try:
    # Récupérer depuis l'endpoint public (rayon large = tout le Bénin)
    r = requests.get(f"{LARAVEL_URL}/ai/services/nearby",
        params={"lat": 9.3, "lng": 2.3, "radius": 800, "limit": 500},
        headers=headers, timeout=30)

    if r.status_code != 200:
        print(f"Erreur HTTP {r.status_code}")
        print("Vérifiez que Laravel tourne : php artisan serve")
        sys.exit(1)

    data = r.json()
    services = data.get("data", data) if isinstance(data, dict) else data
    print(f"{len(services)} entreprise(s) récupérée(s)")

    # Sauvegarder le cache
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(services, f, ensure_ascii=False, indent=2)

    print(f"Cache sauvegardé → {CACHE_FILE}")
    print("\nAperçu des 3 premières entreprises :")
    for s in services[:3]:
        ent  = s.get("entreprise") or {}
        name = s.get("name") or ent.get("name", "?")
        phone = ent.get("call_phone", "?")
        print(f"  - {name} | Tél: {phone}")

except requests.exceptions.ConnectionError:
    print("Laravel inaccessible. Lancez d'abord : php artisan serve")
    sys.exit(1)
