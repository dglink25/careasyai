"""
laravel_bridge.py
──────────────────
Pont Python → API Laravel CareEasy.
Basé sur le schéma réel de la base de données (migrations analysées).

Tables utilisées :
  - entreprises      : name, latitude, longitude, call_phone, whatsapp_phone,
                       google_formatted_address, status (validated), status_online
  - services         : name, entreprise_id, domaine_id, start_time, end_time,
                       is_open_24h, price, descriptions
  - domaines         : id, name (20 types de services auto)
  - locations_benin  : arrondissement, commune, departement, latitude, longitude
  - conversations    : user_one_id, user_two_id
  - messages         : conversation_id, sender_id (null=IA), content, type,
                       file_path, ai_metadata, latitude, longitude
  - ai_logs          : message_id, ai_input, ai_output, detected_intent,
                       detected_language, confidence, model_version
  - ai_sessions      : conversation_id, user_id, detected_language,
                       current_intent, context
  - ai_feedbacks     : message_id, rating, comment, is_helpful
"""

import os
import math
import logging
from typing import Optional, List, Dict, Any

import requests

log = logging.getLogger(__name__)

LARAVEL_API_URL   = os.getenv("LARAVEL_API_URL",   "http://localhost:8000/api")
LARAVEL_AI_TOKEN  = os.getenv("LARAVEL_AI_TOKEN",  "")
TIMEOUT           = 6  # secondes


def _h() -> Dict:
    """Headers HTTP avec token Sanctum."""
    h = {"Accept": "application/json", "Content-Type": "application/json"}
    if LARAVEL_AI_TOKEN:
        h["Authorization"] = f"Bearer {LARAVEL_AI_TOKEN}"
    return h


# ══════════════════════════════════════════════════════════════════════════════
# SERVICES & ENTREPRISES
# ══════════════════════════════════════════════════════════════════════════════

def get_nearby_services(
    lat: float,
    lng: float,
    radius_km: float = 10,
    domaine: Optional[str] = None,
    limit: int = 8,
) -> List[Dict]:
    """
    Retourne les services proches triés par distance.
    L'endpoint Laravel calcule la distance ou on la calcule ici.

    Route Laravel attendue : GET /api/ai/services/nearby
    Params : lat, lng, radius, domaine, limit
    """
    try:
        params: Dict = {"lat": lat, "lng": lng, "radius": radius_km, "limit": limit}
        if domaine:
            params["domaine"] = domaine

        resp = requests.get(
            f"{LARAVEL_API_URL}/ai/services/nearby",
            params=params, headers=_h(), timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            raw = resp.json()
            services = raw.get("data", raw) if isinstance(raw, dict) else raw
            return _add_distance(services, lat, lng)

        log.warning(f"nearby services HTTP {resp.status_code}")
        return []

    except requests.exceptions.ConnectionError:
        log.warning("API Laravel hors ligne — services non disponibles")
        return []
    except Exception as e:
        log.error(f"get_nearby_services : {e}")
        return []


def get_all_domaines() -> List[str]:
    """
    Retourne la liste des domaines depuis la DB.
    Fallback sur la liste statique (identique au seeder DomainesSeeder.php).
    """
    try:
        resp = requests.get(f"{LARAVEL_API_URL}/ai/domaines", headers=_h(), timeout=TIMEOUT)
        if resp.status_code == 200:
            return [d["name"] for d in resp.json().get("data", resp.json())]
    except Exception:
        pass
    # Fallback statique (miroir exact du DomainesSeeder)
    return [
        "Garage mécanique", "Vente de voitures", "Vente de motos",
        "Location de voitures", "Station d'essence", "Lavage automobile",
        "Électricien auto", "Climatisation auto", "Peinture auto", "Tôlerie",
        "Pneumatique / vulcanisation", "Dépannage / remorquage",
        "Diagnostic automobile", "Changement d'huile", "Assurance automobile",
        "École de conduite", "Vente de pièces détachées",
        "Maintenance poids lourds", "Réparation moto", "Vente de vélos / entretien",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# LOCALISATION BÉNIN (table locations_benin)
# ══════════════════════════════════════════════════════════════════════════════

def resolve_location(query: str) -> Optional[Dict]:
    """
    Résout une localité béninoise depuis un texte libre.
    Ex: "je suis à Akpakpa, Cotonou" → {commune, departement, latitude, longitude}

    Route Laravel : GET /api/ai/locations?q=Akpakpa&limit=1
    """
    try:
        resp = requests.get(
            f"{LARAVEL_API_URL}/ai/locations",
            params={"q": query, "limit": 1}, headers=_h(), timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            raw = resp.json()
            results = raw.get("data", raw) if isinstance(raw, dict) else raw
            return results[0] if results else None
    except Exception as e:
        log.warning(f"resolve_location '{query}': {e}")
    return None


def get_communes() -> List[str]:
    """Liste de toutes les communes du Bénin (depuis locations_benin)."""
    try:
        resp = requests.get(f"{LARAVEL_API_URL}/ai/locations/communes", headers=_h(), timeout=TIMEOUT)
        if resp.status_code == 200:
            return resp.json().get("communes", [])
    except Exception:
        pass
    # Fallback statique principales communes béninoises
    return [
        "Cotonou", "Porto-Novo", "Parakou", "Djougou", "Bohicon",
        "Kandi", "Lokossa", "Ouidah", "Abomey", "Natitingou",
        "Abomey-Calavi", "Allada", "Sèmè-Kpodji", "Calavi",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONS & MESSAGES
# ══════════════════════════════════════════════════════════════════════════════

def save_ai_message(
    conversation_id: int,
    content: str,
    ai_metadata: Optional[Dict] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
) -> Optional[int]:
    """
    Sauvegarde le message IA dans la table messages.
    sender_id = null → identifie le message comme venant de l'IA (CarAI user).

    Route Laravel : POST /api/ai/messages
    """
    try:
        payload = {
            "conversation_id": conversation_id,
            "sender_id":       None,        # null = message IA (cf. seeder : CarAI user)
            "content":         content,
            "type":            "text",
            "ai_metadata":     ai_metadata or {},
        }
        if lat: payload["latitude"]  = lat
        if lng: payload["longitude"] = lng

        resp = requests.post(
            f"{LARAVEL_API_URL}/ai/messages",
            json=payload, headers=_h(), timeout=TIMEOUT,
        )
        if resp.status_code in (200, 201):
            return resp.json().get("id")
        log.warning(f"save_ai_message HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        log.warning(f"save_ai_message : {e}")
    return None


def get_conversation_history(conversation_id: int, limit: int = 10) -> List[Dict]:
    """
    Récupère l'historique de la conversation pour le contexte multi-tours.
    Retourne au format OpenAI messages [{"role": "user"|"assistant", "content": "..."}].

    Route Laravel : GET /api/ai/conversations/{id}/messages?limit=10
    """
    try:
        resp = requests.get(
            f"{LARAVEL_API_URL}/ai/conversations/{conversation_id}/messages",
            params={"limit": limit}, headers=_h(), timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            raw  = resp.json()
            msgs = raw.get("data", raw) if isinstance(raw, dict) else raw
            history = []
            for m in msgs:
                role    = "assistant" if m.get("sender_id") is None else "user"
                content = m.get("content", "")
                if content:
                    history.append({"role": role, "content": content})
            return history
    except Exception as e:
        log.warning(f"get_conversation_history : {e}")
    return []


def save_ai_session(
    conversation_id: int,
    user_id: int,
    detected_language: str = "fr",
    current_intent: Optional[str] = None,
    context: Optional[Dict] = None,
) -> None:
    """
    Met à jour ou crée la session IA (table ai_sessions).
    Route Laravel : POST /api/ai/sessions
    """
    try:
        requests.post(
            f"{LARAVEL_API_URL}/ai/sessions",
            json={
                "conversation_id":  conversation_id,
                "user_id":          user_id,
                "detected_language": detected_language,
                "current_intent":   current_intent,
                "context":          context or {},
            },
            headers=_h(), timeout=TIMEOUT,
        )
    except Exception:
        pass  # Non bloquant


def log_ai_interaction(
    message_id: Optional[int],
    ai_input:   str,
    ai_output:  str,
    intent:     str = "",
    language:   str = "fr",
    confidence: float = 1.0,
    model:      str = "gpt-4o",
) -> None:
    """
    Enregistre dans la table ai_logs.
    Route Laravel : POST /api/ai/logs
    """
    try:
        requests.post(
            f"{LARAVEL_API_URL}/ai/logs",
            json={
                "message_id":       message_id,
                "ai_input":         ai_input[:5000],
                "ai_output":        ai_output[:5000],
                "detected_intent":  intent,
                "detected_language": language,
                "confidence":       confidence,
                "model_version":    model,
            },
            headers=_h(), timeout=3,
        )
    except Exception:
        pass  # Non bloquant


# ══════════════════════════════════════════════════════════════════════════════
# FORMATAGE RÉPONSE
# ══════════════════════════════════════════════════════════════════════════════

def format_services_text(services: List[Dict], lang: str = "fr") -> str:
    """
    Formate la liste des services en texte Markdown pour l'IA.
    Utilise les champs réels de la table entreprises/services.
    """
    if not services:
        msgs = {
            "fr":  "Aucun service disponible à proximité.",
            "en":  "No nearby services found.",
            "fon": "Sín service e ɖò alɔgɔ towe mɛ ǎ.",
        }
        return msgs.get(lang, msgs["fr"])

    headers = {
        "fr":  "🔧 **Services disponibles à proximité :**",
        "en":  "🔧 **Nearby services:**",
        "fon": "🔧 **Sín service lɛ ɖò alɔgɔ towe:**",
    }
    lines = [headers.get(lang, headers["fr"])]

    for i, s in enumerate(services[:5], 1):
        # Les données peuvent venir avec ou sans relation entreprise
        ent      = s.get("entreprise") or {}
        name     = s.get("name") or ent.get("name", f"Service {i}")
        dist     = s.get("distance_label", f"{s.get('distance_km', '?')} km")
        phone    = ent.get("call_phone")    or s.get("call_phone", "")
        whatsapp = ent.get("whatsapp_phone") or s.get("whatsapp_phone", "")
        address  = ent.get("google_formatted_address", "")
        is_24h   = s.get("is_open_24h", False)
        start    = s.get("start_time", "")
        end      = s.get("end_time", "")
        online   = "🟢 En ligne" if ent.get("status_online") else ""

        hours = "24h/24" if is_24h else (f"{start}–{end}" if start else "")

        line = f"\n**{i}. {name}** — 📍 {dist} {online}"
        if address:  line += f"\n   📌 {address}"
        if hours:    line += f"\n   🕐 {hours}"
        if phone:    line += f"\n   📞 {phone}"
        if whatsapp: line += f"\n   💬 WhatsApp : {whatsapp}"
        lines.append(line)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    d1 = math.radians(lat2 - lat1)
    d2 = math.radians(lon2 - lon1)
    a  = math.sin(d1/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d2/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _dist_label(km: float) -> str:
    if km < 1:  return f"{int(km * 1000)} m"
    if km < 10: return f"{km:.1f} km"
    return f"{int(km)} km"


def _add_distance(services: List[Dict], user_lat: float, user_lng: float) -> List[Dict]:
    """Calcule la distance pour chaque service et trie par distance."""
    for s in services:
        ent  = s.get("entreprise") or {}
        slat = ent.get("latitude")  or s.get("latitude")
        slng = ent.get("longitude") or s.get("longitude")
        if slat and slng:
            d = _haversine(user_lat, user_lng, float(slat), float(slng))
            s["distance_km"]    = round(d, 1)
            s["distance_label"] = _dist_label(d)
        else:
            s["distance_km"]    = 999
            s["distance_label"] = "?"
    services.sort(key=lambda x: x["distance_km"])
    return services
