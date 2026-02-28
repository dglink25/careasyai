"""
agent.py — CareEasy AI Agent (VERSION CORRIGÉE v2)
─────────────────────────────────────────────────
Corrections :
  1. Intent "localisation" étendu (lavage, essence, cherche, trouver...)
  2. Domaine détecté depuis le texte (lavage → Lavage automobile)
  3. Ville béninoise détectée dans le texte SANS GPS obligatoire
  4. Format réponse adapté (services → liste détaillée, diagnostic → structure)
  5. Services affichés avec TOUS les détails disponibles
"""

import os
import re
import base64
import logging
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

import requests as http_req
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
CHROMA_DIR   = os.getenv("CHROMA_DIR",   "./chroma_db")
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

SYSTEM_PROMPT_DIAGNOSTIC = """Tu es CareEasy AI, assistant automobile et moto au Bénin.
Réponds TOUJOURS dans la même langue que l'utilisateur (Fon/Français/Anglais/Swahili).
FORMAT OBLIGATOIRE :
🔍 **Diagnostic** : [problèmes probables]
⚡ **Urgence** : URGENT 🔴 / ATTENTION 🟡 / OK 🟢
✅ **Vérifications immédiates** : [ce que l'utilisateur peut faire seul]
🔧 **Solutions** : [étapes par ordre de priorité]
💰 **Coût estimé** : [fourchette en FCFA, marché béninois 2025]"""

SYSTEM_PROMPT_SERVICES = """Tu es CareEasy AI, assistant automobile et moto au Bénin.
Réponds TOUJOURS dans la même langue que l'utilisateur.
Si des services sont fournis dans le contexte, liste-les TOUS avec TOUS les détails :
nom, distance, téléphone, WhatsApp, horaires, adresse, tarif.
Si aucun service trouvé, explique comment chercher localement (Google Maps, voisinage)."""

SYSTEM_PROMPT_GENERAL = """Tu es CareEasy AI, assistant automobile et moto au Bénin.
Réponds TOUJOURS dans la même langue que l'utilisateur. Sois utile et concis."""

VEHICLE_ALIASES = {
    "dayang": ("Dayang", "DY110-3"), "dayan": ("Dayang", "DY110-3"),
    "lifan": ("Lifan", "LF110"), "zemidjan": ("Dayang", "DY110-3"),
    "zémidjan": ("Dayang", "DY110-3"), "zem": ("Dayang", "DY110-3"),
    "keke": ("Bajaj", "RE (Keke)"), "kéké": ("Bajaj", "RE (Keke)"),
    "tricycle": ("Bajaj", "RE (Keke)"), "moto chinoise": ("Dayang", "DY110-3"),
    "110cc": ("Dayang", "DY110-3"), "125cc": ("Dayang", "DY125"),
    "hilux": ("Toyota", "Hilux"), "corolla": ("Toyota", "Corolla"),
    "land cruiser": ("Toyota", "Land Cruiser"), "hiace": ("Toyota", "HiAce"),
    "prado": ("Toyota", "Prado"), "206": ("Peugeot", "206"),
    "405": ("Peugeot", "405"), "pejo": ("Peugeot", "206"),
    "vespa": ("Vespa", "Primavera"), "yamaha": ("Yamaha", ""),
    "honda": ("Honda", ""), "suzuki": ("Suzuki", ""),
    "tvs": ("TVS", ""), "bajaj": ("Bajaj", ""),
}

DOMAINE_KEYWORDS = {
    "Lavage automobile":           ["lavage", "laver", "wash", "nettoyer", "nettoyage"],
    "Station d'essence":           ["essence", "carburant", "fuel", "station", "gazoil"],
    "Garage mécanique":            ["mécanicien", "mécanique", "garage", "réparer"],
    "Réparation moto":             ["moto", "zemidjan", "kéké", "keke", "deux-roues"],
    "Changement d'huile":          ["huile", "vidange", "oil"],
    "Pneumatique / vulcanisation": ["pneu", "tyre", "crevaison", "vulcanisation"],
    "Électricien auto":            ["électricien", "électrique", "alternateur", "câblage"],
    "Climatisation auto":          ["climatisation", "clim", "air conditionné"],
    "Peinture auto":               ["peinture", "carrosserie", "rayure", "repeindre"],
    "Tôlerie":                     ["tôlerie", "tôle", "cabossé", "bosse", "accident"],
    "Dépannage / remorquage":      ["dépannage", "remorquage", "remorquer", "dépanneuse"],
    "Diagnostic automobile":       ["diagnostic", "scanner", "voyant", "obd"],
    "Vente de pièces détachées":   ["pièces", "pièce détachée", "spare parts"],
    "Assurance automobile":        ["assurance", "insurance"],
    "École de conduite":           ["permis", "auto-école", "conduire"],
    "Vente de voitures":           ["acheter voiture", "vendre voiture"],
    "Vente de motos":              ["acheter moto", "vendre moto"],
}

VILLES_BENIN = {
    "cotonou": (6.3676, 2.4252), "porto-novo": (6.4969, 2.6289),
    "parakou": (9.3370, 2.6280), "djougou": (9.7085, 1.6660),
    "bohicon": (7.1780, 2.0667), "natitingou": (10.3148, 1.3762),
    "lokossa": (6.6394, 1.7219), "ouidah": (6.3596, 2.0833),
    "abomey": (7.1833, 1.9833), "kandi": (11.1333, 2.9333),
    "abomey-calavi": (6.4486, 2.3551), "calavi": (6.4486, 2.3551),
    "seme-kpodji": (6.3833, 2.6333), "allada": (6.6667, 2.1500),
    "dogbo": (6.7833, 1.7833), "save": (7.9667, 2.4833),
    "bembereke": (10.2250, 2.6639), "nikki": (9.9417, 3.2083),
    "malanville": (11.8667, 3.3833), "kétou": (7.3583, 2.6000),
    "pobe": (6.9667, 2.6667), "aplahoue": (6.9333, 1.6833),
    "dassa-zoume": (7.7500, 2.1833), "glazoue": (7.9833, 2.2167),
    "bante": (8.4167, 1.8833), "tchaourou": (8.8833, 2.5833),
}


class CareEasyAgent:

    def __init__(self):
        log.info("Initialisation CareEasyAgent...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._check_ollama()
        log.info(f"CareEasyAgent prêt (modèle: {OLLAMA_MODEL})")

    def _check_ollama(self):
        try:
            resp = http_req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                log.info(f"Ollama OK — modèles : {models}")
                if not any(OLLAMA_MODEL.split(":")[0] in m for m in models):
                    log.warning(f"⚠️ Modèle '{OLLAMA_MODEL}' absent. Lancez : ollama pull {OLLAMA_MODEL}")
        except Exception:
            log.warning(f"⚠️ Ollama inaccessible. Lancez : ollama serve")

    def _ollama(self, messages: List[Dict], temperature: float = 0.2, max_tokens: int = 1800) -> str:
        try:
            resp = http_req.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                      "options": {"temperature": temperature, "num_predict": max_tokens}},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except http_req.exceptions.ConnectionError:
            raise RuntimeError(f"Ollama inaccessible. Lancez : ollama serve && ollama pull {OLLAMA_MODEL}")
        except Exception as e:
            raise RuntimeError(f"Erreur Ollama : {e}")

    def _embed(self, text: str) -> List[float]:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def index_document(self, chunks: List[str], collection_name: str,
                       metadatas: Optional[List[Dict]] = None) -> int:
        if not chunks:
            return 0
        collection = self.chroma.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})
        embeddings = [self._embed(c) for c in chunks]
        ids        = [f"{collection_name}_{i}" for i in range(len(chunks))]
        metas      = metadatas or [{"source": collection_name}] * len(chunks)
        collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metas)
        return len(chunks)

    def _rag(self, query: str, collection_name: str, k: int = 4) -> Tuple[str, List[str]]:
        if not query or not collection_name:
            return "", []
        try:
            collection = self.chroma.get_collection(collection_name)
            results    = collection.query(
                query_embeddings=[self._embed(query)],
                n_results=min(k, collection.count()),
            )
            chunks = results["documents"][0] if results["documents"] else []
            return "\n\n---\n\n".join(chunks[:4]), chunks
        except Exception as e:
            log.warning(f"RAG '{collection_name}' : {e}")
            return "", []

    def get_collections(self) -> List[str]:
        return [c.name for c in self.chroma.list_collections()]

    def _detect_intent(self, message: str, has_image: bool) -> str:
        if has_image:
            return "diagnostic"
        m = message.lower()
        localisation_mots = [
            "garage", "mécanicien", "proche", "près", "trouver", "cherche",
            "chercher", "où", "prestataire", "entreprise", "adresse", "find",
            "where", "nearby", "lavage", "laver", "station", "essence",
            "vulcanisation", "électricien", "climatisation", "peinture",
            "remorquage", "dépannage", "assurance", "service auto",
        ]
        if any(w in m for w in localisation_mots):
            return "localisation"
        diagnostic_mots = [
            "panne", "bruit", "fume", "frein", "voyant", "problème", "démarre",
            "broken", "noise", "fault", "casse", "coule", "fuite", "vibr",
            "chauffe", "grince", "claque", "ne marche", "marche plus", "phare",
        ]
        if any(w in m for w in diagnostic_mots):
            return "diagnostic"
        if any(w in m for w in ["entretien", "vidange", "changer", "révision",
                                  "maintenance", "huile", "filtre"]):
            return "maintenance"
        return "info_generale"

    def _detect_domaine_from_text(self, message: str) -> Optional[str]:
        m = message.lower()
        for domaine, keywords in DOMAINE_KEYWORDS.items():
            if any(kw in m for kw in keywords):
                return domaine
        return None

    def _detect_ville_from_text(self, message: str) -> Optional[Tuple[str, float, float]]:
        m = message.lower()
        m_simple = (m.replace("è","e").replace("é","e").replace("ô","o")
                    .replace("à","a").replace("ê","e"))
        for ville in sorted(VILLES_BENIN.keys(), key=len, reverse=True):
            ville_simple = (ville.replace("è","e").replace("é","e")
                            .replace("ô","o").replace("à","a"))
            if ville_simple in m_simple or ville in m:
                lat, lng = VILLES_BENIN[ville]
                return ville.title(), lat, lng
        return None

    def _intent_to_domaine(self, intent: str, model: str, message: str = "") -> Optional[str]:
        domaine = self._detect_domaine_from_text(message)
        if domaine:
            return domaine
        is_moto = any(w in model.lower() for w in ["dy", "lf", "wave", "cb", "keke", "bajaj"])
        if intent in ("diagnostic", "localisation"):
            return "Réparation moto" if is_moto else "Garage mécanique"
        if intent == "maintenance":
            return "Changement d'huile"
        return None

    def _quick_identify(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        t = text.lower()
        for alias, (make, model) in VEHICLE_ALIASES.items():
            if alias in t:
                return make, model
        return None, None

    def _find_collection(self, make: str, model: str) -> Optional[str]:
        collections = self.get_collections()
        ms = make.lower().replace(" ", "_").replace("-", "_")
        ml = (model.lower().replace(" ", "_").replace("-", "_")
              .replace("/", "_").replace("(", "").replace(")", ""))
        for name in collections:
            if ms in name and (not ml or ml[:5] in name):
                return name
        return None

    def _get_services(self, message: str, intent: str, vehicle_model: str,
                      user_location: Optional[Dict]) -> List[Dict]:
        lat, lng, ville_nom = None, None, None
        if user_location and user_location.get("lat"):
            lat, lng = user_location["lat"], user_location["lng"]
            ville_nom = user_location.get("address", "")
        else:
            result = self._detect_ville_from_text(message)
            if result:
                ville_nom, lat, lng = result
                log.info(f"Ville détectée : {ville_nom} ({lat}, {lng})")
        if not lat:
            return []
        domaine = self._intent_to_domaine(intent, vehicle_model, message)
        log.info(f"Recherche services : {ville_nom}, domaine={domaine}")
        try:
            from laravel_bridge import get_nearby_services
            return get_nearby_services(lat=lat, lng=lng, radius_km=20,
                                       domaine=domaine, limit=10)
        except Exception as e:
            log.warning(f"Laravel services : {e}")
            return []

    def analyze_photo(self, image_bytes: bytes, mime_type: str = "image/jpeg",
                      vehicle_make: str = "", vehicle_model: str = "",
                      user_description: str = "") -> Tuple[str, str]:
        vehicle      = f"{vehicle_make} {vehicle_model}".strip() or "véhicule"
        vision_model = self._get_vision_model()
        if vision_model:
            b64    = base64.b64encode(image_bytes).decode("utf-8")
            prompt = (f"Mécanicien expert Bénin. Véhicule: {vehicle}. "
                      f"Description: '{user_description or 'aucune'}'. "
                      "Analyse la photo: 1) Ce que tu vois 2) Diagnostic "
                      "3) Urgence URGENT/ATTENTION/OK 4) Solutions 5) Coût FCFA")
            try:
                resp = http_req.post(f"{OLLAMA_URL}/api/generate",
                    json={"model": vision_model, "prompt": prompt,
                          "images": [b64], "stream": False}, timeout=120)
                if resp.status_code == 200:
                    a = resp.json().get("response", "").strip()
                    return a, self._extract_urgency(a)
            except Exception as e:
                log.warning(f"Vision model : {e}")
        desc = user_description or "problème visible sur photo"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DIAGNOSTIC},
            {"role": "user", "content": f"Véhicule: {vehicle}\nProblème: {desc}\nDonne le diagnostic."},
        ]
        try:
            a = self._ollama(messages)
            return a, self._extract_urgency(a)
        except Exception:
            return "Décrivez le problème en texte svp.", "unknown"

    def _get_vision_model(self) -> Optional[str]:
        try:
            resp = http_req.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                for vm in ["llava", "bakllava", "moondream", "llava-phi3", "minicpm-v"]:
                    if any(vm in m for m in models):
                        return next(m for m in models if vm in m)
        except Exception:
            pass
        return None

    def chat(self, message=None, message_fr=None, image_bytes=None,
             image_mime="image/jpeg", collection_name=None, vehicle_make=None,
             vehicle_model=None, user_location=None, user_lang="fr",
             history=None) -> Dict[str, Any]:
        history = history or []
        query   = message_fr or message or ""
        if not collection_name and query:
            make, model = self._quick_identify(query)
            if make:
                vehicle_make    = vehicle_make  or make
                vehicle_model   = vehicle_model or model
                collection_name = self._find_collection(make, model)
        intent           = self._detect_intent(query, image_bytes is not None)
        manual_context, sources = self._rag(query, collection_name or "")
        services_proches = self._get_services(query, intent, vehicle_model or "", user_location)
        answer_fr, urgency = self._generate(
            query=query, image_bytes=image_bytes, image_mime=image_mime,
            vehicle_make=vehicle_make or "", vehicle_model=vehicle_model or "",
            manual_context=manual_context, services_proches=services_proches,
            user_location=user_location, intent=intent, history=history,
        )
        return {
            "answer_fr": answer_fr, "intent": intent, "urgency": urgency,
            "services_proches": services_proches[:10],
            "vehicle": {"make": vehicle_make or "", "model": vehicle_model or "",
                        "collection_name": collection_name or ""},
            "sources": sources[:3], "lang": user_lang,
        }

    def _generate(self, query, image_bytes, image_mime, vehicle_make, vehicle_model,
                  manual_context, services_proches, user_location, intent, history):
        if intent == "localisation":
            system = SYSTEM_PROMPT_SERVICES
        elif intent in ("diagnostic", "maintenance"):
            system = SYSTEM_PROMPT_DIAGNOSTIC
        else:
            system = SYSTEM_PROMPT_GENERAL

        vehicle = f"{vehicle_make} {vehicle_model}".strip() or ""
        parts   = []
        if query:   parts.append(f"Question : {query}")
        if vehicle: parts.append(f"Véhicule : {vehicle}")
        if manual_context:
            parts.append(f"\n[Manuel {vehicle}]\n{manual_context[:1500]}")

        if services_proches:
            parts.append(f"\n{self._format_services_for_llm(services_proches)}")
        elif intent == "localisation":
            domaine = self._detect_domaine_from_text(query) or "ce type de service"
            ville   = self._detect_ville_from_text(query)
            lieu    = ville[0] if ville else "votre zone"
            parts.append(
                f"\n[AUCUN SERVICE TROUVÉ dans la BD pour '{domaine}' près de {lieu}. "
                "Donne des conseils pour chercher localement.]"
            )

        if image_bytes:
            return self.analyze_photo(image_bytes, image_mime, vehicle_make, vehicle_model, query)

        messages = [{"role": "system", "content": system}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": "\n\n".join(parts)})

        try:
            answer  = self._ollama(messages, temperature=0.2, max_tokens=1800)
            urgency = self._extract_urgency(answer)
            log.info(f"Réponse (intent={intent}, urgence={urgency})")
            return answer, urgency
        except RuntimeError as e:
            return f"⚠️ Ollama non démarré. ollama serve && ollama pull {OLLAMA_MODEL}", "unknown"
        except Exception as e:
            log.error(f"Erreur : {e}", exc_info=True)
            return "Erreur. Vérifiez les logs.", "unknown"

    def _format_services_for_llm(self, services: List[Dict]) -> str:
        lines = [f"[{len(services)} SERVICE(S) TROUVÉ(S)]"]
        for i, s in enumerate(services[:10], 1):
            ent      = s.get("entreprise") or {}
            name     = s.get("name") or ent.get("name", f"Service {i}")
            dist     = s.get("distance_label") or f"{s.get('distance_km','?')} km"
            phone    = ent.get("call_phone")    or s.get("call_phone", "")
            whatsapp = ent.get("whatsapp_phone") or s.get("whatsapp_phone", "")
            address  = ent.get("google_formatted_address", "")
            is_24h   = s.get("is_open_24h", False)
            start, end = s.get("start_time",""), s.get("end_time","")
            hours    = "24h/24" if is_24h else (f"{start}–{end}" if start else "?")
            domaine  = s.get("domaine", "")
            price    = s.get("price", "")
            desc     = s.get("descriptions", "")
            ln = f"\n{i}. {name}"
            if domaine:  ln += f" ({domaine})"
            ln += f" | Distance: {dist}"
            if ent.get("status_online"): ln += " ✅ En ligne"
            if address:  ln += f"\n   Adresse: {address}"
            ln += f"\n   Horaires: {hours}"
            if phone:    ln += f"\n   Tél: {phone}"
            if whatsapp: ln += f"\n   WhatsApp: {whatsapp}"
            if price:    ln += f"\n   Tarif: {price}"
            if desc:     ln += f"\n   Info: {desc[:120]}"
            lines.append(ln)
        return "\n".join(lines)

    def _extract_urgency(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["urgent", "critique", "immédiatement", "ne pas rouler",
                                   "arrêter", "dangereux", "🔴"]):
            return "critical"
        if any(w in t for w in ["attention", "important", "48h", "rapidement", "🟡"]):
            return "important"
        if any(w in t for w in ["ok", "mineur", "peut attendre", "normal", "🟢"]):
            return "minor"
        return "unknown"


_agent: Optional[CareEasyAgent] = None

def get_agent() -> CareEasyAgent:
    global _agent
    if _agent is None:
        _agent = CareEasyAgent()
    return _agent