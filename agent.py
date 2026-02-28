"""
agent.py v4 — CareEasy AI (Bénin natif)
─────────────────────────────────────────
Améliorations v4 :
  1. Consultation BD FORCÉE — services injectés directement dans le prompt
  2. Cache JSON local si Laravel hors ligne (export_db.py)
  3. Instructions anti-Google Maps inscrites dans le Modelfile Ollama
  4. Toutes les 77 communes + 12 départements du Bénin
  5. Filtrage intelligent par domaine ET distance
  6. Format réponse structuré avec TOUS les contacts
"""

import os, json, math, base64, logging
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

import requests as http_req
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "careasy")   # modèle personnalisé
CHROMA_DIR   = os.getenv("CHROMA_DIR",   "./chroma_db")
EMBED_MODEL  = os.getenv("EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CACHE_FILE   = os.getenv("CACHE_FILE", "entreprises_cache.json")

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS SYSTÈME
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_STRICT = """Tu es CAREASY, l'IA officielle de CareEasy au Bénin.

⛔ INTERDICTIONS ABSOLUES :
- Ne JAMAIS mentionner Google Maps, Waze, Apple Maps ou toute application externe
- Ne JAMAIS dire "cherchez sur internet" ou "utilisez un moteur de recherche"
- Ne JAMAIS inventer des contacts, adresses ou entreprises
- Ne JAMAIS ignorer les données fournies dans le contexte

✅ OBLIGATIONS :
- Si des entreprises CareEasy sont dans le contexte → les lister TOUTES avec contacts complets
- Si AUCUNE entreprise → dire clairement "Aucune entreprise CareEasy dans cette zone"
  et proposer : "Inscrivez votre entreprise sur CareEasy : support@careasy.bj"
- Toujours répondre dans la langue de l'utilisateur
- Toujours terminer par "Akpé — CareEasy Bénin 🇧🇯"

FORMAT OBLIGATOIRE POUR LES SERVICES :
Pour chaque entreprise :
📌 **[NOM]** — [X] km | [statut en ligne]
📍 Adresse : [adresse]
🕐 Horaires : [horaires]
📞 Tél : [numéro]
💬 WhatsApp : [numéro]
📧 Email : [si disponible]
🔧 Services : [liste]
💰 Tarif : [si disponible]
---"""

SYSTEM_DIAGNOSTIC = SYSTEM_STRICT + """

FORMAT DIAGNOSTIC :
🔍 **Diagnostic** : [causes probables]
⚡ **Urgence** : URGENT 🔴 / ATTENTION 🟡 / OK 🟢
✅ **Vérifier maintenant** : [actions immédiates sans outil]
🔧 **Solutions** : [étapes simples → complexes]
💰 **Coût FCFA** : [fourchette marché béninois 2025]
📍 **Prestataires CareEasy** : [liste depuis la BD ci-dessous]"""

# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES
# ─────────────────────────────────────────────────────────────────────────────

VEHICLE_ALIASES = {
    "dayang": ("Dayang","DY110-3"), "dayan": ("Dayang","DY110-3"),
    "lifan": ("Lifan","LF110"), "zemidjan": ("Dayang","DY110-3"),
    "zémidjan": ("Dayang","DY110-3"), "zem": ("Dayang","DY110-3"),
    "keke": ("Bajaj","RE (Keke)"), "kéké": ("Bajaj","RE (Keke)"),
    "tricycle": ("Bajaj","RE (Keke)"), "moto chinoise": ("Dayang","DY110-3"),
    "110cc": ("Dayang","DY110-3"), "125cc": ("Dayang","DY125"),
    "hilux": ("Toyota","Hilux"), "corolla": ("Toyota","Corolla"),
    "land cruiser": ("Toyota","Land Cruiser"), "hiace": ("Toyota","HiAce"),
    "prado": ("Toyota","Prado"), "206": ("Peugeot","206"),
    "405": ("Peugeot","405"), "pejo": ("Peugeot","206"),
    "vespa": ("Vespa","Primavera"), "yamaha": ("Yamaha",""),
    "honda": ("Honda",""), "suzuki": ("Suzuki",""),
}

DOMAINE_KEYWORDS = {
    "Lavage automobile":           ["lavage","laver","wash","nettoyer","nettoyage"],
    "Station d'essence":           ["essence","carburant","fuel","station","gazoil"],
    "Garage mécanique":            ["mécanicien","mécanique","garage","réparer","réparation","panne"],
    "Réparation moto":             ["moto","zemidjan","kéké","keke","deux-roues"],
    "Changement d'huile":          ["huile","vidange","oil"],
    "Pneumatique / vulcanisation": ["pneu","tyre","crevaison","vulcanisation","gonflage"],
    "Électricien auto":            ["électricien","électrique","alternateur","câblage"],
    "Climatisation auto":          ["climatisation","clim","air conditionné"],
    "Peinture auto":               ["peinture","carrosserie","rayure"],
    "Tôlerie":                     ["tôlerie","tôle","cabossé","bosse","accident"],
    "Dépannage / remorquage":      ["dépannage","remorquage","dépanneuse"],
    "Diagnostic automobile":       ["diagnostic","scanner","voyant","obd"],
    "Vente de pièces détachées":   ["pièces","pièce détachée","spare parts"],
    "Assurance automobile":        ["assurance","insurance"],
    "École de conduite":           ["permis","auto-école","conduire"],
    "Vente de voitures":           ["acheter voiture","vente voiture"],
    "Vente de motos":              ["acheter moto","vente moto"],
    "Location de voitures":        ["louer voiture","location voiture"],
}

# Tous les 77 communes + arrondissements principaux du Bénin
LIEUX_BENIN = {
    # ALIBORI (6 communes)
    "banikoara":(11.300,2.433),"gogounou":(10.833,2.833),"kandi":(11.133,2.933),
    "karimama":(12.067,3.183),"malanville":(11.867,3.383),"segbana":(10.933,3.700),
    # ATACORA (9 communes)
    "boukoumbe":(10.183,1.100),"cobly":(10.517,1.383),"kerou":(10.817,2.100),
    "kouande":(10.333,1.683),"natitingou":(10.315,1.376),"pehonko":(10.217,1.550),
    "tanguieta":(10.617,1.267),"toukountouna":(10.550,1.383),"materie":(10.617,1.067),
    # ATLANTIQUE (9 communes)
    "abomey-calavi":(6.449,2.355),"allada":(6.667,2.150),"calavi":(6.449,2.355),
    "kpomasse":(6.533,2.050),"ouidah":(6.360,2.083),"so-ava":(6.483,2.433),
    "toffo":(6.850,2.083),"tori-bossito":(6.533,2.167),"ze":(6.750,2.333),
    # BORGOU (8 communes)
    "bembereke":(10.225,2.664),"kalale":(10.300,3.367),"n'dali":(9.867,2.717),
    "nikki":(9.942,3.208),"parakou":(9.337,2.628),"perere":(10.467,3.133),
    "sinende":(9.983,2.350),"tchaourou":(8.883,2.583),
    # COLLINES (6 communes)
    "bante":(8.417,1.883),"dassa-zoume":(7.750,2.183),"glazoue":(7.983,2.217),
    "ouesse":(8.500,2.317),"savalou":(7.917,1.983),"save":(7.967,2.483),
    # COUFFO (6 communes)
    "aplahoue":(6.933,1.683),"djakotomey":(6.900,1.717),"dogbo":(6.783,1.783),
    "klouekanme":(6.967,1.750),"lalo":(6.917,1.883),"toviklin":(6.867,1.833),
    # DONGA (4 communes)
    "bassila":(9.000,1.667),"copargo":(9.833,1.550),"djougou":(9.709,1.666),"ouake":(9.617,1.383),
    # LITTORAL (1 commune = Cotonou + quartiers)
    "cotonou":(6.368,2.425),"akpakpa":(6.350,2.450),"cadjehoun":(6.367,2.383),
    "fidjrosse":(6.350,2.383),"haie vive":(6.367,2.417),"dantokpa":(6.367,2.433),
    "zongo":(6.383,2.433),"gbegamey":(6.367,2.383),"agla":(6.383,2.367),
    "godomey":(6.400,2.350),"cocotiers":(6.367,2.400),"jericho":(6.350,2.417),
    "sikeco":(6.333,2.417),"ayetoro":(6.383,2.400),"vossa":(6.350,2.367),
    # MONO (6 communes)
    "athieme":(6.567,1.667),"bopa":(6.600,1.983),"come":(6.400,1.883),
    "grand-popo":(6.283,1.817),"houeyogbe":(6.733,1.717),"lokossa":(6.639,1.722),
    # OUEME (8 communes)
    "adjarra":(6.550,2.700),"adjohoun":(6.717,2.483),"akpro-misserete":(6.567,2.617),
    "avrankou":(6.550,2.667),"bonou":(6.917,2.433),"dangbo":(6.583,2.567),
    "porto-novo":(6.497,2.629),"seme-kpodji":(6.383,2.633),
    # PLATEAU (5 communes)
    "adja-ouere":(7.017,2.500),"ifangni":(6.683,2.733),"ketou":(7.358,2.600),
    "pobe":(6.967,2.667),"sakete":(6.733,2.650),
    # ZOU (9 communes)
    "abomey":(7.183,1.983),"agbangnizoun":(7.100,1.983),"bohicon":(7.178,2.067),
    "cove":(7.317,2.400),"djidja":(7.317,1.983),"ouinhi":(7.233,2.483),
    "zagnanado":(7.267,2.350),"za-kpota":(7.100,2.117),"zogbodomey":(7.000,2.067),
}


def haversine(lat1, lng1, lat2, lng2) -> float:
    R = 6371
    dlat = math.radians(lat2-lat1)
    dlng = math.radians(lng2-lng1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlng/2)**2)
    return round(R*2*math.asin(math.sqrt(a)), 1)


class CareEasyAgent:

    def __init__(self):
        log.info("Initialisation CareEasyAgent v4...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
        self._cache: Optional[List[Dict]] = self._load_cache()
        self._check_ollama()
        log.info(f"CareEasyAgent v4 prêt — modèle: {OLLAMA_MODEL}, cache: {len(self._cache or [])} entreprises")

    # ── Cache local entreprises ───────────────────────────────────────────────

    def _load_cache(self) -> List[Dict]:
        """Charge le cache JSON local des entreprises (fallback si Laravel hors ligne)."""
        if Path(CACHE_FILE).exists():
            try:
                data = json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))
                log.info(f"Cache local chargé : {len(data)} entreprises depuis {CACHE_FILE}")
                return data
            except Exception as e:
                log.warning(f"Cache corrompu : {e}")
        log.info(f"Pas de cache local ({CACHE_FILE}). Lancez : python3 export_db.py")
        return []

    def reload_cache(self):
        self._cache = self._load_cache()

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _check_ollama(self):
        try:
            r = http_req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if r.ok:
                models = [m["name"] for m in r.json().get("models", [])]
                log.info(f"Ollama OK — modèles : {models}")
                base = OLLAMA_MODEL.split(":")[0]
                if not any(base in m for m in models):
                    log.warning(f"⚠️ Modèle '{OLLAMA_MODEL}' absent.")
                    log.warning("  Créez-le : ollama create careasy -f Modelfile")
                    log.warning("  Ou utilisez : OLLAMA_MODEL=qwen2.5:3b dans .env")
        except Exception:
            log.warning("⚠️ Ollama hors ligne. Lancez : ollama serve")

    def _ollama(self, messages: List[Dict], temperature: float = 0.05,
                max_tokens: int = 2000) -> str:
        # Essayer le modèle personnalisé, fallback sur qwen2.5:3b
        for model in [OLLAMA_MODEL, "qwen2.5:3b", "mistral", "llama3.2"]:
            try:
                r = http_req.post(f"{OLLAMA_URL}/api/chat",
                    json={"model": model, "messages": messages, "stream": False,
                          "options": {"temperature": temperature, "num_predict": max_tokens}},
                    timeout=180)
                if r.ok:
                    if model != OLLAMA_MODEL:
                        log.warning(f"Fallback sur modèle : {model}")
                    return r.json()["message"]["content"].strip()
            except http_req.exceptions.ConnectionError:
                raise RuntimeError(f"Ollama hors ligne. Lancez : ollama serve")
            except Exception:
                continue
        raise RuntimeError("Aucun modèle Ollama disponible")

    # ── ChromaDB ──────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def index_document(self, chunks: List[str], collection_name: str,
                       metadatas: Optional[List[Dict]] = None) -> int:
        if not chunks: return 0
        col   = self.chroma.get_or_create_collection(name=collection_name,
                    metadata={"hnsw:space": "cosine"})
        embs  = [self._embed(c) for c in chunks]
        ids   = [f"{collection_name}_{i}" for i in range(len(chunks))]
        metas = metadatas or [{"source": collection_name}] * len(chunks)
        col.upsert(ids=ids, embeddings=embs, documents=chunks, metadatas=metas)
        return len(chunks)

    def _rag(self, query: str, collection_name: str, k: int = 4) -> Tuple[str, List[str]]:
        if not query or not collection_name: return "", []
        try:
            col = self.chroma.get_collection(collection_name)
            res = col.query(query_embeddings=[self._embed(query)],
                            n_results=min(k, col.count()))
            chunks = res["documents"][0] if res["documents"] else []
            return "\n\n---\n\n".join(chunks[:4]), chunks
        except Exception as e:
            log.debug(f"RAG '{collection_name}' : {e}")
            return "", []

    def get_collections(self) -> List[str]:
        return [c.name for c in self.chroma.list_collections()]

    # ── Détection ─────────────────────────────────────────────────────────────

    def _detect_intent(self, message: str, has_image: bool) -> str:
        if has_image: return "diagnostic"
        m = message.lower()
        loc = ["garage","mécanicien","proche","près","trouver","cherche","chercher",
               "où","prestataire","entreprise","adresse","find","where","nearby",
               "lavage","laver","station","essence","vulcanisation","électricien",
               "climatisation","peinture","remorquage","dépannage","assurance",
               "louer","acheter voiture","acheter moto","permis","disponible",
               "inscrit","répertorié","liste","recommande","propose","quel garage"]
        if any(w in m for w in loc): return "localisation"
        diag = ["panne","bruit","fume","frein","voyant","problème","démarre","broken",
                "noise","fault","casse","coule","fuite","vibr","chauffe","grince",
                "claque","ne marche","marche plus","phare","moteur","batterie contact"]
        if any(w in m for w in diag): return "diagnostic"
        if any(w in m for w in ["entretien","vidange","changer","révision",
                                  "maintenance","huile","filtre"]): return "maintenance"
        return "info_generale"

    def _detect_domaine(self, message: str) -> Optional[str]:
        m = message.lower()
        for domaine, kws in DOMAINE_KEYWORDS.items():
            if any(kw in m for kw in kws): return domaine
        return None

    def _detect_ville(self, message: str) -> Optional[Tuple[str, float, float]]:
        m = message.lower()
        m_norm = (m.replace("è","e").replace("é","e").replace("ô","o")
                   .replace("à","a").replace("ê","e").replace("î","i")
                   .replace("-"," "))
        for lieu in sorted(LIEUX_BENIN.keys(), key=len, reverse=True):
            lieu_norm = lieu.replace("è","e").replace("é","e").replace("-"," ")
            if lieu_norm in m_norm or lieu in m:
                lat, lng = LIEUX_BENIN[lieu]
                return lieu.title(), lat, lng
        return None

    def _quick_identify(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        t = text.lower()
        for alias, (make, model) in VEHICLE_ALIASES.items():
            if alias in t: return make, model
        return None, None

    def _find_collection(self, make: str, model: str) -> Optional[str]:
        cols = self.get_collections()
        ms   = make.lower().replace(" ","_").replace("-","_")
        ml   = model.lower().replace(" ","_").replace("-","_").replace("/","_").replace("(","").replace(")","")
        for c in cols:
            if ms in c and (not ml or ml[:5] in c): return c
        return None

    # ── Récupération services — Laravel + Cache ───────────────────────────────

    def _get_services(self, message: str, intent: str, vehicle_model: str,
                      user_location: Optional[Dict]) -> Tuple[List[Dict], str]:
        """
        Récupère les services TOUJOURS :
        1. Depuis Laravel (si disponible)
        2. Sinon depuis le cache JSON local (entreprises_cache.json)
        Fonctionne avec GPS OU nom de ville dans le texte.
        """
        lat, lng, ville_nom = None, None, ""

        if user_location and user_location.get("lat"):
            lat, lng = float(user_location["lat"]), float(user_location["lng"])
            ville_nom = user_location.get("address", "")
        else:
            result = self._detect_ville(message)
            if result:
                ville_nom, lat, lng = result

        # Sans localisation → utiliser le centre du Bénin pour tout afficher
        if not lat:
            lat, lng = 9.3, 2.3
            ville_nom = "Bénin"
            radius_km = 1000
        else:
            radius_km = 100

        domaine = self._detect_domaine(message)

        # 1. Essai Laravel
        try:
            from laravel_bridge import get_nearby_services
            LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://localhost:8000/api")
            test = http_req.get(f"{LARAVEL_API_URL}/ai/domaines", timeout=3)
            if test.ok:
                services = get_nearby_services(lat=lat, lng=lng, radius_km=radius_km,
                                               domaine=domaine, limit=20)
                if services:
                    log.info(f"Laravel: {len(services)} services ({ville_nom}, domaine={domaine})")
                    return services, ville_nom
        except Exception as e:
            log.warning(f"Laravel hors ligne : {e}")

        # 2. Fallback cache local
        if self._cache:
            services = self._filter_cache(self._cache, lat, lng, radius_km, domaine)
            log.info(f"Cache local: {len(services)} services ({ville_nom}, domaine={domaine})")
            return services, ville_nom

        log.info("Aucune source de données disponible (Laravel hors ligne + pas de cache)")
        return [], ville_nom

    def _filter_cache(self, cache: List[Dict], lat: float, lng: float,
                      radius_km: float, domaine: Optional[str]) -> List[Dict]:
        """Filtre et trie le cache local par distance."""
        result = []
        for s in cache:
            ent = s.get("entreprise") or {}
            s_lat = s.get("latitude") or ent.get("latitude")
            s_lng = s.get("longitude") or ent.get("longitude")
            if not s_lat or not s_lng:
                continue
            try:
                dist = haversine(lat, lng, float(s_lat), float(s_lng))
            except Exception:
                continue
            if dist > radius_km:
                continue
            if domaine:
                s_domaine = str(s.get("domaine","")).lower()
                dom_lower = domaine.lower()
                if dom_lower not in s_domaine and s_domaine not in dom_lower:
                    continue
            s["distance_km"]    = dist
            s["distance_label"] = f"{dist} km"
            result.append(s)
        result.sort(key=lambda x: x.get("distance_km", 9999))
        return result[:20]

    # ── Format services pour le LLM ───────────────────────────────────────────

    def _format_services_for_llm(self, services: List[Dict], ville: str = "") -> str:
        if not services: return ""
        titre = f"DONNÉES CAREASY — {len(services)} ENTREPRISE(S)"
        if ville and ville != "Bénin": titre += f" PROCHE DE {ville.upper()}"
        lines = [f"\n{'━'*55}", titre, "━"*55]
        for i, s in enumerate(services, 1):
            ent      = s.get("entreprise") or {}
            name     = s.get("name") or ent.get("name", f"Entreprise {i}")
            dist     = s.get("distance_label") or f"{s.get('distance_km','?')} km"
            phone    = ent.get("call_phone")     or s.get("call_phone")     or "—"
            whatsapp = ent.get("whatsapp_phone") or s.get("whatsapp_phone") or "—"
            email    = ent.get("email") or s.get("email") or ""
            address  = (ent.get("google_formatted_address") or ent.get("address")
                        or s.get("address") or "—")
            google_ref = ent.get("google_place_id") or ent.get("place_url") or ""
            is_24h   = s.get("is_open_24h", False)
            start    = s.get("start_time","")
            end      = s.get("end_time","")
            hours    = "24h/24" if is_24h else (f"{start}–{end}" if start else "—")
            online   = "✅ En ligne" if ent.get("status_online") else "⭕ Hors ligne"
            domaine  = s.get("domaine","")
            price    = s.get("price","")
            desc     = s.get("descriptions","") or ""
            logo     = ent.get("logo","")

            lines.append(f"\n{i}. {name.upper()}")
            lines.append(f"   📏 Distance  : {dist}  {online}")
            if domaine:    lines.append(f"   🔧 Domaine   : {domaine}")
            lines.append(f"   📍 Adresse   : {address}")
            if google_ref: lines.append(f"   🗺️ Réf Google : {google_ref}")
            lines.append(f"   🕐 Horaires  : {hours}")
            lines.append(f"   📞 Tél       : {phone}")
            lines.append(f"   💬 WhatsApp  : {whatsapp}")
            if email:      lines.append(f"   📧 Email     : {email}")
            if price:      lines.append(f"   💰 Tarif     : {price}")
            if desc:       lines.append(f"   ℹ️ Info       : {desc[:200]}")
            lines.append("   " + "─"*45)
        lines.append("━"*55)
        return "\n".join(lines)

    # ── Vision / Photo ────────────────────────────────────────────────────────

    def analyze_photo(self, image_bytes: bytes, mime_type: str = "image/jpeg",
                      vehicle_make: str = "", vehicle_model: str = "",
                      user_description: str = "") -> Tuple[str, str]:
        vehicle = f"{vehicle_make} {vehicle_model}".strip() or "véhicule"

        # Chercher un modèle vision disponible
        vm = self._get_vision_model()
        if vm:
            b64    = base64.b64encode(image_bytes).decode("utf-8")
            prompt = (
                f"Tu es CAREASY, mécanicien expert au Bénin.\n"
                f"Véhicule : {vehicle}\n"
                f"Description : {user_description or 'Photo envoyée par l utilisateur'}\n\n"
                "Analyse précisément cette photo et donne :\n"
                "🔍 Ce que tu vois exactement (pièce, zone, dégât visible)\n"
                "🩺 Diagnostic probable (causes)\n"
                "⚡ Urgence : URGENT 🔴 / ATTENTION 🟡 / OK 🟢\n"
                "✅ Vérifications immédiates possibles sans outil\n"
                "🔧 Solutions étape par étape\n"
                "💰 Coût estimé en FCFA (marché béninois)\n"
                "⚠️ NE JAMAIS mentionner Google Maps\n"
                "Termine par : Akpé — CareEasy Bénin 🇧🇯"
            )
            try:
                r = http_req.post(f"{OLLAMA_URL}/api/generate",
                    json={"model": vm, "prompt": prompt,
                          "images": [b64], "stream": False}, timeout=180)
                if r.ok:
                    a = r.json().get("response","").strip()
                    log.info(f"Photo analysée avec {vm}")
                    return a, self._extract_urgency(a)
            except Exception as e:
                log.warning(f"Vision model {vm} : {e}")

        # Pas de modèle vision disponible
        desc = user_description or "problème signalé via photo"
        messages = [
            {"role": "system", "content": SYSTEM_DIAGNOSTIC},
            {"role": "user", "content": (
                f"Véhicule : {vehicle}\n"
                f"Problème décrit : {desc}\n\n"
                "📷 NOTE : Pour activer l'analyse de photos, installez llava :\n"
                "  ollama pull llava\n\n"
                "En attendant, donne un diagnostic complet basé sur la description."
            )},
        ]
        try:
            a = self._ollama(messages)
            return a, self._extract_urgency(a)
        except Exception:
            return (
                "📷 Photo reçue mais analyse visuelle non disponible.\n\n"
                "✅ Pour activer l'analyse de photos :\n```\nollama pull llava\n```\n\n"
                "En attendant, décrivez votre problème en texte.",
                "unknown"
            )

    def _get_vision_model(self) -> Optional[str]:
        try:
            r = http_req.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if r.ok:
                models = [m["name"] for m in r.json().get("models", [])]
                for vm in ["llava","bakllava","moondream","llava-phi3","minicpm-v"]:
                    if any(vm in m for m in models):
                        return next(m for m in models if vm in m)
        except Exception:
            pass
        return None

    # ── Chat principal ────────────────────────────────────────────────────────

    def chat(self, message=None, message_fr=None, image_bytes=None,
             image_mime="image/jpeg", collection_name=None,
             vehicle_make=None, vehicle_model=None,
             user_location=None, user_lang="fr", history=None) -> Dict[str, Any]:

        history = history or []
        query   = message_fr or message or ""

        # Identification véhicule
        if not collection_name and query:
            make, model = self._quick_identify(query)
            if make:
                vehicle_make    = vehicle_make  or make
                vehicle_model   = vehicle_model or model
                collection_name = self._find_collection(make, model)

        # Intention
        intent = self._detect_intent(query, image_bytes is not None)

        # RAG manuels
        manual_context, sources = self._rag(query, collection_name or "")

        # Services — TOUJOURS récupérés (Laravel ou cache)
        services_proches, ville_nom = self._get_services(
            query, intent, vehicle_model or "", user_location)

        # Génération
        answer_fr, urgency = self._generate(
            query=query, image_bytes=image_bytes, image_mime=image_mime,
            vehicle_make=vehicle_make or "", vehicle_model=vehicle_model or "",
            manual_context=manual_context, services_proches=services_proches,
            ville_nom=ville_nom, user_location=user_location,
            intent=intent, history=history,
        )

        return {
            "answer_fr": answer_fr, "intent": intent, "urgency": urgency,
            "services_proches": services_proches[:15],
            "vehicle": {"make": vehicle_make or "", "model": vehicle_model or "",
                        "collection_name": collection_name or ""},
            "sources": sources[:3], "lang": user_lang,
        }

    # ── Génération ────────────────────────────────────────────────────────────

    def _generate(self, query, image_bytes, image_mime, vehicle_make, vehicle_model,
                  manual_context, services_proches, ville_nom, user_location,
                  intent, history) -> Tuple[str, str]:

        if image_bytes:
            return self.analyze_photo(image_bytes, image_mime, vehicle_make, vehicle_model, query)

        system  = SYSTEM_DIAGNOSTIC if intent in ("diagnostic","maintenance") else SYSTEM_STRICT
        vehicle = f"{vehicle_make} {vehicle_model}".strip()

        # Construire le contexte
        parts = []
        if query:   parts.append(f"Question : {query}")
        if vehicle: parts.append(f"Véhicule : {vehicle}")
        if manual_context:
            parts.append(f"\n[MANUEL {vehicle}]\n{manual_context[:1500]}")

        # Données entreprises — TOUJOURS incluses
        services_txt = self._format_services_for_llm(services_proches, ville_nom)
        if services_txt:
            parts.append(services_txt)
        else:
            domaine = self._detect_domaine(query) or "ce domaine"
            lieu    = f" à {ville_nom}" if ville_nom and ville_nom != "Bénin" else ""
            parts.append(
                f"\n[BD CAREASY : Aucune entreprise trouvée pour '{domaine}'{lieu}.\n"
                "DIS clairement à l'utilisateur qu'aucune entreprise n'est encore enregistrée "
                "dans cette zone sur CareEasy. Invite-le à contacter support@careasy.bj "
                "ou à inscrire son entreprise. NE PAS mentionner Google Maps.]"
            )

        # Instruction finale ultra-stricte
        if services_proches:
            parts.append(
                "\n⚠️ INSTRUCTION FINALE OBLIGATOIRE : "
                "Liste TOUTES les entreprises ci-dessus avec TOUS leurs contacts. "
                "Ne mentionne JAMAIS Google Maps ou toute app externe. "
                "Termine par : Akpé — CareEasy Bénin 🇧🇯"
            )
        else:
            parts.append(
                "\n⚠️ INSTRUCTION : Dis qu'aucune entreprise n'est disponible sur CareEasy "
                "pour cette zone. Ne mentionne JAMAIS Google Maps. "
                "Propose : support@careasy.bj"
            )

        messages = [{"role": "system", "content": system}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": "\n\n".join(parts)})

        try:
            answer  = self._ollama(messages, temperature=0.05, max_tokens=2000)
            urgency = self._extract_urgency(answer)
            log.info(f"Réponse (intent={intent}, urgence={urgency}, {len(answer)} chars)")
            return answer, urgency
        except RuntimeError as e:
            return (f"⚠️ {str(e)}\n\nPour lancer Ollama : ollama serve && ollama pull qwen2.5:3b"), "unknown"
        except Exception as e:
            log.error(f"Erreur : {e}", exc_info=True)
            return "Erreur interne. Consultez les logs.", "unknown"

    def _extract_urgency(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["urgent","critique","immédiatement","ne pas rouler","dangereux","🔴"]):
            return "critical"
        if any(w in t for w in ["attention","important","48h","rapidement","🟡"]):
            return "important"
        if any(w in t for w in ["ok","mineur","peut attendre","normal","🟢"]):
            return "minor"
        return "unknown"


_agent: Optional[CareEasyAgent] = None

def get_agent() -> CareEasyAgent:
    global _agent
    if _agent is None:
        _agent = CareEasyAgent()
    return _agent