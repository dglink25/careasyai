import os, json, math, base64, logging
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

import requests as http_req
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "careasy")
CHROMA_DIR   = os.getenv("CHROMA_DIR",   "./chroma_db")
EMBED_MODEL  = os.getenv("EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CACHE_FILE   = os.getenv("CACHE_FILE", "entreprises_cache.json")

SYSTEM_CAREASY = """Tu es CarAI, l'assistant IA de la plateforme CareEasy au Bénin.

PERSONNALITÉ:
- Tu parles comme un ami mécanicien béninois compétent et chaleureux
- Ton langage est naturel, direct, professionnel mais pas froid
- Tu comprends le français béninois, le Fon, l'anglais, le Swahili, le Yoruba
- Tu réponds TOUJOURS dans la langue utilisée par l'interlocuteur

RÈGLES STRICTES:
1. JAMAIS d'emojis, de symboles spéciaux, ou de caractères decoratifs
2. JAMAIS de ** gras ** ou # titres markdown dans le texte
3. JAMAIS mentionner Google Maps, Apple Maps ou toute app externe
4. JAMAIS recommander de "chercher sur internet"
5. Toutes les entreprises viennent UNIQUEMENT de la base CareEasy
6. Réponses naturelles comme une vraie conversation

STYLE DES RÉPONSES:
- Court et direct pour les questions simples
- Structuré mais sans markdown pour les diagnostics
- Si quelqu'un dit juste "bonjour", répondre juste "Bonjour" naturellement
- Si des entreprises sont disponibles dans le contexte, les lister proprement

FORMAT DIAGNOSTIC (sans emojis ni markdown):
Diagnostic: [causes probables]
Urgence: URGENT / ATTENTION / OK
A verifier maintenant: [actions immédiates]
Solutions: [étapes simples vers complexe]
Cout estimé: [fourchette en FCFA, marché béninois]
[Si entreprises disponibles] Prestataires CareEasy: [liste]

FORMAT SERVICES:
[NOM DE L'ENTREPRISE] - [distance] km
Adresse: [adresse]
Tel: [numéro]
WhatsApp: [numéro]
Email: [si disponible]
Horaires: [horaires]
Services: [liste]
---"""

SYSTEM_GREETING = """Tu es CarAI, l'assistant de CareEasy au Bénin.
Réponds de façon naturelle et chaleureuse, comme un ami.
JAMAIS d'emojis. JAMAIS de ** ou de #. Réponses courtes et directes.
Réponds toujours dans la langue de l'utilisateur."""

VEHICLE_ALIASES = {
    "dayang":("Dayang","DY110-3"),"dayan":("Dayang","DY110-3"),
    "lifan":("Lifan","LF110"),"zemidjan":("Dayang","DY110-3"),
    "zémidjan":("Dayang","DY110-3"),"zem":("Dayang","DY110-3"),
    "keke":("Bajaj","RE (Keke)"),"kéké":("Bajaj","RE (Keke)"),
    "tricycle":("Bajaj","RE (Keke)"),"moto chinoise":("Dayang","DY110-3"),
    "110cc":("Dayang","DY110-3"),"125cc":("Dayang","DY125"),
    "hilux":("Toyota","Hilux"),"corolla":("Toyota","Corolla"),
    "land cruiser":("Toyota","Land Cruiser"),"hiace":("Toyota","HiAce"),
    "prado":("Toyota","Prado"),"206":("Peugeot","206"),
    "405":("Peugeot","405"),"pejo":("Peugeot","206"),
    "vespa":("Vespa","Primavera"),"yamaha":("Yamaha",""),
    "honda":("Honda",""),"suzuki":("Suzuki",""),
}

DOMAINE_KEYWORDS = {
    "Lavage automobile":           ["lavage","laver","wash","nettoyer","nettoyage"],
    "Station d'essence":           ["essence","carburant","fuel","station","gazoil"],
    "Garage mécanique":            ["mécanicien","mécanique","garage","réparer","réparation"],
    "Réparation moto":             ["moto","zemidjan","kéké","keke","deux-roues"],
    "Changement d'huile":          ["huile","vidange","oil"],
    "Pneumatique / vulcanisation": ["pneu","tyre","crevaison","vulcanisation"],
    "Électricien auto":            ["électricien","électrique","alternateur","câblage"],
    "Climatisation auto":          ["climatisation","clim","air conditionné"],
    "Peinture auto":               ["peinture","carrosserie","rayure"],
    "Tôlerie":                     ["tôlerie","tôle","cabossé","bosse"],
    "Dépannage / remorquage":      ["dépannage","remorquage","dépanneuse"],
    "Diagnostic automobile":       ["diagnostic","scanner","voyant","obd"],
    "Vente de pièces détachées":   ["pièces","pièce détachée","spare parts"],
    "Assurance automobile":        ["assurance","insurance"],
    "École de conduite":           ["permis","auto-école","conduire"],
}

# ─────────────────────────────────────────────────────────────────────────────
# 576 LIEUX DU BÉNIN (depuis benin_admin_final_complet.csv)
# ─────────────────────────────────────────────────────────────────────────────
# 576 lieux du Bénin générés depuis benin_admin_final_complet.csv
LIEUX_BENIN_CSV = {
    "10?me arrondissement": (6.379276, 2.394222),
    "11?me arrondissement": (6.364588, 2.407302),
    "12?me arrondissement": (6.352480, 2.391500),
    "13?me arrondissement": (6.369041, 2.377250),
    "1?re c.u.natitingou": (10.323011, 1.371059),
    "1?re com.de bohicon": (7.171550, 2.065993),
    "1?re com.de djougou": (9.761049, 1.653847),
    "1er arrondisement": (6.470533, 2.640279),
    "1er arrondissement": (6.339120, 2.070207),
    "2? arrondissement": (6.357496, 2.072616),
    "2?me arrondissement": (6.470282, 2.620616),
    "2?me c.u.natitingou": (10.342070, 1.435659),
    "2?me com.de bohicon": (7.169660, 2.074710),
    "2?me com.de djougou": (9.699268, 1.646884),
    "3? arrondissement": (6.387813, 2.096555),
    "3?me arrondissement": (6.468882, 2.602282),
    "3?me c.u.natitingou": (10.250108, 1.392279),
    "3?me com.de djougou": (9.745492, 1.724477),
    "4? arrondissement": (6.352428, 2.104360),
    "4?me arrondissement": (6.504595, 2.620256),
    "4?me c.u.natitingou": (10.363277, 1.319165),
    "5?me arrondissement": (6.493638, 2.594193),
    "6?me arrondissement": (6.387638, 2.423103),
    "7?me arrondissement": (6.365645, 2.421539),
    "8?me arrondissement": (6.375441, 2.410995),
    "9?me arrondissement": (6.394107, 2.384364),
    "abomey": (7.214741, 1.943015),
    "abomey-calavi": (6.426625, 2.242111),
    "adakplam?": (7.530464, 2.523554),
    "adanhondjigon": (7.098077, 2.024397),
    "adido": (8.044146, 2.474328),
    "adingningon": (7.128080, 2.036581),
    "adja-ou?r?": (7.023921, 2.586737),
    "adja-ouere": (7.150188, 2.570420),
    "adjaha": (6.343324, 1.827104),
    "adjahonm?": (7.061852, 1.792477),
    "adjan": (6.689786, 2.266670),
    "adjarra": (6.529509, 2.664518),
    "adjarra i": (6.529509, 2.664518),
    "adjarra ii": (6.520198, 2.679974),
    "adjido": (6.917295, 1.772841),
    "adjintimey": (6.813404, 1.668341),
    "adjohoun": (6.775209, 2.486871),
    "adogb?": (7.223116, 2.346661),
    "adohoun": (6.676989, 1.629579),
    "adoukandji": (6.832536, 1.958367),
    "affam?": (6.826655, 2.484097),
    "agam?": (6.714022, 1.754454),
    "aganmalom?": (6.475545, 2.048057),
    "agatogbo": (6.396843, 1.928600),
    "agbangnizoun": (7.132071, 1.922114),
    "agbanou": (6.684606, 2.102747),
    "agbanto": (6.383088, 1.959422),
    "agbodji": (6.672219, 1.973393),
    "agbopka": (7.219307, 1.997897),
    "aglangandan": (6.374855, 2.502909),
    "aglogb?": (6.464342, 2.677918),
    "agondji": (7.292662, 2.039647),
    "agongointo": (7.183545, 2.088886),
    "agonkanm?": (6.378179, 2.009927),
    "agonlin-hou?gbo": (7.232121, 2.415163),
    "agou?": (6.254043, 1.705133),
    "agoua": (8.274175, 1.998056),
    "agouna": (7.545659, 1.714373),
    "agu?": (6.815096, 2.095127),
    "aguegues": (6.432482, 2.515018),
    "aguidi": (6.819152, 2.695855),
    "ahodjinnako": (6.780461, 2.007981),
    "ahogbeya": (7.019315, 1.901184),
    "aholouy?m?": (6.426849, 2.574181),
    "ahom?-lokpo": (6.571338, 2.419722),
    "ahomad?gb?": (6.879984, 1.997241),
    "ahouanonzou": (6.716502, 2.195762),
    "ahoy?y?": (7.008344, 2.675295),
    "akassato": (6.536009, 2.370963),
    "akiza": (6.975060, 2.029510),
    "aklanpka": (8.342941, 2.207860),
    "akod?ha": (6.459035, 1.910909),
    "akofodjoul?": (7.732815, 2.411984),
    "akpadanou": (6.775209, 2.486871),
    "akpassi": (8.175245, 2.002437),
    "akpo-misserete": (6.620518, 2.574903),
    "akpro-miss?r?t?": (6.551829, 2.581579),
    "al?djo": (9.369189, 1.486566),
    "alafiarou": (8.923482, 2.327972),
    "allada": (6.686601, 2.042187),
    "allah?": (7.150933, 2.255237),
    "anandana": (9.921016, 1.386611),
    "angarad?bou": (11.422630, 3.020060),
    "aplahou?": (6.944613, 1.624412),
    "aplahoue": (7.318517, 1.696573),
    "assalin": (7.249079, 2.124076),
    "assant?": (8.155053, 2.252133),
    "atchannou": (6.499883, 1.755087),
    "atchonsa": (6.861868, 2.487792),
    "atchoukpa": (6.538710, 2.621876),
    "athi?m?": (6.588798, 1.685715),
    "athieme": (6.624256, 1.637350),
    "atocoligb?": (8.505137, 1.876935),
    "atogon": (6.736699, 2.161261),
    "atom?": (7.318517, 1.696573),
    "av?djin": (6.903225, 1.859555),
    "avagbodji": (6.509325, 2.525530),
    "avakpa": (6.659987, 2.036911),
    "avam?": (6.530166, 2.224146),
    "avl?k?t?": (6.340263, 2.202882),
    "avlam?": (7.098648, 2.181218),
    "avloh": (6.308009, 1.917621),
    "avogbanna": (7.215009, 2.086405),
    "avrankou": (6.603463, 2.649654),
    "awonou": (6.773647, 2.554224),
    "aya-hohou?": (6.962107, 1.823836),
    "ayomi": (6.775211, 1.716151),
    "ayou": (6.720935, 2.118943),
    "azohou?-aliho": (6.587159, 2.135353),
    "azohou?-kada": (6.568147, 2.077373),
    "azov?": (6.950061, 1.704667),
    "azowiliss?": (6.658288, 2.527316),
    "b?ll?foungou": (9.822143, 1.758570),
    "b?roubouay": (10.496250, 2.835603),
    "b?t?rou": (9.309261, 2.214035),
    "b?toumey": (6.941611, 1.731127),
    "badazoui": (6.733776, 1.962993),
    "badjoud?": (9.717982, 1.417062),
    "bagou": (10.894440, 2.521310),
    "banam?": (7.382043, 2.364941),
    "banigb?": (6.949751, 1.920642),
    "banikoara": (11.532811, 2.508627),
    "bant?": (8.397603, 1.896724),
    "bante": (8.509239, 1.713181),
    "barei": (9.689550, 1.535099),
    "bari?nou": (9.742286, 1.926043),
    "bassila": (9.369189, 1.486566),
    "basso": (10.534014, 3.661923),
    "bemb?r?k?": (10.185591, 2.650148),
    "bembereke": (10.496250, 2.835603),
    "beng?kou": (11.016485, 3.135066),
    "bess?": (7.790484, 2.605694),
    "birni": (10.038684, 1.628717),
    "birni-lafia": (11.966308, 3.207434),
    "biro": (9.883290, 2.880912),
    "bob?": (8.427172, 2.031923),
    "bogo-bogo": (12.109304, 3.083480),
    "bohicon": (7.256490, 2.086788),
    "boni": (8.035844, 2.502415),
    "bonou": (6.939790, 2.456333),
    "bopa": (6.690558, 1.903932),
    "bori": (9.784083, 2.351354),
    "bossito": (6.512986, 2.124628),
    "bouanri": (10.263529, 2.904701),
    "bougou": (9.446707, 1.597010),
    "bouka": (10.199599, 3.151322),
    "boukombe": (10.400974, 1.077821),
    "boukoumb?": (10.155310, 1.126856),
    "brignamaro": (10.624502, 2.051982),
    "calavi": (6.455872, 2.345630),
    "cana  i": (7.110836, 2.089334),
    "cana ii": (7.100651, 2.054836),
    "chabi-couma": (10.010054, 1.450160),
    "challa-ogoi": (8.412234, 2.593503),
    "cobly": (10.481566, 0.970396),
    "colli-agbam?": (6.803995, 2.144383),
    "com?": (6.402519, 1.887047),
    "come": (6.420027, 1.855150),
    "copargo": (9.900162, 1.630370),
    "cotiakou": (10.617643, 1.349633),
    "cotonou": (6.394107, 2.384364),
    "cove": (7.387901, 2.273827),
    "d?d?kpo?": (6.624256, 1.637350),
    "d?dom?": (6.569579, 2.020268),
    "d?kanm?": (6.488254, 2.486845),
    "d?kin": (6.553442, 2.455606),
    "d?kpo": (7.048009, 1.668841),
    "d?m?": (6.695139, 2.514047),
    "d?rassi": (10.181079, 3.435445),
    "d?tohonou": (7.214741, 1.943015),
    "d?v?": (6.747829, 1.652349),
    "daagb?": (6.562166, 2.722320),
    "dah?": (6.530653, 1.937127),
    "dam?": (6.839075, 2.244343),
    "dam?-wogon": (6.939790, 2.456333),
    "dan": (7.316628, 2.080409),
    "dangbo": (6.616740, 2.533936),
    "dassa i": (7.773977, 2.217025),
    "dassa ii": (7.804938, 2.131520),
    "dassa-zoume": (7.873711, 2.149438),
    "dassari": (10.798433, 1.149033),
    "dasso": (7.002983, 2.497697),
    "datori": (10.415434, 0.834215),
    "daw?": (6.723246, 2.236657),
    "dipoili": (10.284202, 0.946042),
    "dj?gb?": (8.298595, 2.423165),
    "dj?gbadji": (6.319806, 2.059109),
    "dj?r?gb?": (6.431558, 2.621337),
    "djakotomey": (6.943262, 1.760979),
    "djakotomey i": (6.874010, 1.717182),
    "djakotomey ii": (6.903101, 1.745769),
    "djaloukou": (7.673401, 1.876382),
    "djanglanm?": (6.761609, 2.068178),
    "djanglanmey": (6.394436, 1.801664),
    "djidja": (7.524160, 1.844424),
    "djigb?": (6.897930, 2.367372),
    "djomon": (6.579463, 2.625709),
    "djotto": (6.992317, 1.774719),
    "djougou": (9.940261, 1.915428),
    "dodji-bata": (6.661180, 2.285602),
    "dogbo": (6.824904, 1.793185),
    "dohouim?": (7.254157, 1.987350),
    "doko": (6.860652, 1.810161),
    "dom?": (7.079048, 2.313248),
    "don-tan": (7.283923, 2.398411),
    "donwari": (11.190873, 2.778706),
    "doum?": (7.975368, 1.734275),
    "doutou": (6.584756, 1.871673),
    "dovi-centre": (7.112210, 2.392095),
    "dunkassa": (10.392437, 3.128386),
    "ekp?": (6.386229, 2.547391),
    "firou": (10.929776, 1.891370),
    "fo-bour?": (10.049159, 2.332454),
    "fo-tanc?": (10.421787, 1.769587),
    "founougo": (11.532811, 2.508627),
    "gakp?": (6.431037, 2.148584),
    "gamia": (10.382793, 2.711437),
    "gangban": (6.644798, 2.455439),
    "ganvi? i": (6.416559, 2.418344),
    "ganvi? ii": (6.449297, 2.397933),
    "garou": (11.757804, 3.463197),
    "gb?con-hounli": (7.162578, 1.973365),
    "gb?gourou": (9.520435, 2.742457),
    "gb?hou?": (6.348499, 1.909867),
    "gb?ko": (6.599503, 2.457603),
    "gbaffo": (7.800538, 2.265201),
    "gbakpodji": (6.698284, 1.852529),
    "gbanlin": (8.607108, 2.294392),
    "gbozounm?": (6.574819, 2.672773),
    "glazou?": (7.975617, 2.247768),
    "glazoue": (8.342941, 2.207860),
    "glo-djib?": (6.543769, 2.300238),
    "gn?masson": (10.400807, 1.999543),
    "gnidjazoun": (7.192852, 2.050536),
    "gninagourou": (9.478316, 2.955698),
    "gninsy": (9.553020, 3.125728),
    "gnizounm?": (6.950362, 1.959297),
    "gnonkourakali": (10.062647, 2.953727),
    "gob?": (7.355810, 2.031253),
    "gobada": (7.785945, 2.021161),
    "godohou": (7.101032, 1.738604),
    "godomey": (6.379904, 2.308189),
    "gogounou": (10.894440, 2.521310),
    "gohomey": (6.844622, 1.739746),
    "gom?": (7.893773, 2.193565),
    "gom?-sota": (6.577775, 2.584530),
    "gomparou": (11.373734, 2.472598),
    "goro": (8.894325, 2.489591),
    "gouand?": (10.814251, 0.915523),
    "gouka": (8.117564, 1.926325),
    "goumori": (11.191965, 2.217515),
    "gounarou": (10.921634, 2.897252),
    "gounli": (7.201819, 2.322754),
    "grand-popo": (6.440503, 1.809072),
    "gu?n?": (11.587084, 3.179881),
    "guilmaro": (10.677664, 1.769258),
    "h?kanm?": (6.793124, 2.323378),
    "h?vi?": (6.426625, 2.242111),
    "hinvi": (6.768142, 2.184398),
    "hlassam?": (6.897743, 1.945867),
    "hondji": (7.007741, 1.827140),
    "honhou?": (6.516301, 1.896809),
    "honton": (6.740864, 1.751106),
    "honvi?": (6.510710, 2.646385),
    "hou?do-agu?kon": (6.511883, 2.456816),
    "hou?dogli": (6.924038, 1.800690),
    "hou?dom?": (6.471182, 2.537768),
    "hou?domey": (6.597931, 2.496052),
    "hou?gamey": (6.943262, 1.760979),
    "hou?gbo": (6.837195, 2.185567),
    "hou?ko": (7.387901, 2.273827),
    "hou?yogb?": (6.597550, 1.822470),
    "houeyogbe": (6.584756, 1.871673),
    "houin": (7.211686, 2.334650),
    "houinvigu?": (6.810813, 2.520563),
    "houngomey": (7.189835, 2.163234),
    "hozin": (6.550031, 2.545137),
    "idigny": (7.542479, 2.691152),
    "ifangni": (6.701366, 2.748858),
    "igana": (7.055191, 2.719492),
    "ikpinl?": (6.883672, 2.624238),
    "ina": (10.003561, 2.692950),
    "issaba": (7.124783, 2.652047),
    "ita-dj?bou": (6.830165, 2.600297),
    "k?mon": (8.566571, 2.500470),
    "k?r?": (7.873711, 2.149438),
    "k?rou": (10.845571, 2.162611),
    "k?ssounou": (6.561790, 2.506007),
    "k?tou": (7.370086, 2.614277),
    "kaboua": (8.239055, 2.624110),
    "kalal?": (10.403271, 3.447522),
    "kalale": (10.534014, 3.661923),
    "kandi": (11.422630, 3.020060),
    "kandi i": (11.231015, 2.911824),
    "kandi ii": (11.109257, 2.934487),
    "kandi iii": (11.144034, 2.986983),
    "karimama": (12.185038, 2.652083),
    "kassakou": (11.044667, 2.943065),
    "katagon": (6.617873, 2.600571),
    "kerou": (11.157693, 1.967624),
    "ketou": (7.542479, 2.691152),
    "kika": (9.240222, 2.914969),
    "kilibo": (8.529779, 2.670584),
    "kinkinhou?": (6.921108, 1.723037),
    "kinta": (7.070575, 2.003378),
    "kissamey": (7.014441, 1.720280),
    "klouekanm?": (6.984924, 1.857832),
    "klouekanme": (7.110783, 1.841374),
    "ko-koumolou": (6.622502, 2.669085),
    "koabagou": (11.157693, 1.967624),
    "kobli": (10.585351, 0.885412),
    "kod?": (6.736387, 2.484084),
    "kokey": (11.355160, 2.702415),
    "kokiborou*": (11.248131, 2.299370),
    "koko": (8.344379, 1.883534),
    "kokohou?": (6.867273, 1.763838),
    "kolokond?": (9.940261, 1.915428),
    "kompa": (11.916667, 2.598623),
    "kond?": (9.607960, 1.451127),
    "koronti?re": (10.259732, 1.021787),
    "kossoucoingou": (10.127194, 1.249474),
    "kotopounga": (10.314945, 1.518132),
    "kouaba": (10.248247, 1.281790),
    "kouand?": (10.311343, 1.760879),
    "kouandata": (10.136739, 1.363678),
    "kouande": (10.677664, 1.769258),
    "kouarfa": (10.554472, 1.545205),
    "koudo": (6.719328, 1.819010),
    "koudokpo?": (6.778768, 2.264769),
    "kountori": (10.368096, 0.928936),
    "koussi": (6.897165, 2.154862),
    "koussoukpa": (7.079547, 2.266655),
    "kouty": (6.603463, 2.649654),
    "kp?d?pko": (7.235711, 2.465050),
    "kpakpam?": (7.313456, 2.169301),
    "kpakpaza": (7.954978, 2.172576),
    "kpan?": (9.689155, 3.077959),
    "kpankou": (7.315418, 2.528684),
    "kpanroun": (6.668630, 2.376499),
    "kpataba": (8.051575, 1.964247),
    "kpingni": (7.728848, 2.118345),
    "kpinnou": (6.572032, 1.756709),
    "kpoba": (6.825535, 1.635405),
    "kpokissa": (7.000083, 2.356124),
    "kpom?": (6.913881, 2.312540),
    "kpomass?": (6.424507, 2.040794),
    "kpomasse": (6.506203, 2.045895),
    "kpoulou": (6.957931, 2.543371),
    "kpozoun": (7.213353, 2.116688),
    "l?ma": (7.867320, 2.301614),
    "lagb?": (6.696713, 2.690087),
    "lahotan": (8.018657, 2.081511),
    "lainta": (7.154188, 2.333992),
    "lalo": (6.981023, 1.956696),
    "laminou": (8.503380, 2.427084),
    "lanta": (7.110783, 1.841374),
    "libant?": (10.726736, 3.639193),
    "liboussou": (10.963720, 3.500483),
    "liss?zoun": (7.216915, 2.056118),
    "lissazounm?": (7.106864, 1.992384),
    "lissegan": (6.623687, 2.053387),
    "lobogo": (6.628946, 1.905736),
    "logozoh?": (7.895544, 2.049207),
    "lokogba": (6.851030, 1.886224),
    "lokogohou?": (6.787260, 1.839913),
    "lokossa": (6.719328, 1.819010),
    "lon agame": (6.712391, 2.052115),
    "lonkly": (7.163807, 1.725472),
    "lougba": (8.287479, 1.755527),
    "lougou": (11.156638, 3.412337),
    "m?d?djonou": (6.496498, 2.703650),
    "mad?cali": (11.584443, 3.424595),
    "madjr?": (6.803325, 1.888357),
    "magoumi": (8.010117, 2.215904),
    "malanhoui": (6.491774, 2.663418),
    "malanville": (11.858033, 3.158716),
    "manigri": (8.794697, 1.977718),
    "manta": (10.334942, 1.123314),
    "mass?": (7.150188, 2.570420),
    "massi": (6.984696, 2.206358),
    "mat?ri": (10.688812, 1.128687),
    "materi": (10.930699, 1.043677),
    "missinko": (6.932002, 1.845504),
    "monkpa": (7.937519, 2.090707),
    "monsey": (12.185038, 2.652083),
    "monsourou": (7.524160, 1.844424),
    "mougnon": (7.245254, 2.043537),
    "n'dahonta": (10.495462, 1.118254),
    "n'dali": (9.784083, 2.351354),
    "naogon": (7.238230, 2.349018),
    "natitingou": (10.314945, 1.518132),
    "natta": (10.227243, 1.144916),
    "nikki": (10.062647, 2.953727),
    "nodi": (10.601221, 1.046269),
    "odom?ta": (7.252448, 2.661652),
    "odougba": (8.514393, 2.410966),
    "of?": (8.082711, 2.434850),
    "oko-akar?": (6.928166, 2.665931),
    "okpara": (8.020662, 2.641363),
    "okpom?ta": (7.398715, 2.672432),
    "onklou": (9.492554, 2.048509),
    "oroukayo": (10.214000, 1.669224),
    "ottola": (8.127794, 1.698198),
    "ou?d?m?": (8.045887, 2.166094),
    "ou?d?m?-adja": (6.691013, 1.679550),
    "ou?do": (6.471788, 2.260904),
    "ou?nou": (9.979959, 3.434851),
    "ou?ss?": (8.470712, 2.463279),
    "ouak?": (9.663045, 1.427667),
    "ouake": (9.792287, 1.395719),
    "ouakp? daho": (6.331159, 2.009626),
    "ouanho": (6.529372, 2.644288),
    "ouara": (10.657894, 2.507592),
    "ouassaho": (7.154622, 2.058294),
    "ouenou": (9.780430, 2.639624),
    "ouesse": (8.683235, 2.535287),
    "ouidah": (6.340263, 2.202882),
    "ouinhi": (7.159089, 2.468204),
    "oumako": (6.420027, 1.855150),
    "ounet": (11.153875, 2.468048),
    "oungb?gam?": (7.267151, 2.021436),
    "outo": (7.609782, 1.669224),
    "p?bi?": (9.679088, 2.951895),
    "p?hunco": (10.168436, 1.944522),
    "p?l?bina": (9.540948, 1.638928),
    "p?n?ssoulou": (9.289508, 1.720891),
    "p?onga": (10.445773, 3.285464),
    "p?r?r?": (9.790383, 3.009214),
    "pab?gou": (9.809292, 1.597588),
    "pahou": (6.386242, 2.192729),
    "paouingnan": (7.597252, 2.291107),
    "parakou": (9.366779, 2.534802),
    "passagon": (7.256490, 2.086788),
    "patargo": (9.547226, 1.914110),
    "pehunco": (10.271830, 2.149743),
    "perere": (9.790383, 3.009214),
    "perma": (10.122855, 1.486607),
    "pira": (8.509239, 1.713181),
    "pob?": (6.980550, 2.664607),
    "pobe": (7.124783, 2.652047),
    "porto-novo": (6.504595, 2.620256),
    "possotom?": (6.540464, 1.965825),
    "s?": (6.767956, 2.117698),
    "s?dj?-d?nou": (6.714585, 2.395440),
    "s?dj?-hou?goudo": (6.795329, 2.395592),
    "s?gbana": (10.948415, 3.668176),
    "s?gbeya": (6.506203, 2.045895),
    "s?gbohou?": (6.440624, 1.974065),
    "s?hou?": (6.906343, 2.254063),
    "s?houn": (7.214139, 2.020980),
    "s?k?r?": (10.492931, 2.435708),
    "s?kou": (6.633604, 2.224961),
    "s?m?-kpodji": (6.381069, 2.610460),
    "s?m?r? i": (9.545410, 1.461351),
    "s?m?r? ii": (9.469815, 1.441694),
    "s?r?kal?": (9.980692, 3.111371),
    "s?rou": (9.617557, 1.770720),
    "saah": (11.199834, 3.125931),
    "saclo": (7.144842, 2.091217),
    "sado": (6.544108, 2.689646),
    "sagon": (7.159089, 2.468204),
    "sah?": (7.080978, 1.923900),
    "sak?t? i": (6.754562, 2.646506),
    "sak?t? ii": (6.747052, 2.625010),
    "sakete": (6.819152, 2.695855),
    "sakin": (7.965755, 2.498756),
    "sam": (10.983106, 2.741965),
    "sanson": (9.287228, 2.423010),
    "sav? plateau": (8.004916, 2.482873),
    "savalou": (8.127794, 1.698198),
    "savalou--agbado": (7.811866, 1.953169),
    "savalou-aga": (7.860245, 1.884945),
    "savalou-attak?": (7.965985, 1.986939),
    "save": (8.239055, 2.624110),
    "savi": (6.412549, 2.107775),
    "sazu?": (6.440503, 1.809072),
    "segbana": (11.156638, 3.412337),
    "seme-kpodji": (6.407636, 2.676761),
    "setto": (7.459233, 2.097846),
    "sikki": (10.200275, 2.371798),
    "sinend?": (10.380768, 2.323119),
    "sinende": (10.492931, 2.435708),
    "singr?": (9.893937, 1.473841),
    "sirarou": (9.553116, 2.623277),
    "siw?-kpota": (7.032087, 1.955588),
    "siw?-l?go": (7.042844, 1.986948),
    "so-ava": (6.488254, 2.486845),
    "soclogbo": (7.776824, 2.323109),
    "sodohom?": (7.150739, 2.143038),
    "sokotindji": (10.740427, 3.329711),
    "sokouhou?": (6.874042, 1.661794),
    "sokponta": (7.867188, 2.236705),
    "soli": (7.206232, 2.348918),
    "somp?r?kou": (11.259985, 2.566372),
    "sonsoro": (11.106771, 2.726188),
    "sontou": (9.740144, 2.854218),
    "sori": (10.638954, 2.940437),
    "soroko": (11.469058, 2.258908),
    "suya": (9.851073, 3.054676),
    "tabota": (10.400974, 1.077821),
    "taiakou": (10.523784, 1.188165),
    "takon": (6.654374, 2.620150),
    "tamp?gr?": (10.438017, 1.316638),
    "tangbo-dj?vi?": (6.590272, 2.262593),
    "tangui?ta": (10.644938, 1.264478),
    "tanguieta": (11.100502, 1.476278),
    "tannou-gola": (6.883730, 1.788095),
    "tanongou": (11.100502, 1.476278),
    "tant?ga": (10.930699, 1.043677),
    "tanv?": (7.132071, 1.922114),
    "tanw?-hessou": (7.037100, 2.081150),
    "tapoga": (10.585351, 0.885412),
    "tasso": (9.745715, 3.225783),
    "tatonnoukon": (6.914096, 2.563191),
    "tchaada": (6.591892, 2.689325),
    "tchalinga": (9.792287, 1.395719),
    "tchanhoun-cossi": (10.691266, 0.972591),
    "tchaorou": (8.866126, 2.652144),
    "tchaourou": (9.309261, 2.214035),
    "tchatchou": (9.115640, 2.621986),
    "tchetti": (7.816770, 1.727521),
    "tchikp?": (6.962577, 1.890039),
    "tchito": (6.893963, 2.063604),
    "tchoumi-tchoumi": (10.042353, 1.355923),
    "thio": (8.010567, 2.315452),
    "tobr?": (10.271830, 2.149743),
    "toffo": (6.761609, 2.068178),
    "togba": (6.468172, 2.299242),
    "togbota": (6.676931, 2.449806),
    "togoudo": (6.647982, 2.175152),
    "tohou": (6.827280, 2.023593),
    "tohou?": (7.010300, 2.432940),
    "tokpa dom?": (6.494176, 1.997667),
    "tokpa-ava": (6.686601, 2.042187),
    "tori-bossito": (6.568147, 2.077373),
    "tori-gare": (6.480095, 2.191527),
    "tori-kada": (6.578005, 2.190384),
    "totchangni": (6.824904, 1.793185),
    "toucountouna": (10.554472, 1.545205),
    "toucoutouna": (10.539594, 1.398253),
    "toui": (8.683235, 2.535287),
    "toumbouctou": (11.858033, 3.158716),
    "toura": (11.266430, 2.353713),
    "toviklin": (6.932002, 1.845504),
    "tow?": (7.155720, 2.728929),
    "tr?": (7.709542, 2.234471),
    "vakon": (6.524571, 2.569970),
    "vekky": (6.456385, 2.446448),
    "vidol?": (7.179485, 1.966013),
    "y?godo?": (6.690558, 1.903932),
    "yoko": (6.707493, 2.597084),
    "yokpo": (6.620978, 2.302370),
    "z?ko": (7.227805, 2.107171),
    "za-kpota": (7.313456, 2.169301),
    "za-tanta": (7.273461, 2.191901),
    "zaff?": (7.922240, 2.260065),
    "zagnanado": (7.382043, 2.364941),
    "zalli": (6.981023, 1.956696),
    "ze": (6.590272, 2.262593),
    "zinvi?": (6.618871, 2.373016),
    "zogba": (7.227150, 2.321543),
    "zogbodomey": (7.100651, 2.054836),
    "zougou-pantrossi": (10.809177, 3.046855),
    "zoukou": (7.302535, 1.998401),
    "zoungam?": (6.432482, 2.515018),
    "zoungbom?": (6.620518, 2.574903),
    "zoungbonou": (6.559086, 1.814040),
    "zoungoudo": (7.012868, 1.966099),
    "zoungu?": (6.616740, 2.533936),
    "zounzonm?": (7.142290, 1.999291),
}
def haversine(lat1, lng1, lat2, lng2) -> float:
    R = 6371
    dlat = math.radians(lat2-lat1)
    dlng = math.radians(lng2-lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlng/2)**2
    return round(R*2*math.asin(math.sqrt(a)), 1)


import unicodedata as _uni
def _norm(s):
    s = s.lower().strip()
    r = ""
    for c in s:
        try:
            r += _uni.normalize('NFD', c).encode('ascii','ignore').decode('ascii') or c
        except:
            r += c
    return r


class CareEasyAgent:

    def __init__(self):
        log.info("Initialisation CareEasyAgent v5...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
        self._cache: List[Dict] = self._load_cache()
        self._check_ollama()
        log.info(f"CareEasyAgent v5 prêt — {len(self._cache)} entreprises en cache")

    def _load_cache(self) -> List[Dict]:
        if Path(CACHE_FILE).exists():
            try:
                d = json.loads(Path(CACHE_FILE).read_text(encoding="utf-8"))
                log.info(f"Cache: {len(d)} entreprises chargées")
                return d
            except Exception as e:
                log.warning(f"Cache corrompu: {e}")
        return []

    def reload_cache(self):
        self._cache = self._load_cache()

    def _check_ollama(self):
        try:
            r = http_req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if r.ok:
                models = [m["name"] for m in r.json().get("models", [])]
                log.info(f"Ollama OK — modèles: {models}")
                base = OLLAMA_MODEL.split(":")[0]
                if not any(base in m for m in models):
                    log.warning(f"Modèle '{OLLAMA_MODEL}' absent. Créez-le: ollama create careasy -f Modelfile")
        except Exception:
            log.warning("Ollama hors ligne. Lancez: ollama serve")

    def _ollama(self, messages: List[Dict], temperature: float = 0.1,
                max_tokens: int = 1500) -> str:
        for model in [OLLAMA_MODEL, "qwen2.5:3b", "mistral", "llama3.2"]:
            try:
                r = http_req.post(f"{OLLAMA_URL}/api/chat",
                    json={"model": model, "messages": messages, "stream": False,
                          "options": {"temperature": temperature, "num_predict": max_tokens}},
                    timeout=180)
                if r.ok:
                    return r.json()["message"]["content"].strip()
            except http_req.exceptions.ConnectionError:
                raise RuntimeError("Ollama hors ligne. Lancez: ollama serve")
            except Exception:
                continue
        raise RuntimeError("Aucun modèle disponible")

    def _embed(self, text: str) -> List[float]:
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def index_document(self, chunks: List[str], collection_name: str,
                       metadatas: Optional[List[Dict]] = None) -> int:
        if not chunks: return 0
        col = self.chroma.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})
        embs  = [self._embed(c) for c in chunks]
        ids   = [f"{collection_name}_{i}" for i in range(len(chunks))]
        metas = metadatas or [{"source": collection_name}] * len(chunks)
        col.upsert(ids=ids, embeddings=embs, documents=chunks, metadatas=metas)
        return len(chunks)

    def _rag(self, query: str, collection_name: str, k: int = 4) -> Tuple[str, List[str]]:
        if not query or not collection_name: return "", []
        try:
            col = self.chroma.get_collection(collection_name)
            res = col.query(query_embeddings=[self._embed(query)], n_results=min(k, col.count()))
            chunks = res["documents"][0] if res["documents"] else []
            return "\n\n---\n\n".join(chunks[:4]), chunks
        except Exception: return "", []

    def get_collections(self) -> List[str]:
        return [c.name for c in self.chroma.list_collections()]

    # ─── Détection contexte ────────────────────────────────────────────────

    def _detect_intent(self, message: str, has_image: bool) -> str:
        if has_image: return "diagnostic"
        m = message.lower()
        # Salutations simples — ne pas chercher d'entreprises
        greet_words = ["bonjour","bonsoir","salut","hello","hi ","yo ","comment tu vas",
                       "comment ca va","ca va","ça va","akpe","gbé","bon matin"]
        if any(w in m for w in greet_words) and len(m) < 60:
            return "salutation"
        loc = ["garage","mécanicien","proche","près","trouver","cherche","chercher",
               "où","prestataire","entreprise","adresse","find","where","nearby",
               "lavage","laver","station","essence","vulcanisation","électricien",
               "climatisation","peinture","remorquage","dépannage","assurance",
               "louer","permis","disponible","inscrit","répertorié","liste"]
        if any(w in m for w in loc): return "localisation"
        diag = ["panne","bruit","fume","frein","voyant","problème","démarre","broken",
                "noise","fault","casse","coule","fuite","vibr","chauffe","grince",
                "claque","ne marche","marche plus","phare","moteur","contact","klaxon"]
        if any(w in m for w in diag): return "diagnostic"
        if any(w in m for w in ["entretien","vidange","changer","révision",
                                  "maintenance","huile","filtre"]): return "maintenance"
        demo = ["montre","démonstration","démo","comment faire","tutorial","tuto",
                "étapes","apprends","apprendre","how to","show me","watch","vidéo",
                "video","voir comment","explique comment"]
        if any(w in m for w in demo): return "demonstration"
        return "info_generale" 

    def _detect_domaine(self, message: str) -> Optional[str]:
        m = message.lower()
        for domaine, kws in DOMAINE_KEYWORDS.items():
            if any(kw in m for kw in kws): return domaine
        return None

    def _detect_ville(self, message: str) -> Optional[Tuple[str, float, float]]:
        m_norm = _norm(message)
        for lieu in sorted(LIEUX_BENIN_CSV.keys(), key=len, reverse=True):
            if lieu in m_norm:
                lat, lng = LIEUX_BENIN_CSV[lieu]
                return lieu.title(), lat, lng
        return None

    def _quick_identify(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        t = text.lower()
        for alias, (make, model) in VEHICLE_ALIASES.items():
            if alias in t: return make, model
        return None, None

    def _find_collection(self, make: str, model: str) -> Optional[str]:
        cols = self.get_collections()
        ms = make.lower().replace(" ","_").replace("-","_")
        ml = model.lower().replace(" ","_").replace("-","_").replace("/","_").replace("(","").replace(")","")
        for c in cols:
            if ms in c and (not ml or ml[:5] in c): return c
        return None

    # ─── Services ─────────────────────────────────────────────────────────

    def _get_services(self, message: str, intent: str, vehicle_model: str,
                      user_location: Optional[Dict]) -> Tuple[List[Dict], str]:
        # Pas de services pour les salutations ou info_generale sans localisation
        if intent in ("salutation",): return [], ""

        lat, lng, ville_nom = None, None, ""
        if user_location and user_location.get("lat"):
            lat, lng = float(user_location["lat"]), float(user_location["lng"])
            ville_nom = user_location.get("address","")
        else:
            result = self._detect_ville(message)
            if result: ville_nom, lat, lng = result

        # Sans localisation explicite + pas de demande directe → pas de services
        if not lat and intent not in ("localisation",): return [], ""
        if not lat:
            lat, lng, ville_nom = 9.3, 2.3, "Bénin"
            radius_km = 500
        else:
            radius_km = 80

        domaine = self._detect_domaine(message)

        # Essai Laravel
        try:
            from laravel_bridge import get_nearby_services
            LARAVEL_API_URL = os.getenv("LARAVEL_API_URL","http://localhost:8000/api")
            test = http_req.get(f"{LARAVEL_API_URL}/ai/domaines", timeout=2)
            if test.ok:
                services = get_nearby_services(lat=lat, lng=lng, radius_km=radius_km,
                                               domaine=domaine, limit=15)
                if services: return services, ville_nom
        except Exception as e:
            log.debug(f"Laravel: {e}")

        # Cache local
        if self._cache:
            services = self._filter_cache(self._cache, lat, lng, radius_km, domaine)
            return services, ville_nom

        return [], ville_nom

    def _filter_cache(self, cache: List[Dict], lat: float, lng: float,
                      radius_km: float, domaine: Optional[str]) -> List[Dict]:
        result = []
        for s in cache:
            ent = s.get("entreprise") or {}
            s_lat = s.get("latitude") or ent.get("latitude")
            s_lng = s.get("longitude") or ent.get("longitude")
            if not s_lat or not s_lng: continue
            try: dist = haversine(lat, lng, float(s_lat), float(s_lng))
            except: continue
            if dist > radius_km: continue
            if domaine:
                sd = str(s.get("domaine","")).lower()
                dl = domaine.lower()
                if dl not in sd and sd not in dl: continue
            s["distance_km"] = dist
            s["distance_label"] = f"{dist} km"
            result.append(s)
        result.sort(key=lambda x: x.get("distance_km",9999))
        return result[:15]

    def _format_services_text(self, services: List[Dict], ville: str = "") -> str:
        """Format propre, sans emojis, pour injection dans le prompt."""
        if not services: return ""
        n = len(services)
        header = f"{n} entreprise(s) CareEasy trouvée(s)"
        if ville and ville != "Bénin": header += f" près de {ville}"
        lines = [header, ""]
        for i, s in enumerate(services, 1):
            ent      = s.get("entreprise") or {}
            name     = s.get("name") or ent.get("name", f"Entreprise {i}")
            dist     = s.get("distance_label") or f"{s.get('distance_km','?')} km"
            phone    = ent.get("call_phone")    or s.get("call_phone")    or "Non renseigné"
            whatsapp = ent.get("whatsapp_phone") or s.get("whatsapp_phone") or "Non renseigné"
            email    = ent.get("email") or s.get("email") or ""
            address  = (ent.get("google_formatted_address") or ent.get("address")
                        or s.get("address") or "Non renseignée")
            google_ref = ent.get("google_place_id") or ent.get("place_url") or ""
            is_24h   = s.get("is_open_24h", False)
            start, end = s.get("start_time",""), s.get("end_time","")
            hours    = "Ouvert 24h/24" if is_24h else (f"{start}-{end}" if start else "Non précisé")
            online   = "En ligne" if ent.get("status_online") else "Hors ligne"
            domaine  = s.get("domaine","")
            price    = s.get("price","")
            desc     = (s.get("descriptions","") or "")[:150]
            lat_e    = s.get("latitude") or ent.get("latitude","")
            lng_e    = s.get("longitude") or ent.get("longitude","")

            lines.append(f"{i}. {name.upper()} - {dist} ({online})")
            if domaine:    lines.append(f"   Domaine: {domaine}")
            lines.append(f"   Adresse: {address}")
            if google_ref: lines.append(f"   Ref Google: {google_ref}")
            if lat_e and lng_e: lines.append(f"   GPS: {lat_e},{lng_e}")
            lines.append(f"   Horaires: {hours}")
            lines.append(f"   Tel: {phone}")
            lines.append(f"   WhatsApp: {whatsapp}")
            if email:      lines.append(f"   Email: {email}")
            if price:      lines.append(f"   Tarif: {price}")
            if desc:       lines.append(f"   Info: {desc}")
            lines.append("")
        return "\n".join(lines)

    # ─── Vision ───────────────────────────────────────────────────────────

    def analyze_photo(self, image_bytes: bytes, mime_type: str = "image/jpeg",
                      vehicle_make: str = "", vehicle_model: str = "",
                      user_description: str = "") -> Tuple[str, str]:
        vehicle = f"{vehicle_make} {vehicle_model}".strip() or "vehicule"
        vm      = self._get_vision_model()
        if vm:
            b64    = base64.b64encode(image_bytes).decode("utf-8")
            prompt = (
                f"Tu es un mécanicien expert au Bénin. Véhicule: {vehicle}.\n"
                f"Description: {user_description or 'photo envoyée'}.\n"
                "Analyse cette photo et réponds SANS emojis ni markdown:\n"
                "Ce que tu vois: [description]\n"
                "Diagnostic: [causes]\n"
                "Urgence: URGENT / ATTENTION / OK\n"
                "Solutions: [étapes]\n"
                "Cout FCFA: [estimation]"
            )
            try:
                r = http_req.post(f"{OLLAMA_URL}/api/generate",
                    json={"model": vm, "prompt": prompt, "images": [b64], "stream": False},
                    timeout=180)
                if r.ok:
                    a = r.json().get("response","").strip()
                    return a, self._extract_urgency(a)
            except Exception as e:
                log.warning(f"Vision: {e}")

        # Fallback
        desc = user_description or "problème visible"
        messages = [
            {"role": "system", "content": SYSTEM_CAREASY},
            {"role": "user", "content": (
                f"Véhicule: {vehicle}\nProblème décrit: {desc}\n\n"
                "Note: Installez llava pour l'analyse photo: ollama pull llava\n"
                "Donne quand même un diagnostic basé sur la description."
            )},
        ]
        try:
            a = self._ollama(messages)
            return a, self._extract_urgency(a)
        except Exception:
            return "Photo reçue. Pour analyser les photos, installez: ollama pull llava", "unknown"

    def _get_vision_model(self) -> Optional[str]:
        try:
            r = http_req.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if r.ok:
                models = [m["name"] for m in r.json().get("models", [])]
                for vm in ["llava","bakllava","moondream","llava-phi3"]:
                    if any(vm in m for m in models):
                        return next(m for m in models if vm in m)
        except: pass
        return None

    # ─── Chat principal ────────────────────────────────────────────────────

    def chat(self, message=None, message_fr=None, image_bytes=None,
             image_mime="image/jpeg", collection_name=None,
             vehicle_make=None, vehicle_model=None,
             user_location=None, user_lang="fr", history=None) -> Dict[str, Any]:

        history = history or []
        query   = message_fr or message or ""

        if not collection_name and query:
            make, model = self._quick_identify(query)
            if make:
                vehicle_make    = vehicle_make  or make
                vehicle_model   = vehicle_model or model
                collection_name = self._find_collection(make, model)

        intent = self._detect_intent(query, image_bytes is not None)
        manual_context, sources = self._rag(query, collection_name or "")

        # Services uniquement si pertinent
        services_proches, ville_nom = [], ""
        if intent not in ("salutation",):
            services_proches, ville_nom = self._get_services(
                query, intent, vehicle_model or "", user_location)

        answer_fr, urgency = self._generate(
            query=query, image_bytes=image_bytes, image_mime=image_mime,
            vehicle_make=vehicle_make or "", vehicle_model=vehicle_model or "",
            manual_context=manual_context, services_proches=services_proches,
            ville_nom=ville_nom, user_location=user_location,
            intent=intent, history=history,
        )

        # Recherche YouTube si demande de démonstration
        video_url = None
        if intent == "demonstration":
            video_url = self._find_youtube_demo(query)

        return {
            "answer_fr": answer_fr, "intent": intent, "urgency": urgency,
            "services_proches": services_proches[:15],
            "vehicle": {"make": vehicle_make or "", "model": vehicle_model or "",
                        "collection_name": collection_name or ""},
            "sources": sources[:3], "lang": user_lang,
            "video_url": video_url,   # URL YouTube démo (ou None)
        }

    def _find_youtube_demo(self, query: str) -> Optional[str]:
        """
        Retourne l'URL YouTube la plus pertinente pour une demande de démonstration.
        Utilise une base de vidéos pré-indexées (pas d'API key requise).
        """
        # Base de vidéos tutoriels automobiles en français — indexée manuellement
        # Format: (mots-clés, url_youtube)
        DEMO_VIDEOS = [
            # Vidange
            (["vidange","huile","oil change"], "https://www.youtube.com/watch?v=0danHFd5HoI"),
            # Changement pneu
            (["pneu","roue","tire","crevaison","changer roue"], "https://www.youtube.com/watch?v=KZKKRMGlZN4"),
            # Batterie
            (["batterie","battery","démarrer","démarrage"], "https://www.youtube.com/watch?v=6VVHhiWLhpI"),
            # Frein
            (["frein","plaquette","brake","disque frein"], "https://www.youtube.com/watch?v=rmxDm8_rdGE"),
            # Filtre à air
            (["filtre air","air filter","filtre à air"], "https://www.youtube.com/watch?v=N6LlDnl2oVo"),
            # Bougie
            (["bougie","spark plug","allumage"], "https://www.youtube.com/watch?v=1bJfGrPBJGM"),
            # Climatisation
            (["clim","climatisation","ac","air conditionné"], "https://www.youtube.com/watch?v=YmrPc_XXUU4"),
            # Diagnostic OBD
            (["obd","scanner","voyant","diagnostic"], "https://www.youtube.com/watch?v=tHl5H8A8jkc"),
            # Lavage
            (["laver","lavage","nettoyer","wash"], "https://www.youtube.com/watch?v=7KxNiNnm3bw"),
            # Courroie
            (["courroie","distribution","timing belt"], "https://www.youtube.com/watch?v=4qH-dlB0-vs"),
        ]
        q = query.lower()
        for keywords, url in DEMO_VIDEOS:
            if any(kw in q for kw in keywords):
                log.info(f"Vidéo démo trouvée: {url}")
                return url

        # Aucune vidéo spécifique → retourner une vidéo générale maintenance auto
        return "https://www.youtube.com/watch?v=0danHFd5HoI"

    def _generate(self, query, image_bytes, image_mime, vehicle_make, vehicle_model,
                  manual_context, services_proches, ville_nom, user_location,
                  intent, history) -> Tuple[str, str]:

        if image_bytes:
            return self.analyze_photo(image_bytes, image_mime, vehicle_make, vehicle_model, query)

        # Salutation simple → réponse courte sans données
        if intent == "salutation":
            system = SYSTEM_GREETING
            messages = [{"role": "system", "content": system}]
            messages.extend(history[-2:])
            messages.append({"role": "user", "content": query})
            try:
                answer = self._ollama(messages, temperature=0.3, max_tokens=150)
                return answer, "unknown"
            except Exception as e:
                return f"Ollama hors ligne: {e}", "unknown"

        system = SYSTEM_CAREASY
        vehicle = f"{vehicle_make} {vehicle_model}".strip()
        parts = []

        if query:   parts.append(f"Question: {query}")
        if vehicle: parts.append(f"Véhicule: {vehicle}")
        if manual_context: parts.append(f"\n[Manuel {vehicle}]\n{manual_context[:1500]}")

        # Services en contexte
        services_txt = self._format_services_text(services_proches, ville_nom)
        if services_txt:
            parts.append(f"\n[DONNÉES BASE CAREASY]\n{services_txt}")
            parts.append(
                "INSTRUCTION: Présente les entreprises ci-dessus avec tous leurs "
                "contacts (tel, whatsapp, adresse, horaires). "
                "Sans emojis. Sans markdown. Sans Google Maps."
            )
        elif intent == "localisation":
            domaine = self._detect_domaine(query) or "ce service"
            lieu = f" à {ville_nom}" if ville_nom and ville_nom != "Bénin" else ""
            parts.append(
                f"[RÉSULTAT CAREASY: Aucune entreprise trouvée pour '{domaine}'{lieu}.\n"
                "Dis-le clairement et propose: support@careasy.bj ou inscriptions sur la plateforme]"
            )

        messages = [{"role": "system", "content": system}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": "\n\n".join(parts)})

        try:
            answer  = self._ollama(messages, temperature=0.1, max_tokens=1500)
            urgency = self._extract_urgency(answer)
            return answer, urgency
        except RuntimeError as e:
            return f"Serveur IA hors ligne. Lancez: ollama serve && ollama pull qwen2.5:3b", "unknown"
        except Exception as e:
            log.error(f"Erreur: {e}", exc_info=True)
            return "Erreur interne. Consultez les logs.", "unknown"

    def _extract_urgency(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["urgent","critique","immédiatement","ne pas rouler","dangereux"]):
            return "critical"
        if any(w in t for w in ["attention","important","48h","rapidement"]):
            return "important"
        if any(w in t for w in ["ok","mineur","peut attendre","normal"]):
            return "minor"
        return "unknown"


_agent: Optional[CareEasyAgent] = None

def get_agent() -> CareEasyAgent:
    global _agent
    if _agent is None:
        _agent = CareEasyAgent()
    return _agent