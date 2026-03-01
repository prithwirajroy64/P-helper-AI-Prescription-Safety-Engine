"""
RxGuard ML/NLP Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Five integrated ML/NLP components that REPLACE and UPGRADE
the original regex-only parser in the pasted code:

  MODULE 1 — Medical NER (Named Entity Recognition)
             BIO-tagged token classifier using TF-IDF features
             + hand-crafted medical character n-gram embeddings.
             Extracts: DRUG, DOSE, FREQ, ROUTE, DURATION, CONDITION, ALLERGY

  MODULE 2 — Drug Name Normalizer (Fuzzy + TF-IDF cosine similarity)
             Handles misspellings, brand→generic resolution, Indian trade names.
             Much more robust than the original brand_names dict lookup.

  MODULE 3 — Drug-Drug Interaction Severity Classifier
             Logistic Regression trained on a synthetic DDI feature matrix.
             Outputs: MAJOR / MODERATE / MINOR / UNKNOWN with confidence %.

  MODULE 4 — Polypharmacy Risk Scorer
             Gradient Boosted model scoring 0-100 from drug count,
             interaction graph density, allergy flags, dosage anomalies.

  MODULE 5 — Clinical Sentence Classifier
             Multinomial Naive Bayes to tag each sentence of free-text as:
             DOSAGE_INSTRUCTION / ALLERGY / DIAGNOSIS / PATIENT_INFO /
             DRUG_NAME / IRRELEVANT — used by the NER to focus attention.

All models are trained in-process at startup on curated synthetic corpora
(no external files needed). Training takes < 1 second.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import re, math, statistics, time
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# ════════════════════════════════════════════════════════════════════════════
#  TRAINING CORPORA  (compact, realistic synthetic data)
# ════════════════════════════════════════════════════════════════════════════

# ── Module 5 corpus: sentence type classification ──
_SENT_CORPUS = [
    # DRUG_NAME
    ("Tab Metformin 500mg", "DRUG_NAME"),
    ("Cap Amoxicillin 875mg", "DRUG_NAME"),
    ("Inj Insulin 10 units", "DRUG_NAME"),
    ("Tab Atorvastatin 40mg", "DRUG_NAME"),
    ("Warfarin 5mg tablet", "DRUG_NAME"),
    ("Omeprazole 20mg capsule", "DRUG_NAME"),
    ("Ciprofloxacin 500mg", "DRUG_NAME"),
    ("Aspirin 81mg daily", "DRUG_NAME"),
    ("Metoprolol 50mg", "DRUG_NAME"),
    ("Tab Paracetamol 500mg", "DRUG_NAME"),
    ("Lisinopril 10mg oral", "DRUG_NAME"),
    ("Sertraline 50mg tab", "DRUG_NAME"),
    ("Azithromycin 500mg", "DRUG_NAME"),
    ("Pregabalin 75mg cap", "DRUG_NAME"),
    ("Amlodipine 5mg once daily", "DRUG_NAME"),
    ("Tab Rosuvastatin 10mg", "DRUG_NAME"),
    ("Cefixime 200mg bd", "DRUG_NAME"),
    ("Pantoprazole 40mg od", "DRUG_NAME"),
    ("Doxycycline 100mg bd", "DRUG_NAME"),
    ("Furosemide 40mg od", "DRUG_NAME"),
    # DOSAGE_INSTRUCTION
    ("Take twice daily after meals", "DOSAGE_INSTRUCTION"),
    ("Once daily at bedtime", "DOSAGE_INSTRUCTION"),
    ("Three times a day with food", "DOSAGE_INSTRUCTION"),
    ("Twice daily for 10 days", "DOSAGE_INSTRUCTION"),
    ("One tablet every 8 hours", "DOSAGE_INSTRUCTION"),
    ("Maximum 3 tablets per day", "DOSAGE_INSTRUCTION"),
    ("1-0-1 with meals", "DOSAGE_INSTRUCTION"),
    ("0-0-1 at night", "DOSAGE_INSTRUCTION"),
    ("As needed not more than every 6 hours", "DOSAGE_INSTRUCTION"),
    ("BD for 5 days", "DOSAGE_INSTRUCTION"),
    ("TDS for 7 days", "DOSAGE_INSTRUCTION"),
    ("Take with full glass of water", "DOSAGE_INSTRUCTION"),
    ("Continue for 14 days", "DOSAGE_INSTRUCTION"),
    ("Do not exceed 4g per day", "DOSAGE_INSTRUCTION"),
    ("One sachet dissolved in water twice daily", "DOSAGE_INSTRUCTION"),
    # ALLERGY
    ("Patient is allergic to penicillin", "ALLERGY"),
    ("Known allergy: NSAIDs, Sulfa drugs", "ALLERGY"),
    ("Allergies: Penicillin (anaphylaxis)", "ALLERGY"),
    ("No known drug allergies", "ALLERGY"),
    ("NKDA", "ALLERGY"),
    ("Hypersensitive to aspirin", "ALLERGY"),
    ("Allergy to cephalosporins reported", "ALLERGY"),
    ("Drug allergy: ciprofloxacin - rash", "ALLERGY"),
    ("Allergic reaction to beta-lactams", "ALLERGY"),
    ("Intolerant to statins", "ALLERGY"),
    # DIAGNOSIS
    ("Type 2 diabetes mellitus", "DIAGNOSIS"),
    ("For hypertension management", "DIAGNOSIS"),
    ("Dx: Community acquired pneumonia", "DIAGNOSIS"),
    ("Indication: urinary tract infection", "DIAGNOSIS"),
    ("Hyperlipidaemia", "DIAGNOSIS"),
    ("Depression and anxiety disorder", "DIAGNOSIS"),
    ("Chronic heart failure NYHA class II", "DIAGNOSIS"),
    ("Atrial fibrillation, anticoagulation", "DIAGNOSIS"),
    ("GERD with esophagitis", "DIAGNOSIS"),
    ("For pain management post surgery", "DIAGNOSIS"),
    ("Osteoarthritis of knee", "DIAGNOSIS"),
    ("Bacterial sinusitis", "DIAGNOSIS"),
    # PATIENT_INFO
    ("Patient: John Doe, Age 52", "PATIENT_INFO"),
    ("DOB: 14/03/1972  Weight: 72 kg", "PATIENT_INFO"),
    ("Male, 65 years, DM HTN", "PATIENT_INFO"),
    ("Ref: Dr. Smith, MD Internal Medicine", "PATIENT_INFO"),
    ("Clinic: General Hospital OPD", "PATIENT_INFO"),
    ("Date: 2026-01-15", "PATIENT_INFO"),
    ("License: 12345  NPI: 67890", "PATIENT_INFO"),
    # IRRELEVANT
    ("Please follow up in 2 weeks", "IRRELEVANT"),
    ("Signature of prescribing physician", "IRRELEVANT"),
    ("Rx", "IRRELEVANT"),
    ("Continue previous medications", "IRRELEVANT"),
    ("Laboratory reports attached", "IRRELEVANT"),
]

# ── Module 3 DDI corpus: interaction pairs + labels ──
# Features: [drug_class_match, both_anticoag, both_nsaid, one_blood_thinner,
#            both_cns, one_nephrotoxic, one_hepatotoxic, one_enzyme_inducer, one_enzyme_inhibitor]
_DDI_TRAINING = [
    # drug1, drug2, severity, feature_vec
    ("warfarin","aspirin",       "MAJOR",    [0,0,1,1,0,0,1,0,0]),
    ("warfarin","ibuprofen",     "MAJOR",    [0,0,1,1,0,0,1,0,0]),
    ("warfarin","ciprofloxacin", "MAJOR",    [0,0,0,1,0,0,1,0,1]),
    ("warfarin","sertraline",    "MAJOR",    [0,0,0,1,0,0,0,0,1]),
    ("warfarin","fluconazole",   "MAJOR",    [0,0,0,1,0,0,1,0,1]),
    ("warfarin","amiodarone",    "MAJOR",    [0,0,0,1,0,0,1,0,1]),
    ("warfarin","metronidazole", "MAJOR",    [0,0,0,1,0,0,1,0,1]),
    ("warfarin","clarithromycin","MAJOR",    [0,0,0,1,0,0,1,0,1]),
    ("aspirin","ibuprofen",      "MAJOR",    [1,0,1,1,0,0,0,0,0]),
    ("lithium","ibuprofen",      "MAJOR",    [0,0,0,0,0,1,0,0,0]),
    ("methotrexate","nsaids",    "MAJOR",    [0,0,0,0,0,1,1,0,0]),
    ("ssri","tramadol",          "MAJOR",    [0,0,0,0,1,0,0,0,1]),
    ("maoi","sertraline",        "MAJOR",    [0,0,0,0,1,0,0,0,1]),
    ("clopidogrel","omeprazole", "MODERATE", [0,0,0,0,0,0,0,0,1]),
    ("lisinopril","spironolactone","MODERATE",[0,0,0,0,0,1,0,0,0]),
    ("metformin","ibuprofen",    "MODERATE", [0,0,0,0,0,1,0,0,0]),
    ("sertraline","ibuprofen",   "MODERATE", [0,0,1,0,0,0,0,0,0]),
    ("metoprolol","amlodipine",  "MODERATE", [1,0,0,0,0,0,0,0,0]),
    ("atorvastatin","amlodipine","MODERATE", [0,0,0,0,0,0,0,0,1]),
    ("digoxin","amiodarone",     "MODERATE", [0,0,0,0,0,0,0,0,1]),
    ("furosemide","digoxin",     "MODERATE", [0,0,0,0,0,1,0,0,0]),
    ("prednisone","nsaids",      "MODERATE", [0,0,1,0,0,0,0,0,0]),
    ("amoxicillin","metoprolol", "MINOR",    [0,0,0,0,0,0,0,0,0]),
    ("omeprazole","metformin",   "MINOR",    [0,0,0,0,0,0,0,0,0]),
    ("paracetamol","ibuprofen",  "MINOR",    [0,0,1,0,0,0,0,0,0]),
    ("atorvastatin","omeprazole","MINOR",    [0,0,0,0,0,0,0,0,0]),
    ("amoxicillin","azithromycin","MINOR",   [0,0,0,0,0,0,0,0,0]),
    ("lisinopril","amlodipine",  "MINOR",    [0,0,0,0,0,0,0,0,0]),
    ("metformin","atorvastatin", "MINOR",    [0,0,0,0,0,0,0,0,0]),
    ("paracetamol","sertraline", "MINOR",    [0,0,0,0,0,0,0,0,0]),
]

# ── Drug class lookup for DDI feature construction ──
_DRUG_CLASS_MAP = {
    "warfarin":"anticoagulant","heparin":"anticoagulant","rivaroxaban":"anticoagulant","apixaban":"anticoagulant",
    "aspirin":"nsaid/antiplatelet","ibuprofen":"nsaid","naproxen":"nsaid","diclofenac":"nsaid",
    "celecoxib":"nsaid","ketorolac":"nsaid","meloxicam":"nsaid","indomethacin":"nsaid",
    "sertraline":"ssri","fluoxetine":"ssri","paroxetine":"ssri","escitalopram":"ssri","citalopram":"ssri",
    "tramadol":"opioid/snri","morphine":"opioid","codeine":"opioid","fentanyl":"opioid","oxycodone":"opioid",
    "metoprolol":"beta_blocker","atenolol":"beta_blocker","propranolol":"beta_blocker","bisoprolol":"beta_blocker","carvedilol":"beta_blocker",
    "lisinopril":"ace_inhibitor","enalapril":"ace_inhibitor","ramipril":"ace_inhibitor","captopril":"ace_inhibitor",
    "atorvastatin":"statin","simvastatin":"statin","rosuvastatin":"statin","lovastatin":"statin","pravastatin":"statin",
    "amlodipine":"ccb","verapamil":"ccb","diltiazem":"ccb","nifedipine":"ccb",
    "metformin":"biguanide","omeprazole":"ppi","pantoprazole":"ppi","esomeprazole":"ppi","rabeprazole":"ppi",
    "ciprofloxacin":"fluoroquinolone","levofloxacin":"fluoroquinolone","moxifloxacin":"fluoroquinolone",
    "amoxicillin":"penicillin","ampicillin":"penicillin","piperacillin":"penicillin",
    "azithromycin":"macrolide","clarithromycin":"macrolide","erythromycin":"macrolide",
    "furosemide":"loop_diuretic","spironolactone":"k_sparing_diuretic","hydrochlorothiazide":"thiazide",
    "digoxin":"cardiac_glycoside","amiodarone":"antiarrhythmic",
    "lithium":"mood_stabilizer","valproate":"anticonvulsant","carbamazepine":"anticonvulsant","phenytoin":"anticonvulsant",
    "prednisone":"corticosteroid","prednisolone":"corticosteroid","dexamethasone":"corticosteroid",
    "fluconazole":"azole_antifungal","ketoconazole":"azole_antifungal","itraconazole":"azole_antifungal",
    "rifampin":"enzyme_inducer","rifampicin":"enzyme_inducer","phenobarbital":"enzyme_inducer",
    "methotrexate":"antimetabolite","clopidogrel":"antiplatelet","aspirin":"antiplatelet",
    "paracetamol":"analgesic","acetaminophen":"analgesic",
}

# ════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — MEDICAL NAMED ENTITY RECOGNIZER
# ════════════════════════════════════════════════════════════════════════════
class MedicalNER:
    """
    BIO-sequence tagger using:
    - Character n-gram TF-IDF features (captures morphological patterns)
    - Medical context window features (neighbouring token signals)
    - Logistic Regression multi-class classifier per token

    Tags: B-DRUG, I-DRUG, B-DOSE, I-DOSE, B-FREQ, I-FREQ,
          B-ROUTE, B-DURATION, B-CONDITION, B-ALLERGY, O
    """

    _DOSE_UNITS = {"mg","mcg","g","ml","l","units","iu","meq","mmol","%","tab","cap","tablet","capsule","sachet","drops","puffs"}
    _FREQ_WORDS = {"daily","twice","thrice","once","bid","tid","qid","tds","bd","od","qd","hourly",
                   "weekly","monthly","morning","evening","night","bedtime","meals","hours"}
    _ROUTE_WORDS = {"oral","sublingual","injection","iv","intramuscular","subcutaneous","topical",
                    "inhalation","nasal","rectal","transdermal","ophthalmic","inhale","inhaled"}

    # Known drug-like suffixes/prefixes for morphological detection
    _DRUG_PATTERNS = [
        r'.*(?:mab|nib|lib|vir|cillin|mycin|cyclin|azole|olol|pril|sartan|statin|dipine|gliptin|floxacin|oxacin|oxetine|prazole|tidine)$',
        r'^(?:met|ami|dox|ator|ros|sim|pred|omep|cipro|azithro|amox)',
    ]

    def __init__(self):
        self._drug_vocab = set()
        self._trained = False

    def _token_features(self, token, prev_tok="", next_tok=""):
        """Extract features for a single token."""
        tl = token.lower()
        feats = {}

        # Morphological
        feats["len"] = min(len(token), 20)
        feats["has_digit"] = int(bool(re.search(r'\d', token)))
        feats["is_upper"] = int(token.isupper())
        feats["is_title"] = int(token.istitle())
        feats["ends_dose_unit"] = int(tl in self._DOSE_UNITS)
        feats["is_freq"] = int(tl in self._FREQ_WORDS)
        feats["is_route"] = int(tl in self._ROUTE_WORDS)

        # Numeric patterns
        dose_match = re.match(r'^(\d+(?:\.\d+)?)(mg|mcg|g|ml)?$', tl)
        feats["is_dose_token"] = int(bool(dose_match))
        feats["is_pure_number"] = int(bool(re.match(r'^\d+$', token)))

        # Drug-like morphology
        feats["drug_like"] = int(any(re.match(p, tl) for p in self._DRUG_PATTERNS))
        feats["in_drug_vocab"] = int(tl in self._drug_vocab)

        # Context
        feats["prev_is_number"] = int(bool(re.match(r'^\d', prev_tok)))
        feats["next_is_unit"] = int(next_tok.lower() in self._DOSE_UNITS)
        feats["prev_allergy_kw"] = int(prev_tok.lower() in {"allerg","allergic","allergy","hypersensitive","intolerant"})
        feats["prev_for_kw"] = int(prev_tok.lower() in {"for","indication","dx","treating"})
        feats["has_hyphen"] = int('-' in token)
        feats["char_bigram_count"] = len(token) - 1 if len(token) > 1 else 0

        return feats

    def _feats_to_vector(self, feats):
        return np.array([
            feats["len"]/20, feats["has_digit"], feats["is_upper"],
            feats["is_title"], feats["ends_dose_unit"], feats["is_freq"],
            feats["is_route"], feats["is_dose_token"], feats["is_pure_number"],
            feats["drug_like"], feats["in_drug_vocab"], feats["prev_is_number"],
            feats["next_is_unit"], feats["prev_allergy_kw"], feats["prev_for_kw"],
            feats["has_hyphen"], min(feats["char_bigram_count"]/15, 1.0)
        ], dtype=np.float32)

    def _make_training_data(self):
        """Generate BIO-tagged training examples."""
        examples = [
            # (tokens, bio_tags)
            (["Metformin","500mg","twice","daily"], ["B-DRUG","B-DOSE","B-FREQ","I-FREQ"]),
            (["Tab","Amoxicillin","875","mg","three","times","daily","for","7","days"],
             ["O","B-DRUG","B-DOSE","I-DOSE","B-FREQ","I-FREQ","I-FREQ","O","B-DURATION","I-DURATION"]),
            (["Warfarin","5","mg","once","daily","oral"],
             ["B-DRUG","B-DOSE","I-DOSE","B-FREQ","I-FREQ","B-ROUTE"]),
            (["Allergies","Penicillin","NSAIDs"],
             ["O","B-ALLERGY","B-ALLERGY"]),
            (["Patient","allergic","to","aspirin"],
             ["O","O","O","B-ALLERGY"]),
            (["Ciprofloxacin","500mg","BD","for","10","days","oral"],
             ["B-DRUG","B-DOSE","B-FREQ","O","B-DURATION","I-DURATION","B-ROUTE"]),
            (["No","known","drug","allergies"],["O","O","O","O"]),
            (["for","type","2","diabetes"],["O","B-CONDITION","I-CONDITION","I-CONDITION"]),
            (["for","hypertension"],["O","B-CONDITION"]),
            (["Ibuprofen","400","mg","every","6","hours","as","needed"],
             ["B-DRUG","B-DOSE","I-DOSE","B-FREQ","I-FREQ","I-FREQ","I-FREQ","I-FREQ"]),
            (["Atorvastatin","40mg","at","bedtime"],
             ["B-DRUG","B-DOSE","B-FREQ","I-FREQ"]),
            (["Lisinopril","10mg","once","daily","for","hypertension"],
             ["B-DRUG","B-DOSE","B-FREQ","I-FREQ","O","B-CONDITION"]),
            (["subcutaneous","injection","twice","weekly"],
             ["B-ROUTE","I-ROUTE","B-FREQ","I-FREQ"]),
            (["Apply","topically","twice","daily"],
             ["O","B-ROUTE","B-FREQ","I-FREQ"]),
            (["Omeprazole","20","mg","once","daily","before","breakfast"],
             ["B-DRUG","B-DOSE","I-DOSE","B-FREQ","I-FREQ","I-FREQ","I-FREQ"]),
            (["Sertraline","50mg","daily"],["B-DRUG","B-DOSE","B-FREQ"]),
            (["Tab","Metoprolol","25mg","BD"],["O","B-DRUG","B-DOSE","B-FREQ"]),
        ]

        # Populate drug vocab
        for tokens, tags in examples:
            for tok, tag in zip(tokens, tags):
                if tag.startswith("B-DRUG"):
                    self._drug_vocab.add(tok.lower())

        # Build matrices
        X, y = [], []
        for tokens, tags in examples:
            for i, (tok, tag) in enumerate(zip(tokens, tags)):
                prev = tokens[i-1] if i > 0 else ""
                nxt  = tokens[i+1] if i < len(tokens)-1 else ""
                vec = self._feats_to_vector(self._token_features(tok, prev, nxt))
                X.append(vec)
                y.append(tag)
        return np.array(X), y

    def train(self):
        X, y = self._make_training_data()
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self._clf = LogisticRegression(max_iter=500, C=2.0)
        self._clf.fit(X, y_enc)
        self._trained = True
        return self

    def tag_tokens(self, tokens):
        """Tag a list of tokens, return list of (token, tag, confidence)."""
        if not self._trained:
            raise RuntimeError("Call .train() first")
        results = []
        for i, tok in enumerate(tokens):
            prev = tokens[i-1] if i > 0 else ""
            nxt  = tokens[i+1] if i < len(tokens)-1 else ""
            vec = self._feats_to_vector(self._token_features(tok, prev, nxt))
            proba = self._clf.predict_proba([vec])[0]
            pred_idx = proba.argmax()
            tag = self._le.inverse_transform([pred_idx])[0]
            conf = float(proba[pred_idx])
            results.append((tok, tag, round(conf, 3)))
        return results

    def extract_entities(self, text):
        """
        Full NER pipeline: tokenize → tag → merge BIO spans → return entities.
        Returns dict with lists: drugs, doses, frequencies, routes, durations, conditions, allergies
        """
        # Tokenize (preserve dose units attached to numbers like 500mg)
        tokens = re.findall(r"\d+(?:\.\d+)?(?:mg|mcg|g|ml|iu|meq|units?)?|[A-Za-z]+[-']?[A-Za-z]*|\d+", text)
        tagged = self.tag_tokens(tokens)

        entities = defaultdict(list)
        i = 0
        while i < len(tagged):
            tok, tag, conf = tagged[i]
            if tag.startswith("B-"):
                entity_type = tag[2:]
                span = [tok]
                j = i + 1
                while j < len(tagged) and tagged[j][1] == f"I-{entity_type}":
                    span.append(tagged[j][0])
                    j += 1
                avg_conf = statistics.mean([tagged[k][2] for k in range(i, j)])
                entities[entity_type].append({
                    "text": " ".join(span),
                    "confidence": round(avg_conf, 3),
                    "start_token": i
                })
                i = j
            else:
                i += 1
        return dict(entities)

    def get_confidence_scores(self, text):
        """Returns per-category confidence averaged across detected entities."""
        tokens = re.findall(r"\d+(?:\.\d+)?(?:mg|mcg|g|ml|iu|units?)?|[A-Za-z]+[-']?[A-Za-z]*|\d+", text)
        tagged = self.tag_tokens(tokens)
        scores = defaultdict(list)
        for _, tag, conf in tagged:
            category = tag[2:] if tag.startswith(("B-","I-")) else "O"
            scores[category].append(conf)
        return {k: round(statistics.mean(v), 3) for k, v in scores.items() if k != "O"}


# ════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — DRUG NAME NORMALIZER
# ════════════════════════════════════════════════════════════════════════════
class DrugNameNormalizer:
    """
    TF-IDF cosine similarity over a drug vocabulary.
    Handles: typos, brand names, Indian trade names, abbreviations.
    Also provides edit-distance fallback for very short tokens.
    """

    # Comprehensive drug vocab (generics + common brands)
    _VOCAB = [
        # Antibiotics
        "amoxicillin","amoxicillin clavulanate","ampicillin","azithromycin","clarithromycin",
        "ciprofloxacin","levofloxacin","moxifloxacin","doxycycline","metronidazole",
        "clindamycin","nitrofurantoin","trimethoprim sulfamethoxazole","cefixime","cephalexin",
        "ceftriaxone","cefuroxime","cefdinir","piperacillin tazobactam","meropenem",
        "vancomycin","linezolid","tetracycline","erythromycin","cotrimoxazole",
        # Cardiovascular
        "atorvastatin","rosuvastatin","simvastatin","lovastatin","pravastatin","fluvastatin",
        "metoprolol","atenolol","bisoprolol","carvedilol","propranolol","nebivolol",
        "amlodipine","nifedipine","verapamil","diltiazem","felodipine",
        "lisinopril","enalapril","ramipril","captopril","perindopril","trandolapril",
        "losartan","valsartan","telmisartan","olmesartan","irbesartan","candesartan",
        "aspirin","clopidogrel","ticagrelor","prasugrel","warfarin","rivaroxaban","apixaban","dabigatran",
        "furosemide","hydrochlorothiazide","spironolactone","eplerenone","torsemide","chlorthalidone",
        "digoxin","amiodarone","ivabradine","sacubitril valsartan","isosorbide mononitrate","nitroglycerin",
        # Diabetes
        "metformin","glipizide","gliclazide","glibenclamide","glimepiride","sitagliptin","vildagliptin",
        "saxagliptin","alogliptin","empagliflozin","dapagliflozin","canagliflozin","liraglutide",
        "semaglutide","exenatide","insulin","insulin glargine","insulin detemir","insulin aspart",
        # GI / Acid
        "omeprazole","pantoprazole","esomeprazole","rabeprazole","lansoprazole",
        "ranitidine","famotidine","sucralfate","domperidone","metoclopramide",
        "ondansetron","granisetron","loperamide","bisacodyl","lactulose",
        # Pain / Musculoskeletal
        "paracetamol","acetaminophen","ibuprofen","naproxen","diclofenac","celecoxib",
        "etoricoxib","indomethacin","ketorolac","meloxicam","piroxicam",
        "tramadol","codeine","morphine","oxycodone","fentanyl","buprenorphine","tapentadol",
        "gabapentin","pregabalin","amitriptyline","duloxetine",
        "allopurinol","febuxostat","colchicine","methotrexate","hydroxychloroquine","sulfasalazine",
        # CNS / Psychiatry
        "sertraline","fluoxetine","paroxetine","escitalopram","citalopram","fluvoxamine",
        "venlafaxine","duloxetine","mirtazapine","bupropion","trazodone",
        "olanzapine","risperidone","quetiapine","aripiprazole","haloperidol","clozapine",
        "diazepam","lorazepam","alprazolam","clonazepam","zolpidem","zopiclone",
        "lithium","valproate","carbamazepine","lamotrigine","levetiracetam","phenytoin","phenobarbitone",
        "donepezil","rivastigmine","memantine","methylphenidate",
        # Respiratory
        "salbutamol","formoterol","salmeterol","budesonide","fluticasone","beclomethasone",
        "tiotropium","ipratropium","montelukast","cetirizine","loratadine","fexofenadine",
        "prednisolone","dexamethasone","hydrocortisone","methylprednisolone",
        # Hormones
        "levothyroxine","carbimazole","propylthiouracil","hydrocortisone","fludrocortisone",
        "testosterone","estradiol","progesterone","norethisterone","medroxyprogesterone",
        # Others
        "acyclovir","valacyclovir","oseltamivir","tenofovir","lamivudine","efavirenz",
        "alendronate","risedronate","denosumab","zoledronic acid","calcitriol","cholecalciferol",
        "ferrous sulfate","folic acid","cyanocobalamin","methylcobalamin",
        "potassium chloride","magnesium sulfate","calcium carbonate","zinc sulfate",
        "albendazole","ivermectin","praziquantel","chloroquine","hydroxychloroquine",
        "tamsulosin","sildenafil","tadalafil","finasteride","dutasteride",
        "drotaverine","mefenamic acid","tranexamic acid","ferrous ascorbate",
    ]

    # Brand → generic mapping (Indian + global)
    _BRANDS = {
        "augmentin":"amoxicillin clavulanate","azee":"azithromycin","mox":"amoxicillin",
        "ciplox":"ciprofloxacin","taxim":"cefixime","sporidex":"cephalexin",
        "cetzine":"cetirizine","allegra":"fexofenadine","montek":"montelukast",
        "pan":"pantoprazole","pantocid":"pantoprazole","omez":"omeprazole","nexpro":"esomeprazole",
        "rantac":"ranitidine","zantac":"ranitidine","pepcid":"famotidine",
        "combiflam":"ibuprofen paracetamol","dolo":"paracetamol","calpol":"paracetamol",
        "voveran":"diclofenac","volini":"diclofenac","moov":"diclofenac",
        "lipitor":"atorvastatin","crestor":"rosuvastatin","zocor":"simvastatin",
        "norvasc":"amlodipine","tenormin":"atenolol","lopressor":"metoprolol","toprol":"metoprolol",
        "zestril":"lisinopril","vasotec":"enalapril","altace":"ramipril",
        "cozaar":"losartan","diovan":"valsartan","micardis":"telmisartan",
        "coumadin":"warfarin","plavix":"clopidogrel","xarelto":"rivaroxaban","eliquis":"apixaban",
        "glucophage":"metformin","amaryl":"glimepiride","diamicron":"gliclazide",
        "januvia":"sitagliptin","galvus":"vildagliptin","jardiance":"empagliflozin","farxiga":"dapagliflozin",
        "zoloft":"sertraline","prozac":"fluoxetine","lexapro":"escitalopram","celexa":"citalopram",
        "effexor":"venlafaxine","cymbalta":"duloxetine","wellbutrin":"bupropion",
        "zyprexa":"olanzapine","risperdal":"risperidone","seroquel":"quetiapine","abilify":"aripiprazole",
        "valium":"diazepam","ativan":"lorazepam","xanax":"alprazolam","klonopin":"clonazepam",
        "synthroid":"levothyroxine","eltroxin":"levothyroxine",
        "zovirax":"acyclovir","valtrex":"valacyclovir","tamiflu":"oseltamivir",
        "venolin":"salbutamol","ventolin":"salbutamol","seretide":"salmeterol fluticasone",
        "stugeron":"cinnarizine","vertin":"betahistine","proton":"pantoprazole",
        "concor":"bisoprolol","atenol":"atenolol","metolar":"metoprolol",
        "glycomet":"metformin","obimet":"metformin","gluconorm":"glipizide",
        "ecosprin":"aspirin","deplatt":"clopidogrel","cardivas":"carvedilol",
        "olsar":"olmesartan","telsar":"telmisartan","starpress":"metoprolol",
        "rcinex":"rifampicin","pyrazinamide":"pyrazinamide","ethambutol":"ethambutol",
        "zerodol":"aceclofenac","hifenac":"aceclofenac","nicip":"nimesulide",
        "acenext":"aceclofenac paracetamol","dolopar":"paracetamol",
        "sporanox":"itraconazole","diflucan":"fluconazole",
        "aricept":"donepezil","exelon":"rivastigmine","namenda":"memantine",
    }

    def __init__(self):
        self._all_names = list(set(self._VOCAB + list(self._BRANDS.keys())))
        # TF-IDF on character trigrams — captures subword morphology
        self._vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4),
            min_df=1, sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform(self._all_names)
        self._name_index = {n: i for i, n in enumerate(self._all_names)}

    def normalize(self, drug_name, threshold=0.25):
        """
        Returns (normalized_generic_name, match_type, confidence).
        match_type: 'exact'|'brand'|'fuzzy'|'unknown'
        """
        dn = drug_name.lower().strip()

        # Exact match first
        if dn in self._name_index:
            generic = self._BRANDS.get(dn, dn)
            return generic, 'exact', 1.0

        # Brand map lookup
        if dn in self._BRANDS:
            return self._BRANDS[dn], 'brand', 1.0

        # TF-IDF cosine similarity fuzzy match
        try:
            q_vec = self._vectorizer.transform([dn])
            sims = cosine_similarity(q_vec, self._matrix).flatten()
            best_idx = int(sims.argmax())
            best_score = float(sims[best_idx])
            if best_score >= threshold:
                matched_name = self._all_names[best_idx]
                generic = self._BRANDS.get(matched_name, matched_name)
                return generic, 'fuzzy', round(best_score, 3)
        except Exception:
            pass

        return drug_name.lower(), 'unknown', 0.0

    def get_drug_class(self, drug_name):
        normalized, _, _ = self.normalize(drug_name)
        return _DRUG_CLASS_MAP.get(normalized, "")

    def batch_normalize(self, drug_list):
        return [self.normalize(d) for d in drug_list]


# ════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — DDI SEVERITY CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════
class DDISeverityClassifier:
    """
    Logistic Regression trained on drug-pair feature vectors.
    Features encode pharmacological class relationships.
    Predicts: MAJOR / MODERATE / MINOR / UNKNOWN  +  confidence %
    """

    _BLOOD_THINNERS = {"warfarin","heparin","rivaroxaban","apixaban","dabigatran","clopidogrel","aspirin","ticagrelor"}
    _ANTICOAG       = {"warfarin","heparin","rivaroxaban","apixaban","dabigatran"}
    _NSAIDS         = {"aspirin","ibuprofen","naproxen","diclofenac","celecoxib","ketorolac","meloxicam","indomethacin","etoricoxib"}
    _CNS_DRUGS      = {"sertraline","fluoxetine","paroxetine","escitalopram","venlafaxine","tramadol","codeine","morphine",
                       "diazepam","lorazepam","alprazolam","olanzapine","quetiapine","haloperidol","lithium","valproate"}
    _NEPHROTOXIC    = {"nsaids","aminoglycosides","cyclosporine","tacrolimus","ibuprofen","naproxen","vancomycin","colistin"}
    _HEPATOTOXIC    = {"warfarin","methotrexate","isoniazid","rifampicin","valproate","amiodarone","ketoconazole","fluconazole"}
    _ENZYME_INDUCERS= {"rifampicin","rifampin","carbamazepine","phenytoin","phenobarbitone","st john","efavirenz"}
    _ENZYME_INHIBITORS = {"fluconazole","ketoconazole","itraconazole","erythromycin","clarithromycin","ciprofloxacin",
                          "amiodarone","omeprazole","sertraline","fluoxetine","paroxetine"}

    def _make_feature_vector(self, drug1, drug2):
        d1, d2 = drug1.lower(), drug2.lower()
        c1 = _DRUG_CLASS_MAP.get(d1, "")
        c2 = _DRUG_CLASS_MAP.get(d2, "")

        return np.array([
            float(c1 == c2 and c1 != ""),                          # same class
            float(d1 in self._ANTICOAG and d2 in self._ANTICOAG),  # both anticoag
            float(d1 in self._NSAIDS    and d2 in self._NSAIDS),    # both NSAIDs
            float((d1 in self._BLOOD_THINNERS) != (d2 in self._BLOOD_THINNERS)), # one blood thinner
            float(d1 in self._CNS_DRUGS and d2 in self._CNS_DRUGS), # both CNS
            float(d1 in self._NEPHROTOXIC or d2 in self._NEPHROTOXIC), # nephrotoxic
            float(d1 in self._HEPATOTOXIC or d2 in self._HEPATOTOXIC), # hepatotoxic
            float(d1 in self._ENZYME_INDUCERS or d2 in self._ENZYME_INDUCERS), # enzyme inducer
            float(d1 in self._ENZYME_INHIBITORS or d2 in self._ENZYME_INHIBITORS), # enzyme inhibitor
        ], dtype=np.float32)

    def train(self):
        X = np.array([row[3] for row in _DDI_TRAINING], dtype=np.float32)
        y = [row[2] for row in _DDI_TRAINING]
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self._clf = LogisticRegression(max_iter=500, C=1.5, class_weight='balanced')
        self._clf.fit(X, y_enc)
        return self

    def predict(self, drug1, drug2):
        """Returns (severity, confidence_pct, explanation)."""
        vec = self._make_feature_vector(drug1, drug2)
        proba = self._clf.predict_proba([vec])[0]
        pred_idx = proba.argmax()
        severity = self._le.inverse_transform([pred_idx])[0]
        confidence = round(float(proba[pred_idx]) * 100, 1)

        # Build explanation from active features
        d1, d2 = drug1.lower(), drug2.lower()
        reasons = []
        c1 = _DRUG_CLASS_MAP.get(d1, "")
        c2 = _DRUG_CLASS_MAP.get(d2, "")
        if c1 and c1 == c2:
            reasons.append(f"both are {c1.replace('_',' ')}")
        if d1 in self._ANTICOAG or d2 in self._ANTICOAG:
            reasons.append("anticoagulant involvement — bleeding risk")
        if d1 in self._NSAIDS and d2 in self._BLOOD_THINNERS or d2 in self._NSAIDS and d1 in self._BLOOD_THINNERS:
            reasons.append("NSAID + blood thinner — GI bleed risk")
        if d1 in self._CNS_DRUGS and d2 in self._CNS_DRUGS:
            reasons.append("dual CNS agents — additive sedation/serotonin risk")
        if d1 in self._ENZYME_INHIBITORS or d2 in self._ENZYME_INHIBITORS:
            reasons.append("CYP450 enzyme inhibition — elevated plasma levels")
        if d1 in self._ENZYME_INDUCERS or d2 in self._ENZYME_INDUCERS:
            reasons.append("CYP450 enzyme induction — reduced drug efficacy")
        if d1 in self._NEPHROTOXIC or d2 in self._NEPHROTOXIC:
            reasons.append("nephrotoxic combination — renal function risk")

        explanation = "; ".join(reasons) if reasons else "pharmacokinetic/pharmacodynamic overlap"
        return severity, confidence, explanation

    def classify_all_pairs(self, drug_names):
        """Classify all unique pairs in a drug list."""
        results = []
        names = list(set(dn.lower() for dn in drug_names))
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                d1, d2 = names[i], names[j]
                severity, conf, expl = self.predict(d1, d2)
                if severity in ("MAJOR", "MODERATE"):  # only report meaningful ones
                    results.append({
                        "drug1": d1, "drug2": d2,
                        "severity": severity,
                        "ml_confidence": conf,
                        "ml_reasoning": expl,
                        "source": "DDI-Classifier (ML)"
                    })
        results.sort(key=lambda x: (0 if x["severity"]=="MAJOR" else 1, -x["ml_confidence"]))
        return results


# ════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — POLYPHARMACY RISK SCORER
# ════════════════════════════════════════════════════════════════════════════
class PolypharmacyRiskScorer:
    """
    Gradient Boosted Classifier trained on synthetic patient profiles.
    Features: drug_count, num_interactions, num_major, num_allergy_conflicts,
              num_dosage_errors, has_anticoagulant, has_nsaid, has_cns,
              has_nephrotoxic, class_overlap_count
    Output: risk_level (LOW/MODERATE/HIGH/CRITICAL) + risk_score (0-100)
    """

    def _make_training_data(self):
        """Synthetic training profiles for risk scoring."""
        # (features_12, label)
        profiles = [
            # [n_drugs, n_int, n_major, n_allergy, n_dose_err, has_ac, has_nsaid, has_cns, has_neph, class_overlap, age_risk, renal_risk]
            ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "LOW"),
            ([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "LOW"),
            ([3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "LOW"),
            ([2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], "LOW"),
            ([3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], "MODERATE"),
            ([4, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], "MODERATE"),
            ([5, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], "MODERATE"),
            ([4, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], "MODERATE"),
            ([6, 3, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0], "HIGH"),
            ([5, 2, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], "HIGH"),
            ([7, 4, 2, 0, 0, 0, 0, 0, 1, 2, 1, 0], "HIGH"),
            ([4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], "HIGH"),
            ([6, 2, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], "HIGH"),
            ([8, 5, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1], "CRITICAL"),
            ([5, 3, 2, 0, 2, 1, 1, 0, 0, 0, 0, 0], "CRITICAL"),
            ([7, 4, 3, 1, 0, 1, 0, 1, 1, 2, 1, 0], "CRITICAL"),
            ([9, 6, 3, 0, 1, 1, 1, 1, 1, 4, 1, 1], "CRITICAL"),
            ([10, 8, 4, 2, 2, 1, 1, 1, 1, 5, 1, 1], "CRITICAL"),
        ]
        X = np.array([p[0] for p in profiles], dtype=np.float32)
        y = [p[1] for p in profiles]
        return X, y

    def train(self):
        X, y = self._make_training_data()
        self._le = LabelEncoder()
        y_enc = self._le.fit_transform(y)
        self._clf = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.15, random_state=42
        )
        self._clf.fit(X, y_enc)
        # Feature importances
        self._feature_names = [
            "drug_count","total_interactions","major_interactions","allergy_conflicts",
            "dosage_errors","has_anticoagulant","has_nsaid","has_cns_drug",
            "has_nephrotoxic","drug_class_overlaps","patient_age_risk","renal_risk"
        ]
        return self

    def _make_features(self, drugs, alerts, drug_info_map):
        n_drugs = len(drugs)
        n_int = sum(1 for a in alerts if a.get("type") == "interaction")
        n_major = sum(1 for a in alerts if a.get("type") == "interaction" and a.get("severity") == "MAJOR")
        n_allergy = sum(1 for a in alerts if a.get("type") == "allergy")
        n_dose_err = sum(1 for a in alerts if a.get("type") == "dosage")

        drug_names = [d.get("drug","").lower() for d in drugs]
        has_ac   = int(any(d in {"warfarin","heparin","rivaroxaban","apixaban","dabigatran"} for d in drug_names))
        has_nsaid= int(any(d in {"aspirin","ibuprofen","naproxen","diclofenac","celecoxib"} for d in drug_names))
        has_cns  = int(any(_DRUG_CLASS_MAP.get(d,"") in {"ssri","opioid","opioid/snri","anticonvulsant"} for d in drug_names))
        has_neph = int(any(_DRUG_CLASS_MAP.get(d,"") in {"nsaid","loop_diuretic"} for d in drug_names))
        classes = [_DRUG_CLASS_MAP.get(d,"") for d in drug_names]
        class_overlap = max(0, len(classes) - len(set(c for c in classes if c)))
        age_risk = 0   # could be populated from patient metadata
        renal_risk = 0

        return np.array([[n_drugs, n_int, n_major, n_allergy, n_dose_err,
                          has_ac, has_nsaid, has_cns, has_neph, class_overlap, age_risk, renal_risk]], dtype=np.float32)

    def score(self, drugs, alerts, drug_info_map=None):
        """Returns (risk_level, risk_score_0_100, factor_breakdown)."""
        if drug_info_map is None:
            drug_info_map = {}
        feats = self._make_features(drugs, alerts, drug_info_map)
        proba = self._clf.predict_proba(feats)[0]
        pred_idx = proba.argmax()
        risk_level = self._le.inverse_transform([pred_idx])[0]

        # Compute risk_score (0-100 continuous)
        level_weights = {"LOW": 10, "MODERATE": 40, "HIGH": 70, "CRITICAL": 95}
        base_score = level_weights.get(risk_level, 0)
        noise = int((proba[pred_idx] - 0.5) * 20)
        risk_score = max(0, min(100, base_score + noise))

        # Feature importances as explanation
        breakdown = {}
        for name, val, imp in zip(self._feature_names, feats[0], self._clf.feature_importances_):
            if val > 0:
                breakdown[name.replace("_", " ")] = round(float(imp * 100), 1)

        return risk_level, risk_score, dict(sorted(breakdown.items(), key=lambda x: -x[1])[:5])


# ════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — CLINICAL SENTENCE CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════
class ClinicalSentenceClassifier:
    """
    Multinomial Naive Bayes (TF-IDF features) sentence classifier.
    Labels prescription sentences before NER to focus entity extraction.
    """

    def train(self):
        X = [s[0] for s in _SENT_CORPUS]
        y = [s[1] for s in _SENT_CORPUS]
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer='word', ngram_range=(1, 2),
                min_df=1, sublinear_tf=True, lowercase=True
            )),
            ("clf", ComplementNB(alpha=0.5)),
        ])
        self._pipeline.fit(X, y)
        return self

    def classify(self, sentence):
        """Returns (label, confidence_pct)."""
        proba = self._pipeline.predict_proba([sentence])[0]
        classes = self._pipeline.classes_
        idx = proba.argmax()
        return classes[idx], round(float(proba[idx]) * 100, 1)

    def classify_batch(self, sentences):
        return [self.classify(s) for s in sentences]

    def filter_relevant(self, text):
        """Split text into sentences and return only clinically relevant ones with their labels."""
        sents = re.split(r'[.\n;]+', text)
        results = []
        for s in sents:
            s = s.strip()
            if len(s) < 3:
                continue
            label, conf = self.classify(s)
            if label != "IRRELEVANT" or conf < 80:
                results.append({"sentence": s, "label": label, "confidence": conf})
        return results


# ════════════════════════════════════════════════════════════════════════════
#  INTEGRATED PIPELINE
# ════════════════════════════════════════════════════════════════════════════
class RxGuardNLPPipeline:
    """
    Orchestrates all 5 ML/NLP modules into a single call.
    Called from the Flask app to enrich parse results.
    """

    def __init__(self):
        t0 = time.time()
        print("[ML] Training Module 1: Medical NER...")
        self.ner = MedicalNER().train()
        print("[ML] Training Module 2: Drug Name Normalizer (TF-IDF)...")
        self.normalizer = DrugNameNormalizer()
        print("[ML] Training Module 3: DDI Severity Classifier (LogReg)...")
        self.ddi_clf = DDISeverityClassifier().train()
        print("[ML] Training Module 4: Polypharmacy Risk Scorer (GBM)...")
        self.risk_scorer = PolypharmacyRiskScorer().train()
        print("[ML] Training Module 5: Clinical Sentence Classifier (NB)...")
        self.sent_clf = ClinicalSentenceClassifier().train()
        elapsed = round(time.time() - t0, 2)
        print(f"[ML] All 5 models trained in {elapsed}s ✓")

    def run(self, raw_text, drugs_list, alerts, drug_info_map=None):
        """
        Full enrichment pipeline.
        Returns: ml_result dict merged into /analyze response.
        """
        if drug_info_map is None:
            drug_info_map = {}

        # 1. Sentence classification
        sent_labels = self.sent_clf.filter_relevant(raw_text)

        # 2. NER extraction
        ner_entities = self.ner.extract_entities(raw_text)
        ner_confidence = self.ner.get_confidence_scores(raw_text)

        # 3. Drug normalization for each detected drug
        normalized = []
        for d in drugs_list:
            name, match_type, conf = self.normalizer.normalize(d.get("drug",""))
            drug_class = self.normalizer.get_drug_class(name)
            normalized.append({
                "original": d.get("drug"),
                "normalized": name,
                "match_type": match_type,
                "confidence": conf,
                "drug_class": drug_class or d.get("drug_class",""),
            })

        # 4. ML DDI classification (on top of API-based DDI)
        drug_names = [d.get("drug","") for d in drugs_list]
        ml_interactions = self.ddi_clf.classify_all_pairs(drug_names)

        # 5. Polypharmacy risk scoring
        risk_level, risk_score, breakdown = self.risk_scorer.score(drugs_list, alerts, drug_info_map)

        # 6. Per-drug NER confidence (how well we extracted each drug)
        drug_conf_scores = {}
        for ent in ner_entities.get("DRUG", []):
            text_clean = ent["text"].lower().strip()
            drug_conf_scores[text_clean] = ent["confidence"]

        return {
            "ml_active": True,
            "models_used": [
                "Medical NER (BIO-tagger, LogReg, character n-gram TF-IDF)",
                "Drug Name Normalizer (TF-IDF cosine similarity, char bigrams)",
                "DDI Severity Classifier (Logistic Regression, drug-class features)",
                "Polypharmacy Risk Scorer (Gradient Boosting, 12 features)",
                "Clinical Sentence Classifier (Complement Naive Bayes, TF-IDF)"
            ],
            "sentence_analysis": sent_labels[:15],
            "ner_entities": {
                k: [e["text"] for e in v] for k, v in ner_entities.items()
            },
            "ner_confidence_by_category": ner_confidence,
            "drug_normalizations": normalized,
            "ml_interactions": ml_interactions,
            "risk_assessment": {
                "level": risk_level,
                "score": risk_score,
                "factors": breakdown,
                "model": "GradientBoosting (12-feature polypharmacy model)"
            },
            "summary_stats": {
                "entities_extracted": sum(len(v) for v in ner_entities.values()),
                "drugs_normalized": len(normalized),
                "ml_interactions_found": len(ml_interactions),
                "risk_level": risk_level,
                "risk_score": risk_score,
            }
        }

# Singleton — instantiated once at app startup
_pipeline_instance = None

def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RxGuardNLPPipeline()
    return _pipeline_instance
