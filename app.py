"""
P-helper — ML/NLP Enhanced Prescription Safety Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original API layers (OpenFDA, RxNorm, RxClass) preserved.
ML/NLP layer injected at the end of /analyze pipeline.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, re, io, json, threading, socket
from flask import Flask, render_template, request, jsonify, send_file
import urllib.request, urllib.parse
import csv
import datetime

# ── ML/NLP Engine ──
from nlp_engine import get_pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
os.makedirs("uploads", exist_ok=True)

# ─── NETWORK DETECTION ──────────────────────────────────────────────────────
def _internet_available():
    try:
        socket.setdefaulttimeout(2)
        socket.getaddrinfo("rxnav.nlm.nih.gov", 443)
        return True
    except Exception:
        return False

_ONLINE = _internet_available()
_MODE   = "live" if _ONLINE else "mock"
_RXNAV_BASE   = "https://rxnav.nlm.nih.gov" if _ONLINE else "http://localhost:5001"
_OPENFDA_BASE = "https://api.fda.gov"        if _ONLINE else "http://localhost:5001"

print(f"[P-helper] Network: {'ONLINE — using live APIs' if _ONLINE else 'OFFLINE — mock mode'}")

_HEADERS = {"User-Agent": "P-helper/2.0 (prescription-safety-tool)"}
_cache = {}
_cache_lock = threading.Lock()

# ─── HTTP HELPER ─────────────────────────────────────────────────────────────
def _get_json(url, timeout=8):
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None

def _cached(key, fn):
    with _cache_lock:
        if key in _cache: return _cache[key]
    result = fn()
    with _cache_lock: _cache[key] = result
    return result

# ════ RxNorm ════════════════════════════════════════════════════════════════
def rxnorm_get_rxcui(drug_name):
    def _f():
        enc = urllib.parse.quote(drug_name)
        url = f"{_RXNAV_BASE}/REST/rxcui.json?name={enc}&search=2"
        data = _get_json(url)
        if data:
            ids = (data.get("idGroup") or {}).get("rxnormId") or []
            return ids[0] if ids else None
        return None
    return _cached(f"rxcui:{drug_name}", _f)

def rxnorm_get_properties(rxcui):
    def _f():
        data = _get_json(f"{_RXNAV_BASE}/REST/rxcui/{rxcui}/properties.json")
        return (data or {}).get("properties") or {}
    return _cached(f"props:{rxcui}", _f)

# ════ RxClass ════════════════════════════════════════════════════════════════
def rxclass_get_classes(rxcui):
    def _f():
        classes = []
        for src in ["FMTSME","ATC1-4","EPC","VA"]:
            url = f"{_RXNAV_BASE}/REST/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource={src}"
            data = _get_json(url)
            if data:
                for e in ((data.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo") or []):
                    cn = (e.get("rxclassMinConceptItem") or {}).get("className","")
                    if cn: classes.append(cn.lower())
        return list(set(classes))
    return _cached(f"classes:{rxcui}", _f)

# ════ OpenFDA Label ══════════════════════════════════════════════════════════
def openfda_get_label(drug_name):
    def _f():
        enc = urllib.parse.quote(f'"{drug_name}"')
        data = _get_json(f"{_OPENFDA_BASE}/drug/label.json?search=openfda.generic_name:{enc}&limit=1")
        if not (data and data.get("results")):
            data = _get_json(f"{_OPENFDA_BASE}/drug/label.json?search=openfda.brand_name:{enc}&limit=1")
        return (data or {}).get("results", [{}])[0] if data else {}
    return _cached(f"label:{drug_name}", _f)

# ════ OpenFDA FAERS ══════════════════════════════════════════════════════════
def openfda_get_adverse_events(drug_name, limit=6):
    def _f():
        enc = urllib.parse.quote(drug_name)
        url = (f"{_OPENFDA_BASE}/drug/event.json"
               f"?search=patient.drug.medicinalproduct:{enc}"
               f"&count=patient.reaction.reactionmeddrapt.exact&limit={limit}")
        data = _get_json(url)
        if data and data.get("results"):
            return [r["term"].lower() for r in data["results"]]
        return []
    return _cached(f"ae:{drug_name}", _f)

# ════ Allergy expansion ══════════════════════════════════════════════════════
_ALLERGY_MAP = {
    "penicillin":      ["penicillin","beta-lactam","amoxicillin","ampicillin","piperacillin","amoxicillin clavulanate"],
    "nsaid":           ["nsaid","ibuprofen","naproxen","diclofenac","aspirin","celecoxib","indomethacin","ketorolac","meloxicam"],
    "sulfa":           ["sulfa","sulfonamide","sulfamethoxazole","trimethoprim-sulfamethoxazole"],
    "cephalosporin":   ["cephalosporin","beta-lactam","cephalexin","cefuroxime","ceftriaxone","cefdinir","cefixime"],
    "fluoroquinolone": ["quinolone","fluoroquinolone","ciprofloxacin","levofloxacin","moxifloxacin"],
    "macrolide":       ["macrolide","erythromycin","azithromycin","clarithromycin"],
    "statin":          ["statin","atorvastatin","simvastatin","lovastatin","pravastatin","rosuvastatin"],
    "ace inhibitor":   ["ace-inhibitor","ace inhibitor","lisinopril","enalapril","ramipril","captopril"],
    "beta blocker":    ["beta-blocker","beta blocker","metoprolol","atenolol","propranolol","carvedilol"],
    "anticoagulant":   ["anticoagulant","warfarin","heparin","rivaroxaban","apixaban"],
    "corticosteroid":  ["corticosteroid","prednisone","prednisolone","dexamethasone","hydrocortisone"],
    "opioid":          ["opioid","tramadol","codeine","morphine","oxycodone","fentanyl"],
}

def build_allergy_classes(drug_name, api_classes):
    allergy_set = {drug_name.lower()}
    for c in api_classes: allergy_set.add(c.lower())
    for group, synonyms in _ALLERGY_MAP.items():
        if any(s in drug_name.lower() or any(s in c for c in api_classes) for s in synonyms):
            allergy_set.update(synonyms)
    return list(allergy_set)

# ════ Drug info aggregator ═══════════════════════════════════════════════════
_COMMON_DRUGS = [
    "warfarin","aspirin","ibuprofen","naproxen","methotrexate","lithium","digoxin","phenytoin",
    "carbamazepine","rifampin","cimetidine","theophylline","cyclosporine","tacrolimus","metformin",
    "insulin","alcohol","clopidogrel","heparin","amiodarone","fluconazole","ketoconazole",
    "erythromycin","clarithromycin","azithromycin","ciprofloxacin","tetracycline","doxycycline",
    "amoxicillin","ampicillin","penicillin","verapamil","diltiazem","amlodipine","atorvastatin",
    "simvastatin","pravastatin","lovastatin","spironolactone","furosemide","hydrochlorothiazide",
    "enalapril","lisinopril","losartan","sertraline","fluoxetine","paroxetine","venlafaxine",
    "olanzapine","risperidone","omeprazole","pantoprazole","ranitidine","allopurinol","colchicine",
    "prednisone","dexamethasone","tramadol","codeine","morphine","oxycodone","fentanyl","naloxone",
    "metoprolol","atenolol","propranolol","paracetamol","acetaminophen","clindamycin","metronidazole",
    "gabapentin","levothyroxine","naproxen","levofloxacin","trimethoprim","sulfamethoxazole",
    "nitrofurantoin","acyclovir","valacyclovir","prednisolone","isoniazid","gemfibrozil",
    "potassium","iron","calcium","zinc","antacids","rosuvastatin","telmisartan","esomeprazole",
    "rabeprazole","etoricoxib","celecoxib","folic acid","methylcobalamin","cholecalciferol",
    "tamsulosin","pregabalin","vildagliptin","sitagliptin","empagliflozin","dapagliflozin",
    "hydroxychloroquine","carbimazole","escitalopram","cefixime","cephalexin","rifampicin",
    "bisoprolol","carvedilol","ramipril","captopril","olmesartan","valsartan","candesartan",
    "glimepiride","gliclazide","glipizide","metronidazole","cloxacillin","flucloxacillin",
    "diazepam","lorazepam","alprazolam","clonazepam","zolpidem","quetiapine","aripiprazole",
    "duloxetine","mirtazapine","bupropion","valproate","lamotrigine","levetiracetam",
    "donepezil","rivastigmine","memantine","salbutamol","budesonide","fluticasone","montelukast",
    "cetirizine","loratadine","fexofenadine","aceclofenac","nimesulide","diclofenac","meloxicam",
    "betahistine","cinnarizine","ondansetron","domperidone","metoclopramide","lactulose","bisacodyl",
    "liraglutide","semaglutide","rivaroxaban","apixaban","dabigatran","febuxostat","allopurinol",
    "colchicine","sulfasalazine","leflunomide","oseltamivir","ivermectin","albendazole",
    "ferrous sulfate","ferrous ascorbate","folic acid","vitamin d","calcitriol","alendronate",
    "tamsulosin","sildenafil","tadalafil","finasteride","pregabalin","gabapentin",
]

def _extract_drug_names_from_text(text):
    tl = text.lower()
    return [d for d in _COMMON_DRUGS if re.search(r'\b' + re.escape(d) + r'\b', tl)]

def get_drug_info(drug_name):
    info = {
        "rxcui":None,"canonical_name":drug_name,"classes":[],
        "dosage_text":"","warnings_text":"","interactions_text":"",
        "drug_interactions_mentioned":[],"allergy_classes":[],
        "adverse_events":[],"found":False,
        "source":f"{'Live APIs' if _ONLINE else 'Mock APIs'} — {_MODE} mode",
        "mode":_MODE,
    }
    rxcui = rxnorm_get_rxcui(drug_name)
    if not rxcui: return info
    info.update({"found":True,"rxcui":rxcui})
    props = rxnorm_get_properties(rxcui)
    if props.get("name"): info["canonical_name"] = props["name"]
    classes = rxclass_get_classes(rxcui)
    info["classes"] = classes
    info["allergy_classes"] = build_allergy_classes(drug_name, classes)
    label = openfda_get_label(drug_name)
    if label:
        def _txt(key):
            v = label.get(key) or []
            return (" ".join(v) if isinstance(v,list) else str(v))[:2500]
        info["dosage_text"]    = _txt("dosage_and_administration") or _txt("dosage_forms_and_strengths")
        info["warnings_text"]  = _txt("warnings") or _txt("boxed_warning") or _txt("warnings_and_cautions")
        info["interactions_text"] = _txt("drug_interactions")
        info["drug_interactions_mentioned"] = _extract_drug_names_from_text(info["interactions_text"])
    info["adverse_events"] = openfda_get_adverse_events(drug_name)
    return info

# ════ Dosage ══════════════════════════════════════════════════════════════════
_MAX_DOSE_FALLBACK = {
    "amoxicillin":(875,3000),"ibuprofen":(800,3200),"warfarin":(15,15),
    "metformin":(1000,2550),"lisinopril":(40,80),"aspirin":(1000,4000),
    "metoprolol":(200,400),"atorvastatin":(80,80),"omeprazole":(40,120),
    "paracetamol":(1000,4000),"acetaminophen":(1000,4000),
    "ciprofloxacin":(750,1500),"azithromycin":(500,2000),"naproxen":(500,1500),
    "sertraline":(200,200),"clopidogrel":(75,75),"tramadol":(100,400),
    "prednisone":(80,300),"doxycycline":(100,200),"metronidazole":(500,2000),
    "clindamycin":(450,1800),"gabapentin":(1200,3600),"levofloxacin":(750,750),
    "amlodipine":(10,10),"simvastatin":(40,80),"losartan":(100,100),
    "levothyroxine":(0.3,0.3),"furosemide":(80,600),"spironolactone":(100,400),
    "rosuvastatin":(40,40),"valsartan":(320,320),"telmisartan":(80,80),
    "glimepiride":(8,8),"gliclazide":(120,320),"glipizide":(20,40),
    "escitalopram":(20,20),"fluoxetine":(80,80),"venlafaxine":(225,225),
    "quetiapine":(800,800),"pregabalin":(600,600),"diclofenac":(150,150),
    "celecoxib":(400,400),"etoricoxib":(90,90),
}

FREQ_DPD = {
    "once daily":1,"od":1,"qd":1,"q24h":1,"every 24 hours":1,"once a day":1,
    "twice daily":2,"bid":2,"bd":2,"q12h":2,"every 12 hours":2,"twice a day":2,
    "three times daily":3,"tid":3,"tds":3,"q8h":3,"every 8 hours":3,
    "thrice daily":3,"thrice a day":3,"three times a day":3,"3 times daily":3,
    "four times daily":4,"qid":4,"qds":4,"q6h":4,"every 6 hours":4,"four times a day":4,
    "every 4 hours":6,"q4h":6,
    "at bedtime":1,"morning":1,"evening":1,"nightly":1,"at night":1,"once":1,
    "weekly":0.14,"monthly":0.033,
}

def parse_max_dose(dosage_text, drug_name):
    if dosage_text:
        hits = [float(m.group(1)) for m in re.finditer(
            r'(?:maximum|max|not exceed|do not exceed|up to)[^.]{0,80}?(\d+(?:\.\d+)?)\s*mg',
            dosage_text, re.IGNORECASE)]
        if hits:
            hits.sort()
            return hits[0], hits[-1] if hits[-1] > hits[0] else hits[0]*4
    return _MAX_DOSE_FALLBACK.get(drug_name.lower(), (None,None))

# ════ Prescription Parser ═════════════════════════════════════════════════════
from brand_names import BRAND_TO_GENERIC, resolve_brand

_ALL_BRAND_NAMES = sorted(list(BRAND_TO_GENERIC.keys()), key=len, reverse=True)
_ALL_GENERIC_NAMES = list(dict.fromkeys(_COMMON_DRUGS + list(_MAX_DOSE_FALLBACK.keys())))

def _normalize_freq(text):
    tl = text.lower()
    for k in sorted(FREQ_DPD.keys(), key=len, reverse=True):
        if k in tl: return k
    return None

def _normalize_route(text):
    tl = text.lower()
    if re.search(r"\b(injection|iv\b|intravenous|intramuscular|im\b|subcutaneous|sc\b)\b", tl): return "injection"
    if re.search(r"\b(topical|cream|ointment|patch|gel|drops?)\b", tl): return "topical"
    if re.search(r"\b(inhaler|nebulizer|inhale)\b", tl): return "inhaled"
    return "oral"

def parse_prescription(text):
    results, seen = [], set()
    lines_raw = re.split(r"\n|(?<=\.)\s+(?=(?:Tab|Cap|Syp|Inj|Drop|Sachet)\b)", text, flags=re.IGNORECASE)
    route_global = _normalize_route(text)

    for line in lines_raw:
        line = line.strip()
        if not line or len(line) < 4: continue
        clean = re.sub(r"^[\d\.\-\*\(\)]+\s*","", line).strip()
        clean = re.sub(r"^(?:Tab|Cap|Tablet|Capsule|Syp|Syrup|Inj|Drop|Oint|Sachet)\.?\s*","", clean, flags=re.IGNORECASE).strip()

        freq     = _normalize_freq(line)
        route    = _normalize_route(line) if _normalize_route(line)!="oral" else route_global
        dur_m    = re.search(r"for\s+(\d+)\s*(days?|weeks?|months?)", line, re.IGNORECASE)
        duration = f"{dur_m.group(1)} {dur_m.group(2)}".strip() if dur_m else "not specified"
        dose_m   = re.search(r"\(?\s*(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)\s*\)?", line, re.IGNORECASE)
        dose     = float(dose_m.group(1)) if dose_m else None
        unit     = dose_m.group(2).lower() if dose_m else "tablet/cap"

        # Brand match
        matched_brand, matched_info = None, None
        for brand in _ALL_BRAND_NAMES:
            if re.search(r"\b" + re.escape(brand) + r"\b", clean, re.IGNORECASE):
                matched_brand, matched_info = brand, BRAND_TO_GENERIC[brand]
                break

        if matched_info:
            generic = matched_info["generic"]
            if generic in seen: continue
            seen.add(generic)
            results.append({
                "drug":generic,"brand_name":matched_brand.title(),
                "display_name":matched_info["display"],
                "components":matched_info.get("components",[generic]),
                "drug_class":matched_info.get("class",""),
                "dose":dose,"unit":unit,"frequency":freq or "not specified",
                "duration":duration,"route":route,"resolved_from_brand":True,
            })
            continue

        # Generic scan
        for gname in sorted(_ALL_GENERIC_NAMES, key=len, reverse=True):
            if re.search(r"\b" + re.escape(gname) + r"\b", clean, re.IGNORECASE):
                if gname in seen: break
                seen.add(gname)
                results.append({
                    "drug":gname,"brand_name":None,
                    "display_name":gname.title(),
                    "components":[gname],"drug_class":"",
                    "dose":dose,"unit":unit,"frequency":freq or "not specified",
                    "duration":duration,"route":route,"resolved_from_brand":False,
                })
                break

    allergies = []
    am = re.search(r"allerg(?:y|ies|ic)[^\n:]*[:]\s*([^\n]+)", text, re.IGNORECASE)
    if am:
        allergies = [a.strip().lower() for a in re.split(r"[,;/]", am.group(1)) if a.strip()]
    nka = bool(re.search(r"no known (?:drug )?allerg", text, re.IGNORECASE))
    return results, allergies, nka

# ════ Error Detection (original API-based) ═══════════════════════════════════
def detect_errors(drugs_list, allergies, drug_info_map):
    alerts, drug_names = [], [d["drug"] for d in drugs_list]
    for item in drugs_list:
        drug, dose, unit, freq = item["drug"], item["dose"], item["unit"], item["frequency"]
        info = drug_info_map.get(drug, {})
        if not info.get("found"):
            alerts.append({"type":"unknown","severity":"MEDIUM","drug":drug,"source":"RxNorm API",
                           "message":f"'{drug.title()}' was not found in RxNorm. Please verify drug name."})
            continue
        src = info.get("source","")
        # Dosage check
        if unit == "mg":
            max_s, max_d = parse_max_dose(info.get("dosage_text",""), drug)
            if max_s and dose and dose > max_s:
                alerts.append({"type":"dosage","severity":"HIGH","drug":drug,"source":src,
                               "message":f"Single dose {dose}mg exceeds FDA max single dose of {max_s}mg for {drug.title()}."})
            dpd = next((v for k,v in FREQ_DPD.items() if k in freq.lower()),None) if freq else None
            if dpd and max_d and dose:
                daily = dose * dpd
                if daily > max_d:
                    alerts.append({"type":"dosage","severity":"HIGH","drug":drug,"source":src,
                                   "message":f"Daily dose {daily}mg ({dose}mg × {dpd}/day) exceeds FDA max {max_d}mg for {drug.title()}."})
        # Interaction check (API-based)
        label_ix = info.get("drug_interactions_mentioned",[])
        for other in drug_names:
            if other == drug: continue
            pair = tuple(sorted([drug, other]))
            dup = any(a["type"]=="interaction" and tuple(sorted([a.get("drug1",""),a.get("drug2","")])) == pair for a in alerts)
            if not dup and other in label_ix:
                alerts.append({"type":"interaction","severity":"HIGH","drug1":drug,"drug2":other,"source":src,
                               "message":f"FDA label for {drug.title()} warns about interaction with {other.title()}."})
            other_ix = (drug_info_map.get(other) or {}).get("drug_interactions_mentioned",[])
            dup2 = any(a["type"]=="interaction" and tuple(sorted([a.get("drug1",""),a.get("drug2","")])) == pair for a in alerts)
            if not dup2 and drug in other_ix:
                alerts.append({"type":"interaction","severity":"HIGH","drug1":drug,"drug2":other,
                               "source":(drug_info_map.get(other) or {}).get("source",""),
                               "message":f"FDA label for {other.title()} warns about interaction with {drug.title()}."})
        # Allergy check
        ac = info.get("allergy_classes",[])
        for allergy in allergies:
            al = allergy.lower().strip()
            if any(al in cls or cls in al for cls in ac):
                cls_display = ", ".join(c for c in info.get("classes",[])[:3]) or "its class"
                alerts.append({"type":"allergy","severity":"CRITICAL","drug":drug,"source":f"{src} + RxClass",
                               "message":f"ALLERGY CONFLICT — Patient allergic to '{allergy}'. {drug.title()} belongs to {cls_display}."})
    return alerts

# ════ Report Generation ═══════════════════════════════════════════════════════
def _generate_csv_report(analysis):
    buf = io.StringIO()
    w = csv.writer(buf)
    ts = analysis.get("timestamp","")
    w.writerow(["P-helper ML/NLP Prescription Safety Report"])
    w.writerow(["Generated", ts])
    w.writerow(["Mode", analysis.get("mode","")])
    w.writerow([])
    scores = (analysis.get("ml_result") or {}).get("risk_assessment",{})
    w.writerow(["RISK ASSESSMENT (ML)"])
    w.writerow(["Risk Level", scores.get("level","N/A")])
    w.writerow(["Risk Score", scores.get("score","N/A")])
    w.writerow(["Model", scores.get("model","GradientBoosting")])
    w.writerow([])
    w.writerow(["SUMMARY"])
    s = analysis.get("summary",{})
    for k,v in s.items(): w.writerow([k.replace("_"," ").title(), v])
    w.writerow([])
    w.writerow(["DRUGS DETECTED"])
    w.writerow(["Generic Name","Brand","Drug Class","RxCUI","Dose","Unit","Frequency","Duration","Route","API Found","NLP Match Type"])
    drugs = analysis.get("drugs",[])
    norm_map = {n["original"]:n for n in (analysis.get("ml_result") or {}).get("drug_normalizations",[])}
    for d in drugs:
        nm = norm_map.get(d.get("drug",""),{})
        w.writerow([d.get("drug",""),d.get("brand_name",""),d.get("drug_class",""),
                    d.get("rxcui",""),d.get("dose",""),d.get("unit",""),
                    d.get("frequency",""),d.get("duration",""),d.get("route",""),
                    d.get("api_found",""),nm.get("match_type","regex")])
    w.writerow([])
    w.writerow(["ALERTS"])
    w.writerow(["Type","Severity","Drug/Pair","Message","Source"])
    for a in analysis.get("alerts",[]):
        pair = a.get("drug","") or f"{a.get('drug1','')} + {a.get('drug2','')}"
        w.writerow([a.get("type",""),a.get("severity",""),pair,a.get("message",""),a.get("source","")])
    w.writerow([])
    w.writerow(["ML INTERACTIONS (Classifier)"])
    w.writerow(["Drug 1","Drug 2","Severity","ML Confidence","Reasoning"])
    for i in (analysis.get("ml_result") or {}).get("ml_interactions",[]):
        w.writerow([i.get("drug1"),i.get("drug2"),i.get("severity"),
                    f"{i.get('ml_confidence')}%",i.get("ml_reasoning","")])
    w.writerow([])
    w.writerow(["NLP ENTITIES EXTRACTED"])
    w.writerow(["Category","Entities"])
    for cat,items in ((analysis.get("ml_result") or {}).get("ner_entities",{}) or {}).items():
        w.writerow([cat, ", ".join(items)])
    buf.seek(0)
    return buf.getvalue()

def _generate_html_report(analysis):
    drugs = analysis.get("drugs",[])
    alerts = analysis.get("alerts",[])
    ml = analysis.get("ml_result") or {}
    scores = ml.get("risk_assessment",{})
    risk = scores.get("level","N/A")
    risk_color = {"LOW":"#10b981","MODERATE":"#f59e0b","HIGH":"#ef4444","CRITICAL":"#7c3aed"}.get(risk,"#6b7280")
    ts = analysis.get("timestamp","")

    drugs_rows = ""
    for d in drugs:
        drugs_rows += f"""<tr>
          <td><strong>{d.get('drug','').title()}</strong>{f'<br><small style="color:#6b7280">({d.get("brand_name","")})</small>' if d.get("brand_name") else ''}</td>
          <td>{d.get('drug_class','')}</td>
          <td>{d.get('rxcui','—')}</td>
          <td>{d.get('dose','—')} {d.get('unit','')}</td>
          <td>{d.get('frequency','—')}</td>
          <td>{d.get('duration','—')}</td>
          <td>{'<span style="color:#10b981">✓</span>' if d.get('api_found') else '<span style="color:#f59e0b">?</span>'}</td>
        </tr>"""

    alert_rows = ""
    sev_colors = {"CRITICAL":"#7c3aed","HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#6b7280"}
    for a in alerts:
        sc = sev_colors.get(a.get("severity","LOW"),"#6b7280")
        pair = a.get("drug","").title() or f"{a.get('drug1','').title()} + {a.get('drug2','').title()}"
        alert_rows += f"""<tr>
          <td style="color:{sc};font-weight:700">{a.get('severity','')}</td>
          <td>{a.get('type','').title()}</td>
          <td><strong>{pair}</strong></td>
          <td>{a.get('message','')}</td>
        </tr>"""

    ml_int_rows = ""
    for i in ml.get("ml_interactions",[]):
        sc = "#ef4444" if i["severity"]=="MAJOR" else "#f59e0b"
        ml_int_rows += f"""<tr>
          <td style="color:{sc};font-weight:700">{i['severity']}</td>
          <td>{i['drug1'].title()} + {i['drug2'].title()}</td>
          <td>{i.get('ml_confidence','?')}%</td>
          <td>{i.get('ml_reasoning','')}</td>
        </tr>"""

    ner_rows = ""
    for cat, items in (ml.get("ner_entities",{}) or {}).items():
        if items:
            tags = "".join(f'<span style="background:#e0f2fe;color:#0369a1;padding:2px 8px;border-radius:4px;font-size:11px;margin:2px;display:inline-block">{x}</span>' for x in items)
            ner_rows += f"<tr><td style='font-weight:600'>{cat}</td><td>{tags}</td></tr>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>P-helper ML Report — {ts[:10]}</title>
<style>
  @media print{{.no-print{{display:none}}body{{margin:0;padding:20px}}}}
  body{{font-family:-apple-system,Arial,sans-serif;max-width:1100px;margin:0 auto;padding:32px;color:#1e293b;background:#f8faff}}
  .header{{background:linear-gradient(135deg,#1e3a5f,#3b82f6);color:#fff;padding:28px 32px;border-radius:12px;margin-bottom:24px}}
  .header h1{{margin:0 0 6px;font-size:22px}}.header p{{margin:0;opacity:.8;font-size:13px}}
  .score-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}}
  .score-card{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
  .score-num{{font-size:30px;font-weight:800}}.score-label{{font-size:11px;color:#64748b;margin-top:4px;text-transform:uppercase;letter-spacing:.5px}}
  .section{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;margin-bottom:20px;overflow:hidden}}
  .st{{background:#f1f5f9;padding:11px 20px;font-weight:700;font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:#475569;border-bottom:1px solid #e2e8f0}}
  table{{width:100%;border-collapse:collapse}}
  th{{background:#f8faff;padding:9px 13px;text-align:left;font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
  td{{padding:9px 13px;font-size:12px;border-bottom:1px solid #f1f5f9}}
  tr:last-child td{{border-bottom:none}}
  .empty{{padding:14px 20px;color:#94a3b8;font-style:italic;font-size:13px}}
  .print-btn{{background:#3b82f6;color:#fff;border:none;padding:10px 20px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;margin-right:8px}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid {risk_color}44;background:{risk_color}15;color:{risk_color}}}
  .footer{{margin-top:28px;padding-top:14px;border-top:1px solid #e2e8f0;font-size:11px;color:#94a3b8;text-align:center;line-height:1.8}}
  .ml-badge{{display:inline-block;background:#dbeafe;color:#1d4ed8;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:600;margin-left:6px}}
</style></head><body>
<div class="no-print" style="text-align:right;margin-bottom:16px">
  <button class="print-btn" onclick="window.print()">🖨️ Print / Save PDF</button>
</div>
<div class="header">
  <h1>💊 P-helper — AI Prescription Safety Report <span class="ml-badge">ML ENHANCED</span></h1>
  <p>Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')} &nbsp;|&nbsp;
     {len(drugs)} drugs analyzed &nbsp;|&nbsp; Risk: <strong>{risk}</strong> (score: {scores.get('score','?')}/100) &nbsp;|&nbsp;
     APIs: {analysis.get('api_sources',[''])[0] if analysis.get('api_sources') else '—'}</p>
</div>

<div class="score-grid">
  <div class="score-card"><div class="score-num" style="color:{risk_color}">{risk}</div><div class="score-label">ML Risk Level</div></div>
  <div class="score-card"><div class="score-num" style="color:#3b82f6">{scores.get('score','?')}</div><div class="score-label">Risk Score</div></div>
  <div class="score-card"><div class="score-num" style="color:#ef4444">{analysis.get('summary',{}).get('critical',0)}</div><div class="score-label">Critical Alerts</div></div>
  <div class="score-card"><div class="score-num" style="color:#f59e0b">{analysis.get('summary',{}).get('total_alerts',0)}</div><div class="score-label">Total Alerts</div></div>
</div>

<div class="section">
  <div class="st">💊 Medications ({len(drugs)})</div>
  {'<table><tr><th>Drug</th><th>Class</th><th>RxCUI</th><th>Dose</th><th>Frequency</th><th>Duration</th><th>API</th></tr>' + drugs_rows + '</table>' if drugs_rows else '<div class="empty">No drugs identified.</div>'}
</div>

<div class="section">
  <div class="st">⚠️ Safety Alerts ({len(alerts)})</div>
  {'<table><tr><th>Severity</th><th>Type</th><th>Drug(s)</th><th>Details</th></tr>' + alert_rows + '</table>' if alert_rows else '<div class="empty">✅ No safety alerts found.</div>'}
</div>

<div class="section">
  <div class="st">🤖 ML Interaction Classifier ({len(ml.get("ml_interactions",[]))} flagged) <span class="ml-badge">LogReg + GBM</span></div>
  {'<table><tr><th>Severity</th><th>Drug Pair</th><th>Confidence</th><th>ML Reasoning</th></tr>' + ml_int_rows + '</table>' if ml_int_rows else '<div class="empty">✅ No ML-flagged interactions (MAJOR/MODERATE).</div>'}
</div>

<div class="section">
  <div class="st">🔬 NLP Entities (BIO-NER) <span class="ml-badge">Medical NER</span></div>
  {'<table><tr><th>Category</th><th>Extracted Entities</th></tr>' + ner_rows + '</table>' if ner_rows else '<div class="empty">No entities extracted.</div>'}
</div>

<div class="section">
  <div class="st">📈 Risk Factor Breakdown <span class="ml-badge">GradientBoosting</span></div>
  <table><tr><th>Feature</th><th>Importance</th></tr>
  {''.join(f"<tr><td>{k.title()}</td><td><div style='background:#e0f2fe;display:inline-block;height:8px;border-radius:4px;width:{min(v*2,200)}px'></div> {v}%</td></tr>" for k,v in scores.get('factors',{}).items())}
  </table>
</div>

<div class="section">
  <div class="st">⚗️ ML Models Used</div>
  <table>{''.join(f"<tr><td>✓</td><td>{m}</td></tr>" for m in ml.get('models_used',[]))}</table>
</div>

<div class="footer">
  P-helper AI Engine v2.0 &nbsp;|&nbsp; Clinical decision support only — not a substitute for professional pharmacist judgment<br>
  API Sources: OpenFDA · RxNorm · RxClass &nbsp;|&nbsp; ML: scikit-learn LogReg · GradientBoosting · Complement Naive Bayes · TF-IDF NER
</div>
</body></html>"""

# ════ Flask Routes ═════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify({"mode":_MODE,"online":_ONLINE,"ml":"active","version":"2.0-ML"})

@app.route("/analyze", methods=["POST"])
def analyze():
    text = ""
    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        raw = f.read()
        if f.filename.lower().endswith(".pdf"):
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            except Exception as e:
                text = f"[PDF error: {e}]"
        else:
            for enc in ["utf-8","latin-1","ascii"]:
                try: text = raw.decode(enc); break
                except: pass
    elif "text" in request.form and request.form["text"].strip():
        text = request.form["text"]
    else:
        return jsonify({"error":"No input provided"}), 400

    # Parse
    drugs, allergies, nka = parse_prescription(text)

    # API calls (parallel)
    drug_info_map = {}
    lock = threading.Lock()
    def _fetch(name):
        i = get_drug_info(name)
        with lock: drug_info_map[name] = i
    threads = [threading.Thread(target=_fetch, args=(d["drug"],)) for d in drugs]
    for t in threads: t.start()
    for t in threads: t.join(timeout=15)

    # API-based error detection
    alerts = detect_errors(drugs, allergies, drug_info_map)
    alerts.sort(key=lambda x: {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}.get(x.get("severity","LOW"),3))

    # Enrich drugs
    enriched = []
    for item in drugs:
        info = drug_info_map.get(item["drug"],{})
        enriched.append({
            **item,
            "rxcui": info.get("rxcui"),
            "canonical_name": info.get("canonical_name", item["drug"]),
            "display_name": item.get("display_name") or info.get("canonical_name") or item["drug"].title(),
            "brand_name": item.get("brand_name"),
            "drug_class": item.get("drug_class") or "",
            "components": item.get("components", [item["drug"]]),
            "classes": info.get("classes",[])[:5],
            "adverse_events": info.get("adverse_events",[])[:6],
            "api_found": info.get("found",False),
            "api_source": info.get("source",""),
            "dosage_snippet": (info.get("dosage_text") or "")[:280],
            "warnings_snippet": (info.get("warnings_text") or "")[:200],
        })

    # ── ML/NLP LAYER ──────────────────────────────────────────────────────────
    ml_result = None
    try:
        pipeline = get_pipeline()
        ml_result = pipeline.run(text, enriched, alerts, drug_info_map)
        # Merge ML interactions into alerts (avoid duplicates)
        existing_pairs = set(
            tuple(sorted([a.get("drug1",""),a.get("drug2","")]))
            for a in alerts if a.get("type")=="interaction"
        )
        for ml_int in ml_result.get("ml_interactions",[]):
            pair = tuple(sorted([ml_int["drug1"], ml_int["drug2"]]))
            if pair not in existing_pairs:
                alerts.append({
                    "type":"interaction",
                    "severity": "HIGH" if ml_int["severity"]=="MAJOR" else "MEDIUM",
                    "drug1": ml_int["drug1"],"drug2": ml_int["drug2"],
                    "source": ml_int["source"],
                    "message": f"[ML Classifier] {ml_int['severity']} interaction predicted between "
                               f"{ml_int['drug1'].title()} and {ml_int['drug2'].title()}. "
                               f"Reason: {ml_int['ml_reasoning']}. "
                               f"ML confidence: {ml_int['ml_confidence']}%",
                })
                existing_pairs.add(pair)
        alerts.sort(key=lambda x: {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}.get(x.get("severity","LOW"),3))

        # Update drug classes from ML normalizer
        norm_map = {n["original"]:n for n in ml_result.get("drug_normalizations",[])}
        for ed in enriched:
            nm = norm_map.get(ed["drug"])
            if nm and nm.get("drug_class") and not ed.get("drug_class"):
                ed["drug_class"] = nm["drug_class"]
    except Exception as e:
        ml_result = {"ml_active": False, "error": str(e)}

    api_label = (
        "OpenFDA · RxNorm · RxClass (LIVE)"
        if _ONLINE else
        "OpenFDA · RxNorm · RxClass (mock — identical API shapes)"
    )

    response = {
        "raw_text": text[:2000],
        "drugs": enriched,
        "allergies": allergies,
        "no_known_allergies": nka,
        "alerts": alerts,
        "mode": _MODE,
        "api_sources": [api_label],
        "ml_result": ml_result,
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total_drugs": len(drugs),
            "total_alerts": len(alerts),
            "critical": sum(1 for a in alerts if a.get("severity")=="CRITICAL"),
            "high":     sum(1 for a in alerts if a.get("severity")=="HIGH"),
            "medium":   sum(1 for a in alerts if a.get("severity")=="MEDIUM"),
            "ml_risk_level": (ml_result or {}).get("risk_assessment",{}).get("level","N/A"),
            "ml_risk_score":  (ml_result or {}).get("risk_assessment",{}).get("score","N/A"),
        },
    }
    return jsonify(response)

@app.route("/report/<fmt>", methods=["POST"])
def download_report(fmt):
    analysis = request.get_json()
    if not analysis:
        return jsonify({"error":"No data"}), 400
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if fmt == "json":
        buf = io.BytesIO(json.dumps(analysis, indent=2).encode())
        buf.seek(0)
        return send_file(buf, mimetype="application/json", as_attachment=True,
                         download_name=f"P-helper {ts}.json")
    elif fmt == "csv":
        csv_text = _generate_csv_report(analysis)
        buf = io.BytesIO(csv_text.encode())
        buf.seek(0)
        return send_file(buf, mimetype="text/csv", as_attachment=True,
                         download_name=f"p-helper{ts}.csv")
    elif fmt == "pdf":
        html = _generate_html_report(analysis)
        buf = io.BytesIO(html.encode())
        buf.seek(0)
        return send_file(buf, mimetype="text/html", as_attachment=False,
                         download_name=f"p-helper{ts}.html")
    return jsonify({"error":"Invalid format"}), 400

@app.route("/drug-info/<drug_name>")
def drug_info_endpoint(drug_name):
    info = get_drug_info(drug_name.lower())
    ml = get_pipeline()
    normalized, match_type, conf = ml.normalizer.normalize(drug_name)
    info["ml_normalized"] = normalized
    info["ml_match_type"] = match_type
    info["ml_confidence"] = conf
    return jsonify(info)

@app.route("/sample")
def sample():
    return jsonify({"text": """PRESCRIPTION
Patient: John Doe  Age: 52  DOB: 1972-03-14
Allergies: Penicillin, NSAIDs

Rx:
1. Amoxicillin 1000 mg  three times daily  for 10 days
2. Ibuprofen 800 mg  four times daily  for 7 days
3. Warfarin 5 mg  once daily
4. Metformin 1000 mg  twice daily

Dr. Smith  License: 12345
Date: 2026-02-26"""})

if __name__ == "__main__":
    # Pre-warm ML pipeline at startup
    print("[P-helper] Initializing ML/NLP pipeline...")
    get_pipeline()
    print("[P-helper] Starting server on ....")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
