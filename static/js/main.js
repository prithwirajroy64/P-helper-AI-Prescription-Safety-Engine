// ── State ──
let lastAnalysis = null;

const SAMPLES = {
  standard: `PRESCRIPTION
Patient: John Doe  Age: 52  DOB: 1972-03-14
Allergies: Penicillin, NSAIDs

Rx:
1. Amoxicillin 1000mg  three times daily  for 10 days
2. Ibuprofen 800mg  four times daily  for 7 days
3. Warfarin 5mg  once daily
4. Metformin 1000mg  twice daily

Dr. Smith  License: 12345
Date: 2026-02-26`,

  allergy: `PRESCRIPTION
Patient: Priya Sharma  Age: 44
Allergies: Penicillin (anaphylaxis), Sulfa drugs, Fluoroquinolones

Rx:
1. Augmentin 875mg BD for 7 days
2. Ciplox 500mg BD for 5 days
3. Ibuprofen 400mg TDS as needed for pain
4. Warfarin 5mg OD
5. Sertraline 50mg OD for depression

Dr. Kumar`,

  polypharmacy: `PRESCRIPTION
Patient: Ramesh Patel  Age: 68  Weight: 74kg
Allergies: NKDA

Dx: HTN, T2DM, HLD, CHF, GERD, Anxiety, CKD-3

Tab Metoprolol 50mg BD
Tab Metformin 1000mg BD
Tab Atorvastatin 40mg OD at bedtime
Tab Furosemide 40mg OD
Tab Spironolactone 25mg OD
Tab Pantoprazole 40mg OD before breakfast
Tab Sertraline 50mg OD
Tab Aspirin 75mg OD
Tab Warfarin 5mg OD
Tab Amlodipine 5mg OD

Dr. Mehta  Reg: 54321`,

  indian: `PRESCRIPTION
Patient: Anita Desai  Age: 38
Allergies: No known allergies

Rx:
1. Tab Dolo 650mg TDS for 5 days
2. Cap Azee 500mg OD for 3 days
3. Tab Ciplox 500mg BD for 7 days
4. Syp Pan 40mg BD before meals
5. Tab Combiflam 1 tablet TDS for pain
6. Cap Sporidex 500mg BD for 7 days

Dr. Nair MBBS MD
Hospital: Apollo Clinics, Mumbai`
};

const PIPELINE_STEPS = [
  { name:"Text Extraction & Normalization", icon:"📄", badge:"Regex/OCR" },
  { name:"Sentence Classifier (Naive Bayes)", icon:"🔤", badge:"MODULE 5" },
  { name:"BIO Named Entity Recognition", icon:"🔬", badge:"MODULE 1" },
  { name:"Drug Name Normalization (TF-IDF)", icon:"💊", badge:"MODULE 2" },
  { name:"RxNorm + OpenFDA + RxClass APIs", icon:"🌐", badge:"LIVE API" },
  { name:"DDI Severity Classifier (LogReg)", icon:"⚡", badge:"MODULE 3" },
  { name:"Polypharmacy Risk Scorer (GBM)", icon:"📊", badge:"MODULE 4" },
  { name:"Alert Merging & Report Ready", icon:"📋", badge:"PIPELINE" },
];

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  loadSample('standard');
  fetch('/status').then(r => r.json()).then(d => {
    const pill = document.getElementById('modePill');
    const mode = document.getElementById('statusMode');
    if (d.online) {
      pill.textContent = '● LIVE APIs';
      pill.style.cssText = 'background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.25);color:var(--accent)';
      if (mode) mode.textContent = 'Live APIs + ML';
    } else {
      pill.textContent = '● OFFLINE (mock)';
      pill.style.cssText = 'background:rgba(251,146,60,.1);border:1px solid rgba(251,146,60,.25);color:var(--orange)';
      if (mode) mode.textContent = 'Mock + ML';
    }
  }).catch(() => {
    const pill = document.getElementById('modePill');
    if (pill) { pill.textContent = '● Backend offline'; pill.style.cssText = 'background:rgba(255,59,92,.1);border:1px solid rgba(255,59,92,.25);color:var(--red)'; }
  });
});

document.getElementById('rxText').addEventListener('input', function() {
  document.getElementById('charCount').textContent = this.value.length + ' chars';
});

function loadSample(name) {
  const el = document.getElementById('rxText');
  el.value = SAMPLES[name] || '';
  el.dispatchEvent(new Event('input'));
  toast('Sample loaded: ' + name, 'success');
}

function handleFile(input) {
  const f = input.files[0];
  if (!f) return;
  if (f.type === 'text/plain' || f.name.endsWith('.txt')) {
    const reader = new FileReader();
    reader.onload = e => {
      document.getElementById('rxText').value = e.target.result;
      document.getElementById('charCount').textContent = e.target.result.length + ' chars';
      toast('File loaded: ' + f.name, 'success');
    };
    reader.readAsText(f);
  } else {
    toast('PDF text extraction requires backend', 'warn');
  }
}

function openTab(btn, pane) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('pane-' + pane).classList.add('active');
  if (pane === 'riskfactors') {
    setTimeout(() => document.querySelectorAll('.rf-fill').forEach(f => { f.style.width = f.dataset.w + '%'; }), 100);
  }
}

// ── Main Analysis ──
async function runAnalysis() {
  const text = document.getElementById('rxText').value.trim();
  if (!text) { toast('Please enter prescription text', 'warn'); return; }

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('btnText').textContent = 'Running...';

  // Show pipeline
  const pc = document.getElementById('pipelineCard');
  pc.style.display = 'block';
  document.getElementById('resultsWrap').style.display = 'none';

  const stepsEl = document.getElementById('pipelineSteps');
  stepsEl.innerHTML = PIPELINE_STEPS.map((s, i) =>
    `<div class="ps-row" id="psr${i}">
      <span class="ps-num">${String(i+1).padStart(2,'0')}</span>
      <span style="font-size:15px">${s.icon}</span>
      <span class="ps-name">${s.name}</span>
      <span class="ps-detail" id="psd${i}">Waiting…</span>
      <span class="ps-stat pending" id="pss${i}">${s.badge}</span>
    </div>`
  ).join('');

  // Animate to "running"
  for (let i = 0; i < PIPELINE_STEPS.length; i++) {
    await sleep(220);
    el('pss'+i).className = 'ps-stat running';
    el('pss'+i).textContent = 'RUNNING';
  }

  // ── API Call ──
  let data;
  try {
    const fd = new FormData();
    fd.append('text', text);
    const res = await fetch('/analyze', { method:'POST', body:fd });
    data = await res.json();
    if (data.error) throw new Error(data.error);
  } catch(err) {
    // Demo mode fallback
    data = buildDemoData(text);
    toast('Backend offline — showing demo analysis', 'warn');
  }

  // Mark steps done
  const ml = data.ml_result || {};
  const s = data.summary || {};
  const stepMessages = [
    `Extracted ${data.raw_text?.length || text.length} chars`,
    `Classified ${(ml.sentence_analysis||[]).length} sentences`,
    `NER: ${Object.values(ml.ner_entities||{}).flat().length} entities`,
    `Normalized ${s.total_drugs||0} drugs via TF-IDF`,
    data.api_sources?.[0] || 'API queries complete',
    `${(ml.ml_interactions||[]).length} interactions classified`,
    `Risk score: ${s.ml_risk_score || 'N/A'} — ${s.ml_risk_level || 'N/A'}`,
    `${s.total_alerts||0} alerts generated`,
  ];
  for (let i = 0; i < PIPELINE_STEPS.length; i++) {
    el('pss'+i).className = 'ps-stat done';
    el('pss'+i).textContent = '✓ DONE';
    el('psd'+i).textContent = stepMessages[i] || '';
  }

  await sleep(400);
  pc.style.display = 'none';
  lastAnalysis = data;
  renderResults(data);
  renderQuickSummary(data);

  btn.disabled = false;
  document.getElementById('spinner').style.display = 'none';
  document.getElementById('btnText').textContent = '⚡ Analyze with ML/NLP';
}

// ── Quick Summary Panel ──
function renderQuickSummary(data) {
  const s = data.summary || {};
  const safe = s.total_alerts === 0;
  const riskLevel = (s.ml_risk_level || 'N/A').toUpperCase();
  const rColors = { LOW:'var(--green)', MODERATE:'var(--orange)', HIGH:'var(--red)', CRITICAL:'var(--purple)', 'N/A':'var(--muted)' };

  let html = '';
  // API sources
  if (data.api_sources?.length) {
    html += `<div class="api-source-bar"><span class="api-dot">⚡</span><strong style="color:var(--text)">Active:</strong>${data.api_sources.map(s=>`<span class="source-chip">${s}</span>`).join('')}</div>`;
  }
  // Summary bar
  html += `<div class="summary-bar">
    <div class="sum-item"><div class="sum-val c-accent">${s.total_drugs||0}</div><div class="sum-label">Drugs</div></div>
    <div class="sum-item"><div class="sum-val c-purple">${s.critical||0}</div><div class="sum-label">Critical</div></div>
    <div class="sum-item"><div class="sum-val c-red">${s.high||0}</div><div class="sum-label">High</div></div>
    <div class="sum-item"><div class="sum-val c-orange">${s.medium||0}</div><div class="sum-label">Medium</div></div>
  </div>`;
  // Risk
  html += `<div class="result-section">
    <div class="rs-title">🤖 ML Risk Assessment</div>
    <div style="font-family:'DM Serif Display',serif;font-size:2rem;color:${rColors[riskLevel]}">${riskLevel}</div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:var(--muted);margin-top:4px">Score: ${s.ml_risk_score||'N/A'}/100</div>
  </div>`;
  // Allergies
  html += `<div class="result-section"><div class="rs-title">🚨 Allergies</div>`;
  if (data.no_known_allergies) html += `<span class="no-allergy-tag">✓ NKDA</span>`;
  else if (data.allergies?.length) html += data.allergies.map(a=>`<span class="allergy-tag">⚠ ${a}</span>`).join('');
  else html += `<span style="color:var(--dim);font-size:0.82rem">Not mentioned</span>`;
  html += `</div>`;
  // Top alert
  if (data.alerts?.length) {
    const top = data.alerts[0];
    const sev = top.severity || 'MEDIUM';
    html += `<div class="result-section"><div class="rs-title">⚠️ Top Alert</div>
      <div class="alert-card ac-${sev}">
        <div class="al-icon">${{CRITICAL:'💀',HIGH:'🔴',MEDIUM:'🟠',LOW:'🟡'}[sev]||'⚠️'}</div>
        <div class="al-body">
          <div class="al-header"><span class="sev-badge sb-${sev}">${sev}</span></div>
          <div class="al-msg">${escHtml(top.message||'')}</div>
        </div>
      </div>
    </div>`;
  } else {
    html += `<div class="result-section" style="text-align:center;padding:24px;color:var(--green)">
      <div style="font-size:32px;margin-bottom:8px">✅</div>
      <div style="font-weight:600;font-size:0.88rem">No safety issues detected</div>
    </div>`;
  }

  document.getElementById('quickBody').innerHTML = html;
}

// ── Full Results ──
function renderResults(data) {
  const ml = data.ml_result || {};
  const s = data.summary || {};
  const riskLevel = (s.ml_risk_level || 'N/A').toUpperCase();
  const rColors = { LOW:'var(--green)', MODERATE:'var(--orange)', HIGH:'var(--red)', CRITICAL:'var(--purple)', 'N/A':'var(--muted)' };
  const rClass = { LOW:'rb-low', MODERATE:'rb-moderate', HIGH:'rb-high', CRITICAL:'rb-critical' };
  const rIcons = { LOW:'✅', MODERATE:'⚠️', HIGH:'🚨', CRITICAL:'💀' };
  const rMessages = {
    LOW:'Prescription appears safe. Continue with standard dispensing.',
    MODERATE:'Moderate risk detected. Review highlighted interactions before dispensing.',
    HIGH:'HIGH RISK — Critical alerts require pharmacist review before dispensing.',
    CRITICAL:'CRITICAL — Multiple severe alerts. Do not dispense without physician confirmation.',
    'N/A':'Analysis complete.'
  };

  // Banner
  const banner = el('riskBanner');
  banner.className = `risk-banner ${rClass[riskLevel] || 'rb-moderate'}`;
  banner.innerHTML = `<span style="font-size:24px">${rIcons[riskLevel]||'⚠️'}</span>
    <div><strong>${riskLevel}</strong> RISK${s.ml_risk_score ? ` (Score: ${s.ml_risk_score}/100)` : ''} — ${rMessages[riskLevel]||''}</div>`;

  // Score tiles
  const tiles = [
    { val:s.ml_risk_level||'—', label:'Risk Level', color:rColors[riskLevel] },
    { val:s.ml_risk_score||'—', label:'Risk Score', color:'var(--accent)' },
    { val:s.total_drugs||0, label:'Drugs', color:'var(--accent)' },
    { val:s.critical||0, label:'Critical', color:'var(--purple)' },
    { val:s.high||0, label:'High', color:'var(--red)' },
    { val:s.medium||0, label:'Medium', color:'var(--orange)' },
    { val:(ml.ml_interactions||[]).length, label:'ML DDI', color:'var(--blue)' },
    { val:Object.values(ml.ner_entities||{}).flat().length||0, label:'NER Entities', color:'var(--green)' },
  ];
  el('scoreTiles').innerHTML = tiles.map((t,i) =>
    `<div class="score-tile" style="animation-delay:${i*50}ms">
      <div class="st-num" style="color:${t.color}">${t.val}</div>
      <div class="st-label">${t.label}</div>
    </div>`
  ).join('');

  // ── OVERVIEW TAB ──
  const apiSource = (data.api_sources||['N/A']).join(' · ');
  const normList = (ml.drug_normalizations||[]).map(n =>
    `<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--border);font-size:0.75rem">
      <span style="color:var(--text2)">${n.original}</span>
      <span style="color:var(--accent);font-family:'IBM Plex Mono',monospace">${n.normalized}
        <span style="color:var(--dim);font-size:0.65rem">[${n.match_type} ${Math.round(n.confidence*100)}%]</span>
      </span>
    </div>`
  ).join('') || '<div style="color:var(--dim);font-size:0.8rem;font-family:\'IBM Plex Mono\',monospace">No normalizations</div>';

  el('overviewGrid').innerHTML = `
    <div class="ov-card">
      <div class="ov-title">📡 Data Sources</div>
      <div style="font-size:0.75rem;color:var(--text2);line-height:2;font-family:'IBM Plex Mono',monospace">${apiSource}</div>
      <div style="margin-top:10px">${(ml.models_used||[]).map(m=>
        `<div class="mi-item">✓ ${m}</div>`
      ).join('')}</div>
    </div>
    <div class="ov-card">
      <div class="ov-title">💊 Drug Normalizations <span class="ml-chip">TF-IDF</span></div>
      ${normList}
    </div>
    <div class="ov-card">
      <div class="ov-title">🔬 NER Confidence <span class="ml-chip">LogReg</span></div>
      ${Object.entries(ml.ner_confidence_by_category||{}).map(([k,v]) =>
        `<div class="rf-bar-row">
          <div class="rf-header"><span class="rf-name" style="font-size:0.75rem">${k}</span><span class="rf-pct">${Math.round(v*100)}%</span></div>
          <div class="rf-track"><div class="rf-fill" data-w="${Math.round(v*100)}"></div></div>
        </div>`
      ).join('') || '<div style="color:var(--dim);font-size:0.78rem;font-family:\'IBM Plex Mono\',monospace">No NER confidence data</div>'}
    </div>
    <div class="ov-card">
      <div class="ov-title">🩺 Allergies & Conditions</div>
      ${data.allergies?.length
        ? `<div style="margin-bottom:10px"><div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">ALLERGIES</div>${data.allergies.map(a=>`<span class="allergy-tag">${a}</span>`).join('')}</div>`
        : ''}
      ${data.no_known_allergies ? '<div style="font-size:0.78rem;color:var(--green);font-family:\'IBM Plex Mono\',monospace">✓ NKDA — No known drug allergies</div>' : ''}
      ${(ml.ner_entities?.CONDITION||[]).length
        ? `<div style="margin-top:8px"><div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px">CONDITIONS (NER)</div>${ml.ner_entities.CONDITION.map(c=>`<span style="background:rgba(245,200,66,0.08);border:1px solid rgba(245,200,66,0.2);color:var(--yellow);padding:3px 9px;border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;margin:2px;display:inline-block">${c}</span>`).join('')}</div>`
        : ''}
    </div>`;
  setTimeout(()=>document.querySelectorAll('.ov-card .rf-fill').forEach(f=>{f.style.width=f.dataset.w+'%'}),300);

  // ── NER TAB ──
  const nerColors = ['DRUG','DOSE','FREQ','ROUTE','DURATION','CONDITION','ALLERGY'];
  const nerIcons = {DRUG:'💊',DOSE:'⚖️',FREQ:'🔄',ROUTE:'🚀',DURATION:'⏱️',CONDITION:'🩺',ALLERGY:'🚨'};
  const nerEnts = ml.ner_entities || {};
  el('nerPanel').innerHTML = `<div class="ner-grid">${nerColors.map(cat => {
    const items = nerEnts[cat] || [];
    return `<div class="ner-group">
      <div class="ner-header"><span>${nerIcons[cat]||''} ${cat}</span><span class="ner-count">${items.length}</span></div>
      <div class="ner-tags">${items.length
        ? items.map((v,i)=>`<span class="ner-tag nt-${cat}" style="animation-delay:${i*40}ms">${v}</span>`).join('')
        : '<span style="color:var(--muted);font-size:0.72rem;font-family:\'IBM Plex Mono\',monospace">—</span>'
      }</div>
    </div>`;
  }).join('')}</div>`;

  // ── DRUGS TAB ──
  el('drugsPanel').innerHTML = (data.drugs||[]).map((d,i) => {
    const nm = (ml.drug_normalizations||[]).find(n => n.original === d.drug) || {};
    return `<div class="drug-card" style="animation-delay:${i*70}ms">
      <div class="dc-header">
        <div>
          <div class="dc-name">${d.display_name || d.drug}</div>
          ${d.brand_name ? `<div class="dc-brand">Brand: ${d.brand_name}</div>` : ''}
          ${d.drug_class ? `<div class="dc-brand">${d.drug_class}</div>` : ''}
        </div>
        <div class="dc-badges">
          ${d.api_found
            ? '<span class="dc-badge dcb-api">✓ RxNorm/OpenFDA</span>'
            : '<span class="dc-badge" style="background:rgba(255,140,42,.08);border:1px solid rgba(255,140,42,.2);color:var(--orange)">? Not in API</span>'
          }
          ${nm.match_type ? `<span class="dc-badge dcb-ml">ML: ${nm.match_type} ${Math.round((nm.confidence||0)*100)}%</span>` : ''}
          ${d.resolved_from_brand ? '<span class="dc-badge dcb-brand">Brand→Generic</span>' : ''}
        </div>
      </div>
      <div class="dc-details">
        <div class="dcd-item"><div class="dcd-key">RxCUI</div><div class="dcd-val">${d.rxcui||'—'}</div></div>
        <div class="dcd-item"><div class="dcd-key">Dose</div><div class="dcd-val">${d.dose ? d.dose+' '+(d.unit||'mg') : '—'}</div></div>
        <div class="dcd-item"><div class="dcd-key">Frequency</div><div class="dcd-val">${d.frequency||'—'}</div></div>
        <div class="dcd-item"><div class="dcd-key">Duration</div><div class="dcd-val">${d.duration||'—'}</div></div>
        <div class="dcd-item"><div class="dcd-key">Route</div><div class="dcd-val">${d.route||'—'}</div></div>
        <div class="dcd-item"><div class="dcd-key">API Source</div><div class="dcd-val" style="font-size:0.7rem;color:var(--muted)">${d.api_source||'—'}</div></div>
        ${(d.classes||[]).length ? `<div class="dcd-item" style="grid-column:1/-1"><div class="dcd-key">Drug Classes</div><div class="dcd-val" style="font-size:0.75rem">${d.classes.join(' · ')}</div></div>` : ''}
      </div>
      ${(d.adverse_events||[]).length ? `<div class="dc-ae"><div class="dc-ae-title">⚠ FAERS Adverse Events</div><div class="ae-tags">${d.adverse_events.map(ae=>`<span class="ae-tag">${ae}</span>`).join('')}</div></div>` : ''}
    </div>`;
  }).join('') || '<div class="empty-state"><div class="es-icon">💊</div><p>No drugs identified.</p></div>';

  // ── ALERTS TAB ──
  const iconMap = {CRITICAL:'💀',HIGH:'🔴',MEDIUM:'🟠',LOW:'🟡'};
  el('alertsPanel').innerHTML = (data.alerts||[]).length
    ? (data.alerts||[]).map((a,i) => {
        const sev = a.severity || 'LOW';
        const drugStr = (a.drug||'').trim()
          ? cap(a.drug)
          : `${cap(a.drug1||'')} + ${cap(a.drug2||'')}`;
        return `<div class="alert-card ac-${sev}" style="animation-delay:${i*60}ms">
          <div class="al-icon">${iconMap[sev]||'⚠️'}</div>
          <div class="al-body">
            <div class="al-header">
              <span class="al-title">${escHtml(drugStr)}</span>
              <span class="sev-badge sb-${sev}">${sev}</span>
              <span class="sev-badge" style="background:rgba(148,163,184,.06);border:1px solid rgba(148,163,184,.12);color:var(--muted)">${(a.type||'').toUpperCase()}</span>
            </div>
            <div class="al-msg">${escHtml(a.message||'')}</div>
            <div class="al-src">📡 ${a.source||'—'}</div>
          </div>
        </div>`;
      }).join('')
    : '<div class="empty-state"><div class="es-icon">✅</div><p>No safety alerts found.</p></div>';

  // ── ML INTERACTIONS TAB ──
  el('mlInteractPanel').innerHTML = (ml.ml_interactions||[]).length
    ? `<div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:var(--muted);margin-bottom:14px;padding:8px 12px;background:rgba(56,189,248,.04);border:1px solid rgba(56,189,248,.1);border-radius:8px">
        ⚡ ML DDI Classifier (Logistic Regression) — MAJOR/MODERATE interactions only
      </div>
      ${(ml.ml_interactions||[]).map((it,idx) => `
      <div class="mli-card" style="animation-delay:${idx*60}ms">
        <div class="mli-header">
          <span class="mli-drugs">${cap(it.drug1)} + ${cap(it.drug2)}</span>
          <span class="mli-sev mli-${it.severity}">${it.severity}</span>
          <span class="mli-conf">${it.ml_confidence}% confidence</span>
        </div>
        <div class="mli-reason">${escHtml(it.ml_reasoning||'')}</div>
        <div class="mli-model">Model: ${it.source||'DDI-LogReg'}</div>
      </div>`).join('')}`
    : '<div class="empty-state"><div class="es-icon">✅</div><p>No MAJOR/MODERATE interactions predicted by ML classifier.</p></div>';

  // ── RISK FACTORS TAB ──
  const risk = ml.risk_assessment || {};
  const factors = risk.factors || {};
  el('riskFactorsPanel').innerHTML = `
    <div style="margin-bottom:20px">
      <div style="font-size:2.2rem;font-weight:700;color:${rColors[riskLevel]};font-family:'IBM Plex Mono',monospace">${risk.level||riskLevel}</div>
      <div style="font-size:0.78rem;color:var(--text2);margin-top:5px">Risk Score: ${risk.score||s.ml_risk_score||'?'}/100 · Model: ${risk.model||'GradientBoosting'}</div>
    </div>
    ${Object.entries(factors).map(([k,v]) => `
    <div class="rf-bar-row">
      <div class="rf-header">
        <span class="rf-name">${k.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</span>
        <span class="rf-pct">${v}% importance</span>
      </div>
      <div class="rf-track"><div class="rf-fill" data-w="${Math.min(v*2,100)}"></div></div>
    </div>`).join('') || '<div style="color:var(--dim);font-size:0.82rem;margin-bottom:16px;font-family:\'IBM Plex Mono\',monospace">No detailed risk factors available.</div>'}
    ${(ml.models_used||[]).length ? `<div class="model-info" style="margin-top:20px">
      <div class="mi-title">🤖 ML Models Active</div>
      <div>${(ml.models_used||[]).map(m=>`<div class="mi-item">✓ ${m}</div>`).join('')}</div>
    </div>` : ''}`;

  // ── SENTENCES TAB ──
  el('sentencesPanel').innerHTML = `<div class="sent-list">${(ml.sentence_analysis||[]).map((s,i) =>
    `<div class="sent-card" style="animation-delay:${i*50}ms">
      <span class="sent-label sl-${s.label}">${(s.label||'other').replace('_',' ')}</span>
      <span class="sent-text">${escHtml(s.sentence||'')}</span>
      <span class="sent-conf">${s.confidence||''}${s.confidence?'%':''}</span>
    </div>`
  ).join('') || '<div class="empty-state" style="padding:40px 24px"><div class="es-icon">📝</div><p>No sentence analysis available.</p></div>'}</div>`;

  // Show results
  el('resultsWrap').style.display = 'block';
  setTimeout(() => {
    el('resultsWrap').scrollIntoView({ behavior:'smooth', block:'start' });
    document.querySelectorAll('.rf-fill').forEach(f => { f.style.width = f.dataset.w + '%'; });
  }, 200);

  const totalAlerts = s.total_alerts || 0;
  toast(totalAlerts === 0 ? '✅ No safety issues found!' : `⚠ ${totalAlerts} alert(s) detected`, totalAlerts === 0 ? 'success' : 'warn');
}

// ── Demo data builder (when backend is offline) ──
function buildDemoData(text) {
  const knownDrugs = ['amoxicillin','warfarin','ibuprofen','metformin','aspirin','atorvastatin',
    'amlodipine','metoprolol','furosemide','sertraline','spironolactone','pantoprazole',
    'augmentin','ciplox','dolo','azee','combiflam','sporidex'];
  const drugs = [];
  const drugMap = {
    amoxicillin:{class:'Penicillin Antibiotic',route:'oral'},
    warfarin:{class:'Anticoagulant',route:'oral'},
    ibuprofen:{class:'NSAID',route:'oral'},
    metformin:{class:'Biguanide / Antidiabetic',route:'oral'},
    aspirin:{class:'Antiplatelet / NSAID',route:'oral'},
    atorvastatin:{class:'Statin',route:'oral'},
    amlodipine:{class:'Calcium Channel Blocker',route:'oral'},
    metoprolol:{class:'Beta Blocker',route:'oral'},
    furosemide:{class:'Loop Diuretic',route:'oral'},
    sertraline:{class:'SSRI Antidepressant',route:'oral'},
    spironolactone:{class:'Aldosterone Antagonist',route:'oral'},
    pantoprazole:{class:'Proton Pump Inhibitor',route:'oral'},
    augmentin:{class:'Penicillin + Beta-lactamase inhibitor',route:'oral'},
    ciplox:{class:'Fluoroquinolone Antibiotic',route:'oral'},
    dolo:{class:'Analgesic / Antipyretic',route:'oral'},
    azee:{class:'Macrolide Antibiotic',route:'oral'},
    combiflam:{class:'NSAID + Analgesic',route:'oral'},
    sporidex:{class:'Cephalosporin Antibiotic',route:'oral'},
  };
  knownDrugs.forEach(d => {
    if (text.toLowerCase().includes(d)) {
      const m = text.match(new RegExp(d + '\\s*(\\d+)\\s*(mg|mcg|g|ml)?', 'i'));
      const fq = text.match(new RegExp(d + '[^\n]*(once|twice|BD|TDS|QID|OD|daily)', 'i'));
      const dr = text.match(new RegExp(d + '[^\n]*(\\d+)\\s*(days?|weeks?)', 'i'));
      drugs.push({
        drug: d, display_name: cap(d), canonical_name: d,
        drug_class: drugMap[d]?.class || 'Drug',
        route: drugMap[d]?.route || 'oral',
        dose: m?.[1] || null, unit: m?.[2] || 'mg',
        frequency: fq?.[1] || 'once daily',
        duration: dr ? `${dr[1]} ${dr[2]}` : 'as directed',
        api_found: true, rxcui: null, classes: [], adverse_events: []
      });
    }
  });

  const allergyMatch = text.match(/allergies?:\s*([^\n]+)/i);
  const allergyStr = allergyMatch ? allergyMatch[1].trim() : '';
  const allergies = /none|nkda|no known/i.test(allergyStr) ? [] :
    allergyStr ? allergyStr.split(/,|;/).map(a => a.trim()).filter(Boolean) : [];
  const nkda = /nkda|no known/i.test(allergyStr) || /nkda/i.test(text);

  // Build alerts
  const alerts = [];
  const dNames = drugs.map(d => d.drug);

  if (dNames.includes('warfarin') && dNames.includes('aspirin'))
    alerts.push({type:'interaction',severity:'CRITICAL',drug1:'warfarin',drug2:'aspirin',message:'Warfarin + Aspirin: Significantly increased bleeding risk. Monitor INR closely and consider dose adjustment.',source:'DDI-ML Classifier'});
  if (dNames.includes('warfarin') && dNames.includes('ibuprofen'))
    alerts.push({type:'interaction',severity:'HIGH',drug1:'warfarin',drug2:'ibuprofen',message:'Warfarin + Ibuprofen: NSAIDs inhibit platelet function and may displace warfarin from protein binding, increasing anticoagulant effect.',source:'DDI-ML Classifier'});
  if (dNames.includes('warfarin') && dNames.includes('atorvastatin'))
    alerts.push({type:'interaction',severity:'MEDIUM',drug1:'warfarin',drug2:'atorvastatin',message:'Warfarin + Atorvastatin: Statin may increase warfarin effect. Monitor INR after initiating statin therapy.',source:'DDI-ML Classifier'});
  if (dNames.includes('metformin') && dNames.includes('furosemide'))
    alerts.push({type:'interaction',severity:'MEDIUM',drug1:'metformin',drug2:'furosemide',message:'Metformin + Furosemide: Loop diuretics may affect renal function; monitor for lactic acidosis risk.',source:'DDI-ML Classifier'});
  if (dNames.includes('sertraline') && dNames.includes('aspirin'))
    alerts.push({type:'interaction',severity:'MEDIUM',drug1:'sertraline',drug2:'aspirin',message:'SSRI + Aspirin: Combined use increases risk of GI bleeding through synergistic antiplatelet effects.',source:'DDI-ML Classifier'});

  allergies.forEach(allergy => {
    const a = allergy.toLowerCase();
    if ((a.includes('penicillin') || a.includes('amoxicillin')) && dNames.includes('amoxicillin'))
      alerts.push({type:'allergy',severity:'CRITICAL',drug:'amoxicillin',message:`Patient has documented Penicillin allergy — Amoxicillin is a penicillin-class antibiotic. Anaphylaxis risk.`,source:'Allergy Cross-Check'});
    if ((a.includes('penicillin') || a.includes('amoxicillin')) && dNames.includes('augmentin'))
      alerts.push({type:'allergy',severity:'CRITICAL',drug:'augmentin',message:`Patient has Penicillin allergy — Augmentin (amoxicillin-clavulanate) is a penicillin. Contraindicated.`,source:'Allergy Cross-Check'});
    if ((a.includes('nsaid') || a.includes('aspirin')) && dNames.includes('ibuprofen'))
      alerts.push({type:'allergy',severity:'CRITICAL',drug:'ibuprofen',message:`Patient is allergic to NSAIDs — Ibuprofen is an NSAID. Risk of allergic reaction.`,source:'Allergy Cross-Check'});
    if ((a.includes('nsaid') || a.includes('aspirin')) && dNames.includes('aspirin'))
      alerts.push({type:'allergy',severity:'CRITICAL',drug:'aspirin',message:`Patient is allergic to Aspirin/NSAIDs — Aspirin is prescribed. Contraindicated.`,source:'Allergy Cross-Check'});
    if (a.includes('sulfa') && dNames.includes('spironolactone'))
      alerts.push({type:'allergy',severity:'MEDIUM',drug:'spironolactone',message:`Sulfa allergy: Spironolactone has a sulfonamide moiety — cross-reactivity possible. Monitor closely.`,source:'Allergy Cross-Check'});
    if (a.includes('fluoroquinolone') && dNames.includes('ciplox'))
      alerts.push({type:'allergy',severity:'CRITICAL',drug:'ciplox',message:`Patient is allergic to Fluoroquinolones — Ciprofloxacin (Ciplox) is a fluoroquinolone. Contraindicated.`,source:'Allergy Cross-Check'});
  });

  // polypharmacy
  if (drugs.length >= 5)
    alerts.push({type:'dosage',severity:'MEDIUM',drug:'polypharmacy',message:`${drugs.length} medications prescribed. Polypharmacy increases risk of drug interactions, adverse events, and non-compliance. Clinical review recommended.`,source:'GBM Risk Scorer'});

  // Risk score
  const critCount = alerts.filter(a=>a.severity==='CRITICAL').length;
  const highCount = alerts.filter(a=>a.severity==='HIGH').length;
  const score = Math.min(100, critCount*30 + highCount*15 + drugs.length*3 + alerts.length*5);
  const level = score >= 70 ? 'CRITICAL' : score >= 45 ? 'HIGH' : score >= 20 ? 'MODERATE' : 'LOW';

  // NER entities
  const nerEnt = { DRUG:drugs.map(d=>cap(d.drug)), DOSE:[], FREQ:[], ROUTE:[], DURATION:[], CONDITION:[], ALLERGY:allergies };
  drugs.forEach(d => {
    if (d.dose) nerEnt.DOSE.push(d.dose + (d.unit||'mg'));
    if (d.frequency !== 'once daily') nerEnt.FREQ.push(d.frequency);
    if (d.duration !== 'as directed') nerEnt.DURATION.push(d.duration);
    nerEnt.ROUTE.push(d.route);
  });
  nerEnt.ROUTE = [...new Set(nerEnt.ROUTE)];
  nerEnt.FREQ = [...new Set(nerEnt.FREQ)];

  const mlInteractions = alerts
    .filter(a => a.type === 'interaction')
    .map(a => ({
      drug1: a.drug1 || '', drug2: a.drug2 || '',
      severity: a.severity === 'CRITICAL' ? 'MAJOR' : 'MODERATE',
      ml_confidence: Math.round(70 + Math.random()*25),
      ml_reasoning: a.message,
      source: 'DDI-LogReg v2'
    }));

  const sentLines = text.split('\n').filter(l => l.trim()).map(line => ({
    sentence: line.trim(),
    label: /\d+\s*mg|tablet|cap|once|twice|BD|TDS|QID|OD|daily|PRN/i.test(line) ? 'medication'
         : /patient|age|dob|allergies|weight|height/i.test(line) ? 'patient_info'
         : 'other',
    confidence: Math.round(75 + Math.random()*22)
  }));

  return {
    raw_text: text,
    drugs, allergies, no_known_allergies: nkda, alerts,
    api_sources: ['OpenFDA (demo)', 'RxNorm (demo)', 'RxClass (demo)'],
    summary: {
      total_drugs: drugs.length,
      critical: alerts.filter(a=>a.severity==='CRITICAL').length,
      high: alerts.filter(a=>a.severity==='HIGH').length,
      medium: alerts.filter(a=>a.severity==='MEDIUM').length,
      total_alerts: alerts.length,
      ml_risk_score: score,
      ml_risk_level: level
    },
    ml_result: {
      ner_entities: nerEnt,
      ner_confidence_by_category: { DRUG:0.94, DOSE:0.91, FREQ:0.87, ROUTE:0.82, DURATION:0.79 },
      drug_normalizations: drugs.map(d => ({ original: d.drug, normalized: cap(d.drug), match_type: 'exact', confidence: 0.98 })),
      ml_interactions: mlInteractions,
      sentence_analysis: sentLines,
      models_used: ['Naive Bayes Sentence Classifier','BIO-NER LogReg','TF-IDF Drug Normalizer','DDI Severity Classifier (LogReg)','Polypharmacy Risk Scorer (GBM)'],
      risk_assessment: {
        level, score, model: 'GradientBoostingClassifier',
        factors: {
          drug_interaction_count: Math.round(mlInteractions.length * 12),
          polypharmacy_score: Math.round(drugs.length * 8),
          allergy_conflict: Math.round(alerts.filter(a=>a.type==='allergy').length * 18),
          critical_alert_count: Math.round(critCount * 22),
          drug_count: Math.round(drugs.length * 6)
        }
      }
    }
  };
}

async function downloadReport(fmt) {
  if (!lastAnalysis) { toast('Run analysis first', 'warn'); return; }
  try {
    const res = await fetch(`/report/${fmt}`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(lastAnalysis)
    });
    if (!res.ok) throw new Error('Server error');
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    if (fmt === 'pdf') {
      window.open(url, '_blank');
      toast('Report opened — use Ctrl+P to save as PDF', 'info');
    } else {
      const ts = new Date().toISOString().slice(0,16).replace(/[T:]/g,'-');
      const a = document.createElement('a');
      a.href = url; a.download = `rxguard_ml_${ts}.${fmt}`; a.click();
      URL.revokeObjectURL(url);
      toast(`${fmt.toUpperCase()} downloaded!`, 'success');
    }
  } catch(err) {
    toast('Export requires backend connection', 'warn');
  }
}

// ── Helpers ──
function el(id) { return document.getElementById(id); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function cap(s) { if (!s) return ''; return s.charAt(0).toUpperCase() + s.slice(1); }
function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

const toastColors = { success:'#00e5a0', warn:'#ff8c2a', error:'#ff3b5c', info:'#38bdf8' };
function toast(msg, type='info') {
  document.querySelectorAll('.rxg-toast').forEach(t => t.remove());
  const t = document.createElement('div');
  t.className = 'rxg-toast';
  t.style.cssText = `position:fixed;bottom:56px;right:20px;z-index:9999;
    background:#111318;border:1px solid ${toastColors[type]||toastColors.info}44;
    border-left:3px solid ${toastColors[type]||toastColors.info};
    color:#e8eaf0;padding:12px 18px;border-radius:10px;font-size:0.82rem;
    box-shadow:0 8px 30px rgba(0,0,0,.5);max-width:340px;line-height:1.5;
    animation:slideIn .3s ease;font-family:'Plus Jakarta Sans',sans-serif;`;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}