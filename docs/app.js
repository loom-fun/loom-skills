'use strict';
const MET = [['decode','decode'],['decode_2k','decode 2k'],['decode_32k','decode 32k'],
  ['prefill_1k','prefill 1k'],['prefill_2k','prefill 2k'],['prefill_8k','prefill 8k'],
  ['prefill_16k','prefill 16k'],['prefill_32k','prefill 32k'],['streams4','4 streams']];
const FEAT = [['api','API'],['tool_calling','tool calling'],['thinking','thinking'],['reasoning_effort','reasoning effort'],
  ['speculative_decoding','spec decoding'],['image_input','image input'],['audio','audio'],['sampling','sampling'],
  ['context','context'],['concurrency','concurrency'],['max_output_tokens','max output']];
const GATE = [['kernel_tests','kernel tests'],['golden','golden'],['compare','compare'],['vision','vision'],
  ['consistency','consistency'],['spec','spec'],['agent','agent'],['agent_think_off','agent think-off'],
  ['sampling','sampling'],['needle','needle'],['needle_full','needle full'],['decision','decision']];
const TIMED = MET.map((m) => m[0]);
const NORMNAME = {rocm: 'llama.cpp ROCm', vulkan: 'llama.cpp Vulkan'};
const COL = {loom: '#a84f0e', rocm: '#8f887b', vulkan: '#4a7fb0'};
const ST = {norm: 'rocm', q: '', sort: 'gain', dir: 'desc'};
const HF = 'https://huggingface.co';

/* ---- info.json -> row (port of compile.py + card_lib) ---- */
const isDevlab = (info, s) => {
  const src = (s.source && typeof s.source === 'object') ? s.source : (info.sources || {})[s.source];
  return !!src && src.kind === 'devlab';
};
const devlabMetrics = (info) => {
  const out = {};
  ((info.performance || {}).sessions || []).forEach((s) => {
    if (!isDevlab(info, s)) return;
    for (const [k, m] of Object.entries(s.metrics || {})) if (TIMED.includes(k)) out[k] = m;
  });
  return out;
};
const overContext = (info, native) => {
  const sessions = (info.performance || {}).sessions || [];
  const measured = new Set();
  sessions.forEach((s) => { for (const [k, m] of Object.entries(s.metrics || {})) if (m.loom != null) measured.add(k); });
  const out = new Set();
  sessions.forEach((s) => { for (const [k, n] of Object.entries(s.over_context || {})) if (native && n && n > native && !measured.has(k)) out.add(k); });
  return out;
};
const drafting = (f) => { const sd = f.speculative_decoding; return !!sd && sd.status === 'yes' && !!(sd.value && sd.value.method); };
const gateApplies = (name, kind, f) => {
  if (name === 'kernel_tests' || name === 'consistency') return true;
  if (kind === 'decision') return name === 'decision';
  if (name === 'decision') return false;
  const st = (k) => (f[k] || {}).status;
  if (name === 'vision') return st('image_input') === 'yes';
  if (name === 'spec') return drafting(f);
  if (name === 'agent') return st('tool_calling') === 'yes';
  if (name === 'agent_think_off') {
    const th = f.thinking || {};
    return st('tool_calling') === 'yes' && th.status === 'yes' && !!th.value && th.value.mode === 'toggle';
  }
  return true;
};
const metricApplies = (key, kind, f, info) => {
  if (kind === 'decision' && ['decode', 'decode_2k', 'decode_32k', 'streams4', 'spec_acceptance'].includes(key)) return false;
  const native = ((f.context || {}).value || {}).native;
  return !overContext(info, native).has(key);
};
function rowFromInfo(repo, info) {
  const meta = window.REPOS[repo] || {};
  const gates = info.gates || {};
  const stored = (info.card || {}).features || {};
  const kind = (gates.decision && !gates.golden) ? 'decision' : 'chat';
  const full = {};
  FEAT.forEach(([k]) => { full[k] = stored[k] ? {status: stored[k].status, value: stored[k].value} : {status: 'unknown', value: null}; });
  const r = {repo, model: meta.model, arch: meta.arch, params_b: meta.params_b == null ? null : meta.params_b};
  const dl = devlabMetrics(info);
  const vk = ((((info.performance || {}).baselines || {}).vulkan || {}).session || {}).metrics || {};
  TIMED.forEach((k) => {
    const ok = metricApplies(k, kind, full, info);
    const m = ok ? (dl[k] || {}) : {};
    r[k + '_loom'] = m.loom == null ? null : m.loom;
    r[k + '_rocm'] = m.baseline == null ? null : m.baseline;
    r[k + '_vulkan'] = ok && vk[k] && vk[k].baseline != null ? vk[k].baseline : null;
  });
  FEAT.forEach(([k]) => { r['f_' + k] = full[k].status; });
  GATE.forEach(([g]) => {
    const ev = gates[g];
    const applies = gateApplies(g, kind, full);
    r['g_' + g] = (applies || ev) ? (ev ? ev.status : 'missing') : 'n/a';
  });
  return r;
}

/* ---- ratios / summary (dash.calc port) ---- */
const med = (a) => { a = a.filter((x) => x != null).sort((x, y) => x - y); const n = a.length;
  return n ? (n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2) : null; };
function computeRatios(models) {
  const keys = TIMED;
  return models.map((r) => {
    const o = {repo: r.repo};
    for (const b of ['rocm', 'vulkan']) {
      const rs = keys.map((k) => { const l = Number(r[k + '_loom']), x = Number(r[k + '_' + b]);
        return r[k + '_loom'] != null && r[k + '_' + b] != null && x > 0 ? l / x : null; });
      keys.forEach((k, i) => { o[k + '_' + b] = rs[i]; o[k + '_' + b + '_pct'] = rs[i] == null ? null : Math.round((rs[i] - 1) * 100); });
      const m = med(rs); o['median_' + b] = m; o['median_' + b + '_pct'] = m == null ? null : Math.round((m - 1) * 100);
    }
    for (const [b, oth] of [['rocm', 'vulkan'], ['vulkan', 'rocm']]) {
      keys.forEach((k) => { const x = Number(r[k + '_' + b]), y = Number(r[k + '_' + oth]);
        const v = r[k + '_' + b] != null && r[k + '_' + oth] != null && x > 0 ? y / x : null;
        o[k + '_' + oth + '_vs_' + b] = v; o[k + '_' + oth + '_vs_' + b + '_pct'] = v == null ? null : Math.round((v - 1) * 100); });
    }
    return o;
  });
}

/* ---- rendering ---- */
let MODELS = [], RATIOS = [], ERRORS = [], LOADED = false;
const $ = (id) => document.getElementById(id);
const fmtP = (p) => (p >= 0 ? '+' : '-') + Math.abs(p) + '%';
const RN = () => ST.norm === 'abs' ? 'rocm' : ST.norm;
const mk = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const tip = mk('div', 'tip'); tip.hidden = true;

function drawBrag() {
  $('n_models').textContent = LOADED ? MODELS.length : '—';
  $('n_archs').textContent = LOADED ? new Set(MODELS.map((m) => m.arch)).size : '—';
  $('normname').textContent = NORMNAME[RN()];
  const v = LOADED ? Math.round(10 * med(RATIOS.map((r) => r['prefill_32k_' + RN()]))) / 10 : null;
  $('med32').textContent = v == null || isNaN(v) ? '—' : v.toFixed(1) + '×';
}

function drawCards() {
  const el = $('grid');
  if (!LOADED) { el.replaceChildren(document.createTextNode('Loading…'), tip); return; }
  const byRepo = new Map(RATIOS.map((r) => [r.repo, r]));
  const nk = RN(), absm = ST.norm === 'abs', oth = nk === 'rocm' ? 'vulkan' : 'rocm';
  const all = [];
  RATIOS.forEach((r) => MET.forEach(([k]) => { for (const f of [k + '_' + nk, k + '_' + oth + '_vs_' + nk]) if (r[f] != null) all.push(r[f]); }));
  const lo = 0, hi = Math.max(2, all.length ? Math.max(...all) : 2);
  const X = (r) => (r - lo) / (hi - lo) * 100;
  const q = ST.q.trim().toLowerCase();
  const rows = MODELS.filter((m) => !q || m.repo.toLowerCase().includes(q) || String(m.model).toLowerCase().includes(q) || String(m.arch).toLowerCase().includes(q));
  const key = (m) => ST.sort === 'size' ? m.params_b : byRepo.get(m.repo)['median_' + nk];
  const sign = ST.dir === 'asc' ? 1 : -1;
  rows.sort((a, b) => { const x = key(a), y = key(b);
    if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1; return sign * (x - y); });
  const showTip = (ev, text) => {
    tip.textContent = text; tip.hidden = false;
    const r = el.getBoundingClientRect();
    let x = ev.clientX - r.left + 12; const y = ev.clientY - r.top + 14;
    const w = tip.offsetWidth; if (x + w > el.clientWidth) x = Math.max(0, el.clientWidth - w);
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  };
  const hide = () => { tip.hidden = true; };
  const cards = rows.map((m) => {
    const rt = byRepo.get(m.repo);
    const card = mk('div', 'card');
    const h = mk('h3'); const a = mk('a', null, m.repo.replace(/^loom-fun\//, ''));
    a.href = HF + '/' + m.repo; a.target = '_blank'; a.rel = 'noopener'; h.append(a); card.append(h);
    const sub = mk('div', 'sub');
    const md = rt['median_' + nk + '_pct'];
    sub.append((m.params_b != null ? m.params_b + 'B' : 'n/a'), ' · ', m.arch, ' · ', absm ? 'median vs ROCm ' : 'median ', md == null ? 'n/a' : fmtP(md));
    card.append(sub);
    MET.forEach(([k, label]) => {
      const r = rt[k + '_' + nk];
      if (absm) {
        const vals = [['loom', 'Loom'], ['rocm', NORMNAME.rocm], ['vulkan', NORMNAME.vulkan]].map(([e, n]) => [e, n, COL[e], m[k + '_' + e]]);
        const mx = Math.max(...vals.map((v) => v[3] ?? 0)) || 1;
        const row = mk('div', 'a'); row.append(mk('span', 'lab', label));
        const bars = mk('div', 'bars');
        vals.forEach(([e, n, c, v]) => {
          const ar = mk('div', 'ar'), at = mk('div', 'at'), ab = mk('div', 'ab');
          ab.style.background = c; ab.style.width = v == null ? '0%' : Math.max(v / mx * 100, 0.5) + '%'; at.append(ab);
          ar.append(at, mk('span', 'av', v == null ? 'n/a' : String(v)));
          const text = label + ': ' + n + ' ' + (v == null ? 'n/a' : v + ' tok/s');
          ar.addEventListener('mousemove', (ev) => showTip(ev, text)); ar.addEventListener('mouseleave', hide);
          bars.append(ar);
        });
        row.append(bars); card.append(row); return;
      }
      const row = mk('div', 'm'); row.append(mk('span', 'lab', label));
      const ro = rt[k + '_' + oth + '_vs_' + nk];
      const tr = mk('div', 'track');
      [[r, COL.loom, '0', '55%'], [ro, COL[oth], '55%', '0']].forEach(([v, c, t, b]) => {
        if (v == null) return;
        const bar = mk('div', 'bar'); bar.style.top = t; bar.style.bottom = b; bar.style.left = '0';
        bar.style.width = Math.max(X(v), 0.6) + '%'; bar.style.background = c; tr.append(bar);
      });
      const bl = mk('div', 'base'); bl.style.left = X(1) + '%'; tr.append(bl);
      row.append(tr);
      const pcts = mk('span', 'pct');
      [[r, k + '_' + nk + '_pct', COL.loom], [ro, k + '_' + oth + '_vs_' + nk + '_pct', COL[oth]]].forEach(([v, f, c]) => {
        const p = mk('span', null, v == null ? 'n/a' : fmtP(rt[f])); p.style.color = c; p.style.display = 'block'; pcts.append(p);
      });
      row.append(pcts);
      const fx = (v) => v == null ? 'n/a' : v.toFixed(2) + '×';
      const text = label + ' vs ' + NORMNAME[nk] + ': Loom ' + fx(r) + ', ' + NORMNAME[oth] + ' ' + fx(ro);
      row.addEventListener('mousemove', (ev) => showTip(ev, text)); row.addEventListener('mouseleave', hide);
      card.append(row);
    });
    const chips = (list, pre, okv, kind) => {
      const w = mk('div', 'chips');
      list.forEach(([k, label]) => {
        const st = m[pre + k];
        const c = mk('span', 'chip');
        let sym, cls, word;
        if (st === okv) { sym = '✓'; cls = 'ok'; word = okv; }
        else if (st === 'no' || st === 'fail') { sym = kind === 'f' ? '❌' : '✗'; cls = 'no'; word = st; }
        else if (st === 'n/a') { sym = '–'; cls = 'na'; word = 'not applicable'; }
        else { sym = '?'; cls = 'unk'; word = st; }
        const s = mk('span', cls, sym); s.setAttribute('aria-label', label + ': ' + word);
        c.append(s, mk('span', null, label)); c.title = label + ': ' + word; w.append(c);
      });
      return w;
    };
    card.append(chips(FEAT, 'f_', 'yes', 'f'), chips(GATE, 'g_', 'pass', 'g'));
    return card;
  });
  const errs = ERRORS.filter((e) => !q || e.repo.toLowerCase().includes(q)).map((e) => {
    const card = mk('div', 'card'); const h = mk('h3'); const a = mk('a', null, e.repo.replace(/^loom-fun\//, ''));
    a.href = HF + '/' + e.repo; a.target = '_blank'; a.rel = 'noopener'; h.append(a);
    card.append(h, mk('div', 'err', 'Could not load info.json (' + e.msg + ')')); return card;
  });
  el.replaceChildren(...cards, ...errs, tip);
}
const redraw = () => { drawBrag(); drawCards(); };

/* ---- data ---- */
async function fetchJSON(url) {
  const r = await fetch(url, {redirect: 'follow'});
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}
async function load() {
  const mock = new URLSearchParams(location.search).get('data');
  const repos = Object.keys(window.REPOS);
  if (mock) {
    MODELS = await fetchJSON(mock);
  } else {
    try { await fetchJSON(HF + '/api/models?author=loom-fun&limit=1000'); } catch (e) { /* listing is advisory; the 50 repos are fixed */ }
    const res = await Promise.all(repos.map(async (repo) => {
      try { return rowFromInfo(repo, await fetchJSON(HF + '/' + repo + '/resolve/main/info.json')); }
      catch (e) { return {repo, error: e.message || String(e)}; }
    }));
    MODELS = res.filter((r) => !r.error);
    ERRORS = res.filter((r) => r.error).map((r) => ({repo: r.repo, msg: r.error}));
  }
  RATIOS = computeRatios(MODELS);
  LOADED = true;
  redraw();
}

document.querySelectorAll('[data-norm]').forEach((b) => b.addEventListener('click', () => {
  ST.norm = b.getAttribute('data-norm');
  document.querySelectorAll('[data-norm]').forEach((o) => { const on = o === b; o.classList.toggle('on', on); o.setAttribute('aria-pressed', String(on)); });
  redraw();
}));
$('q').addEventListener('input', (e) => { ST.q = e.target.value; drawCards(); });
$('sort').addEventListener('change', (e) => { ST.sort = e.target.value; drawCards(); });
document.querySelectorAll('[data-dir]').forEach((b) => b.addEventListener('click', () => {
  ST.dir = b.getAttribute('data-dir');
  document.querySelectorAll('[data-dir]').forEach((o) => { const on = o === b; o.classList.toggle('on', on); o.setAttribute('aria-pressed', String(on)); });
  drawCards();
}));
drawCards();
load().catch((e) => { $('grid').textContent = 'Could not load data (' + (e.message || e) + ')'; });
