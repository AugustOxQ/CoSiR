'use strict';
const $ = id => document.getElementById(id);
const COLORS = {'first token': '#999', 'instruction': '#6a9fd8', 'image': '#e2a23a', 'caption': '#6bbf6b', 'other template': '#c9a0d8'};
const MODS = [['img', 'Image'], ['txt', 'Text'], ['joint', 'Image + text']];
const st = {ks: [], k: null, view: 'all', mod: 'joint', cap: 0, layerKind: 'layers 16-27', layer: 27, word: null,
            meta: null, pinfo: null, img: null, maps: null, req: 0};

// jet-like colormap, t in 0..1 -> [r,g,b]
function jet(t) {
  t = Math.max(0, Math.min(1, t));
  const f = x => Math.max(0, Math.min(1, x));
  return [255 * f(1.5 - Math.abs(4 * t - 3)), 255 * f(1.5 - Math.abs(4 * t - 2)), 255 * f(1.5 - Math.abs(4 * t - 1))];
}
function layerParam() { return st.layerKind === 'single' ? String(st.layer) : st.layerKind; }
async function jget(url) {
  const r = await fetch(url); const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}
function loadImg(url) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = url; });
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c])); }
function el(tag, cls, text) {
  const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e;
}
function cap1(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function viewLabel(v) { return v.replace(/_/g, ' '); }
function hue(s) { let h = 0; for (const c of String(s)) h = (h * 31 + c.charCodeAt(0)) % 360; return h; }

// segmented control: opts = [[value, label], ...]; onpick(value)
function makeSeg(box, opts, onpick) {
  box.innerHTML = '';
  for (const [v, label] of opts) {
    const b = el('button', '', label); b.type = 'button'; b.dataset.v = v; b.setAttribute('role', 'radio');
    b.onclick = () => onpick(v);
    box.appendChild(b);
  }
}
function markSeg(box, value) {
  for (const b of box.children) { const on = b.dataset.v === value; b.classList.toggle('on', on); b.setAttribute('aria-checked', on); }
}

async function init() {
  st.meta = await jget('/api/meta');
  st.ks = st.meta.ks;
  if (!st.ks.length) { $('status').textContent = 'no complete painting files found'; return; }

  // static legends
  const stops = []; for (let i = 0; i <= 10; i++) { const c = jet(i / 10); stops.push(`rgb(${c.map(Math.round)}) ${i * 10}%`); }
  $('jetbar').style.background = `linear-gradient(90deg, ${stops.join(',')})`;
  const sl = $('splitlegend');
  for (const [n, c] of Object.entries(COLORS)) sl.innerHTML += `<span><i style="background:${c}"></i>${esc(n)}</span>`;

  // controls
  makeSeg($('viewseg'), [['all', 'All four'], ...Object.keys(st.meta.views).map(v => [v, viewLabel(v)])],
          v => { st.view = v; markSeg($('viewseg'), v); render(); });
  for (const b of $('viewseg').children) if (st.meta.views[b.dataset.v]) b.title = st.meta.views[b.dataset.v];
  markSeg($('viewseg'), st.view);
  makeSeg($('modseg'), MODS, v => { if (v === st.mod) return; st.mod = v; st.word = null; updateVis(); refreshMaps(); });
  makeSeg($('layerseg'), [...st.meta.bands.map(b => [b, cap1(b)]), ['single', 'Single']],
          v => { if (v === st.layerKind) return; st.layerKind = v; updateVis(); refreshMaps(); });

  $('kslider').max = st.ks.length - 1;
  $('kcount').textContent = 'of ' + st.ks.length;
  $('kcount').title = 'number of paintings; the number box is the painting index k';
  $('kslider').oninput = e => setK(st.ks[+e.target.value]);
  $('knum').onchange = e => { const k = +e.target.value; setK(nearest(k)); };
  $('prev').onclick = () => step(-1); $('next').onclick = () => step(1);
  $('rand').onclick = () => setK(st.ks[Math.floor(Math.random() * st.ks.length)]);
  document.addEventListener('keydown', e => {
    const t = e.target;
    if (t.tagName === 'INPUT' && (t.type === 'number' || t.type === 'range' || t.type === 'text')) return;
    if (e.key === 'ArrowLeft') step(-1);
    else if (e.key === 'ArrowRight') step(1);
    else if (e.key === 'Escape' && st.word !== null) selectWord(null);
  });
  $('layerslider').oninput = e => { st.layer = +e.target.value; $('layerval').textContent = st.layer; };
  $('layerslider').onchange = () => refreshMaps();
  $('opacity').oninput = () => drawAll();
  $('smooth').onchange = () => drawAll();
  for (const id of ['rank', 'blur']) {
    try { $(id).checked = localStorage.getItem('att_' + id) === '1'; } catch (e) {}
    $(id).onchange = () => { try { localStorage.setItem('att_' + id, $(id).checked ? '1' : '0'); } catch (e) {} legend(); drawAll(); };
  }
  legend();
  $('sidetoggle').onclick = () => setSide($('side').classList.contains('hidden'));
  let open = true; try { open = localStorage.getItem('att_side') !== '0'; } catch (e) {}
  setSide(open, true);
  updateVis();
  setK(st.ks[0]);
}
function setSide(open, noDraw) {
  $('side').classList.toggle('hidden', !open);
  $('sidetoggle').setAttribute('aria-pressed', open);
  try { localStorage.setItem('att_side', open ? '1' : '0'); } catch (e) {}
  if (!noDraw) drawAll();
}
function nearest(k) { let b = st.ks[0]; for (const x of st.ks) if (Math.abs(x - k) < Math.abs(b - k)) b = x; return b; }
function step(d) {
  const i = st.ks.indexOf(st.k); const j = Math.max(0, Math.min(st.ks.length - 1, i + d)); setK(st.ks[j]);
}
function updateVis() {
  markSeg($('modseg'), st.mod);
  markSeg($('layerseg'), st.layerKind);
  $('single_wrap').style.display = st.layerKind === 'single' ? '' : 'none';
  $('caplist').classList.toggle('off', st.mod === 'img');
  $('capnote').textContent = st.mod === 'img' ? '(not used for image only)' : '(click to choose)';
}

async function setK(k) {
  st.k = k; st.word = null; st.cap = 0;
  const i = st.ks.indexOf(k);
  $('kslider').value = i; $('knum').value = k;
  const my = ++st.req;
  $('status').textContent = 'loading...';
  try {
    const [pi, im] = await Promise.all([jget('/api/painting?k=' + k), loadImg('/api/image?k=' + k)]);
    if (my !== st.req) return;
    st.pinfo = pi; st.img = im;
    showPinfo();
    await refreshMaps(true);
  } catch (e) { $('status').textContent = 'error: ' + e.message; }
}
function showPinfo() {
  const p = st.pinfo;
  $('pinfo').innerHTML = `<span class="k">k</span><span class="v"><b>${p.k}</b></span>` +
    `<span class="k">id</span><span class="v">${esc(p.painting_id)}</span>` +
    `<span class="k">style</span><span class="v"><b>${esc(p.style)}</b></span>` +
    `<span class="k">genre</span><span class="v"><b>${esc(p.genre)}</b></span>`;
  const box = $('caplist'); box.innerHTML = '';
  p.captions.forEach((c, i) => {
    const b = el('button', 'capitem' + (i === st.cap ? ' on' : '')); b.type = 'button'; b.dataset.i = i;
    b.innerHTML = `<span class="idx">${i}</span><span class="chip" style="--h:${hue(c.emotion)}">${esc(c.emotion)}</span>${esc(c.text)}`;
    b.onclick = () => pickCap(i);
    box.appendChild(b);
  });
}
function pickCap(i) {
  st.cap = i; st.word = null;
  for (const b of $('caplist').children) b.classList.toggle('on', +b.dataset.i === i);
  if (st.mod !== 'img') refreshMaps();
}

async function refreshMaps() {
  if (st.k === null) return;
  const my = ++st.req;
  $('status').textContent = 'loading maps...';
  try {
    const m = await jget(`/api/maps?k=${st.k}&mod=${st.mod}&cap=${st.cap}&layer=${encodeURIComponent(layerParam())}`);
    if (my !== st.req) return;
    st.maps = m; $('status').textContent = '';
    render();
  } catch (e) { $('status').textContent = 'error: ' + e.message; }
}

function render() {
  const P = $('panels'); P.innerHTML = '';
  if (!st.maps) return;
  const views = (st.view === 'all' ? Object.keys(st.meta.views) : [st.view]).filter(v => st.maps.views[v]);
  P.style.setProperty('--n', views.length);
  for (const v of views) {
    const d = st.maps.views[v];
    const solo = views.length === 1;
    const div = el('section', 'card panel' + (solo ? ' solo' : ''));
    const head = el('div', 'phead');
    head.appendChild(el('div', 'pname', viewLabel(v)));
    const ins = el('div', 'pinstr', st.meta.views[v]); ins.title = st.meta.views[v]; head.appendChild(ins);
    div.appendChild(head);
    d._label = el('div', 'plabel'); div.appendChild(d._label);
    const box = el('div', 'imgbox'); d._cv = el('canvas'); box.appendChild(d._cv); d._box = box; div.appendChild(box);
    div.appendChild(words(d));
    div.appendChild(foot(d));
    P.appendChild(div);
  }
  drawAll();
}
function foot(d) {
  const sh = d.shares;
  const wrap = el('div', 'pfoot');
  const b = el('div', 'bar');
  for (const [n, c] of Object.entries(COLORS)) {
    const x = sh[n] || 0;
    const s = document.createElement('div'); s.style.cssText = `width:${x * 100}%;background:${c}`; s.title = `${n} ${(x * 100).toFixed(1)}%`;
    b.appendChild(s);
  }
  b.title = Object.entries(COLORS).map(([n]) => `${n} ${((sh[n] || 0) * 100).toFixed(1)}%`).join('\n');
  wrap.appendChild(b);
  const t = el('div', 'splittxt', `image ${((sh.image || 0) * 100).toFixed(1)}% · caption ${((sh.caption || 0) * 100).toFixed(1)}%`);
  t.title = b.title; wrap.appendChild(t);
  return wrap;
}
function words(d) {
  const box = el('div', 'pcap');
  if (!d.caption_tokens) { box.classList.add('none'); box.textContent = 'Image only: no caption in the input.'; d._spans = []; return box; }
  const inner = el('div', 'words' + (st.mod === 'joint' ? ' joint' : ''));
  const mx = Math.max(...d.caption_tokens.map(t => t.weight), 1e-12);
  d._spans = d.caption_tokens.map((t, i) => {
    const s = el('span', 'w' + (st.word === i ? ' sel' : ''), t.text);
    s.style.background = `rgba(255,140,0,${(t.weight / mx).toFixed(3)})`;
    s.title = 'weight ' + t.weight.toPrecision(3) + (st.mod === 'joint' ? '; click for this word\'s map' : '');
    if (st.mod === 'joint') s.onclick = () => selectWord(st.word === i ? null : i);
    inner.appendChild(s);
    return s;
  });
  box.appendChild(inner);
  return box;
}
function selectWord(i) {
  st.word = i;
  if (!st.maps) return;
  for (const d of Object.values(st.maps.views)) if (d._spans) d._spans.forEach((s, j) => s.classList.toggle('sel', j === i));
  drawAll();
}

// size the canvas to its box: aspect kept, crisp on high-dpi screens
function fit(d) {
  const w = d._box.clientWidth, h = d._box.clientHeight;
  if (w < 2 || h < 2) return false;
  const k = Math.min(w / st.img.width, h / st.img.height);
  const cw = Math.max(1, Math.floor(st.img.width * k)), ch = Math.max(1, Math.floor(st.img.height * k));
  const dpr = window.devicePixelRatio || 1;
  d._cv.style.width = cw + 'px'; d._cv.style.height = ch + 'px';
  d._cv.width = Math.round(cw * dpr); d._cv.height = Math.round(ch * dpr);
  return true;
}
function legend() {
  const rank = $('rank').checked;
  $('jetlo').textContent = rank ? 'lowest' : '0';
  $('jethi').textContent = rank ? 'highest token of each map' : 'max of each map';
  $('jetnote').textContent = rank
    ? 'Colours show each token\'s rank within its own map (percentile), so equal steps of colour are equal numbers of tokens.'
    : 'Each map is scaled to its own maximum, so colours compare positions inside one map, not between maps.';
}
// Gaussian blur over the token grid, sigma 1 token, radius 2; weights renormalised at the edges
function blurMap(map) {
  const R = map.length, Cc = map[0].length, w = [-2, -1, 0, 1, 2].map(x => Math.exp(-x * x / 2));
  const pass = (src, horiz) => src.map((row, i) => row.map((_, j) => {
    let s = 0, n = 0;
    for (let t = -2; t <= 2; t++) {
      const a = horiz ? i : i + t, b = horiz ? j + t : j;
      if (a < 0 || a >= R || b < 0 || b >= Cc) continue;
      s += w[t + 2] * src[a][b]; n += w[t + 2];
    }
    return s / n;
  }));
  return pass(pass(map, true), false);
}
// percentile rank within the map, 0 (lowest) to 1 (highest); ties share their average rank
function rankMap(map) {
  const flat = map.flat(), n = flat.length, idx = flat.map((_, i) => i).sort((a, b) => flat[a] - flat[b]);
  const r = new Array(n);
  for (let i = 0; i < n;) {
    let j = i; while (j + 1 < n && flat[idx[j + 1]] === flat[idx[i]]) j++;
    for (let t = i; t <= j; t++) r[idx[t]] = (i + j) / 2 / Math.max(n - 1, 1);
    i = j + 1;
  }
  const C = map[0].length;
  return map.map((row, i) => row.map((_, j) => r[i * C + j]));
}
function overlay(d, map) {
  const mode = ($('rank').checked ? 'r' : '') + ($('blur').checked ? 'b' : '');
  if (d._ocFor === map && d._ocMode === mode) return d._oc;
  const R = map.length, Cc = map[0].length;
  let mx = 0; for (const r of map) for (const x of r) if (x > mx) mx = x;
  let shown = $('blur').checked ? blurMap(map) : map;
  let smx = 1;
  if ($('rank').checked) shown = rankMap(shown);
  else { smx = 0; for (const r of shown) for (const x of r) if (x > smx) smx = x; }
  const oc = document.createElement('canvas'); oc.width = Cc; oc.height = R;
  const og = oc.getContext('2d'), id = og.createImageData(Cc, R);
  for (let i = 0; i < R; i++) for (let j = 0; j < Cc; j++) {
    const c = jet(shown[i][j] / (smx || 1)), o = 4 * (i * Cc + j);
    id.data[o] = c[0]; id.data[o + 1] = c[1]; id.data[o + 2] = c[2]; id.data[o + 3] = 255;
  }
  og.putImageData(id, 0, 0);
  d._oc = oc; d._ocFor = map; d._ocMode = mode; d._mx = mx; d._R = R; d._C = Cc;
  return oc;
}

function drawAll() {
  if (!st.maps || !st.img) return;
  for (const v of Object.keys(st.maps.views)) {
    const d = st.maps.views[v]; if (!d._cv || !d._cv.isConnected) continue;
    if (!fit(d)) continue;
    const cv = d._cv, g = cv.getContext('2d');
    g.clearRect(0, 0, cv.width, cv.height);
    g.drawImage(st.img, 0, 0, cv.width, cv.height);
    let map = d.image_map, label = 'pooled last token', note = '';
    if (st.mod === 'joint' && st.word !== null && d.word_maps) {
      map = d.word_maps[st.word]; label = 'word ' + JSON.stringify(d.caption_tokens[st.word].text);
      if (st.layerKind === 'single') note = ` (word maps use ${d.word_maps_band})`;
    }
    if (!map) {
      d._label.innerHTML = '<b>Text only</b> · no image map'; d._label.title = '';
      g.fillStyle = 'rgba(255,255,255,0.7)'; g.fillRect(0, 0, cv.width, cv.height); continue;
    }
    const oc = overlay(d, map);
    g.globalAlpha = +$('opacity').value;
    g.imageSmoothingEnabled = $('smooth').checked; g.imageSmoothingQuality = 'high';
    g.drawImage(oc, 0, 0, cv.width, cv.height);
    g.globalAlpha = 1;
    d._label.innerHTML = `Map of <b>${esc(label)}</b>` + (note ? `<span class="note">${esc(note)}</span>` : '');
    d._label.title = ($('rank').checked ? 'colour = rank within the map; ' : '') + ($('blur').checked ? 'blurred (sigma 1 token); ' : '')
      + `raw max ${d._mx.toPrecision(4)} (uniform = ${(1 / (d._R * d._C)).toPrecision(3)})`;
    cv.title = d._label.title;
  }
}
let rt = null;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(render, 150); });
init();
