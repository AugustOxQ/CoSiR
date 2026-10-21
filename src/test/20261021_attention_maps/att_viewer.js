'use strict';
const $ = id => document.getElementById(id);
const COLORS = {'first token': '#999', 'instruction': '#6a9fd8', 'image': '#e2a23a', 'caption': '#6bbf6b', 'other template': '#c9a0d8'};
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

async function init() {
  st.meta = await jget('/api/meta');
  st.ks = st.meta.ks;
  if (!st.ks.length) { $('status').textContent = 'no complete painting files found'; return; }
  const vb = $('viewbtns');
  for (const v of ['all', ...Object.keys(st.meta.views)]) {
    const b = document.createElement('button');
    b.textContent = v === 'all' ? 'all four side by side' : v; b.dataset.v = v;
    b.onclick = () => { st.view = v; st.word = st.word; markButtons(); render(); };
    vb.appendChild(b);
  }
  markButtons();
  const ls = $('layersel');
  for (const b of st.meta.bands) ls.add(new Option(b, b));
  ls.add(new Option('single layer', 'single'));
  ls.value = st.layerKind;
  $('kslider').max = st.ks.length - 1;
  $('kcount').textContent = st.ks.length + ' paintings';
  $('kslider').oninput = e => setK(st.ks[+e.target.value]);
  $('knum').onchange = e => { const k = +e.target.value; setK(nearest(k)); };
  $('prev').onclick = () => step(-1); $('next').onclick = () => step(1);
  $('rand').onclick = () => setK(st.ks[Math.floor(Math.random() * st.ks.length)]);
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' && e.target.type === 'number') return;
    if (e.key === 'ArrowLeft') step(-1); else if (e.key === 'ArrowRight') step(1);
  });
  for (const r of document.getElementsByName('mod')) r.onchange = () => { st.mod = r.value; if (r.checked) { st.mod = r.value; st.word = null; refreshMaps(); updateVis(); } };
  $('capsel').onchange = e => { st.cap = +e.target.value; st.word = null; refreshMaps(); };
  ls.onchange = e => { st.layerKind = e.target.value; updateVis(); refreshMaps(); };
  $('layerslider').oninput = e => { st.layer = +e.target.value; $('layerval').textContent = st.layer; };
  $('layerslider').onchange = () => refreshMaps();
  $('opacity').oninput = () => drawAll();
  $('smooth').onchange = () => drawAll();
  updateVis();
  setK(st.ks[0]);
}
function nearest(k) { let b = st.ks[0]; for (const x of st.ks) if (Math.abs(x - k) < Math.abs(b - k)) b = x; return b; }
function step(d) {
  const i = st.ks.indexOf(st.k); const j = Math.max(0, Math.min(st.ks.length - 1, i + d)); setK(st.ks[j]);
}
function markButtons() { for (const b of $('viewbtns').children) b.classList.toggle('on', b.dataset.v === st.view); }
function updateVis() {
  $('capsel_wrap').style.display = st.mod === 'img' ? 'none' : '';
  $('single_wrap').style.display = st.layerKind === 'single' ? '' : 'none';
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
  const sel = $('capsel'); sel.innerHTML = '';
  p.captions.forEach((c, i) => sel.add(new Option(`${i}: ${c.emotion}`, i)));
  sel.value = st.cap;
  let h = `<b>k=${p.k}</b> ${esc(p.painting_id)} &mdash; style: <b>${esc(p.style)}</b>, genre: <b>${esc(p.genre)}</b><ol start="0">`;
  p.captions.forEach(c => { h += `<li>[${esc(c.emotion)}] ${esc(c.text)}</li>`; });
  $('pinfo').innerHTML = h + '</ol>';
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c])); }

async function refreshMaps(force) {
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
  const views = st.view === 'all' ? Object.keys(st.meta.views) : [st.view];
  const w = st.view === 'all' ? Math.max(300, Math.min(480, Math.floor((Math.min(window.innerWidth, 1500) - 70) / 2))) : 520;
  for (const v of views) {
    const d = st.maps.views[v]; if (!d) continue;
    const div = document.createElement('div'); div.className = 'panel'; div.style.width = (w + 14) + 'px';
    div.innerHTML = `<div class="title">${v}</div><div class="instr">${esc(st.meta.views[v])}</div>`;
    const title = document.createElement('div'); title.className = 'stat'; div.appendChild(title);
    const cv = document.createElement('canvas'); cv.width = w; cv.height = Math.round(w * st.img.height / st.img.width);
    div.appendChild(cv);
    const stat = document.createElement('div'); stat.className = 'stat'; div.appendChild(stat);
    div.appendChild(bar(d.shares));
    if (d.caption_tokens) div.appendChild(words(d));
    P.appendChild(div);
    d._cv = cv; d._stat = stat; d._title = title; d._v = v;
  }
  drawAll();
}
function bar(sh) {
  const wrap = document.createElement('div');
  const b = document.createElement('div'); b.className = 'bar';
  const lg = document.createElement('div'); lg.className = 'legend';
  for (const [n, c] of Object.entries(COLORS)) {
    const x = sh[n] || 0;
    const s = document.createElement('div'); s.style.cssText = `width:${x * 100}%;background:${c}`; s.title = `${n} ${(x * 100).toFixed(1)}%`;
    b.appendChild(s);
    lg.innerHTML += `<span><i style="background:${c}"></i>${n} ${(x * 100).toFixed(1)}%</span>`;
  }
  wrap.appendChild(b); wrap.appendChild(lg); return wrap;
}
function words(d) {
  const box = document.createElement('div'); box.className = 'words' + (st.mod === 'joint' ? ' joint' : '');
  const mx = Math.max(...d.caption_tokens.map(t => t.weight), 1e-12);
  d.caption_tokens.forEach((t, i) => {
    const s = document.createElement('span'); s.className = 'w' + (st.word === i ? ' sel' : '');
    s.textContent = t.text; s.style.background = `rgba(255,140,0,${(t.weight / mx).toFixed(3)})`;
    s.title = 'weight ' + t.weight.toPrecision(3);
    if (st.mod === 'joint') s.onclick = () => { st.word = st.word === i ? null : i; render(); };
    box.appendChild(s);
  });
  return box;
}

function drawAll() {
  if (!st.maps) return;
  for (const v of Object.keys(st.maps.views)) {
    const d = st.maps.views[v]; if (!d._cv) continue;
    const cv = d._cv, g = cv.getContext('2d');
    g.clearRect(0, 0, cv.width, cv.height);
    g.drawImage(st.img, 0, 0, cv.width, cv.height);
    let map = d.image_map, label = 'pooled token', note = '';
    if (st.mod === 'joint' && st.word !== null && d.word_maps) {
      map = d.word_maps[st.word]; label = 'word: ' + JSON.stringify(d.caption_tokens[st.word].text);
      if (st.layerKind === 'single') note = ` (word maps use ${d.word_maps_band})`;
    }
    d._title.innerHTML = `<b>${esc(label)}</b>` + (note ? `<span class="note">${esc(note)}</span>` : '');
    if (!map) { d._stat.textContent = 'text only: no image map'; g.fillStyle = 'rgba(255,255,255,0.7)'; g.fillRect(0, 0, cv.width, cv.height); continue; }
    const R = map.length, Cc = map[0].length;
    let mx = 0; for (const r of map) for (const x of r) if (x > mx) mx = x;
    const oc = document.createElement('canvas'); oc.width = Cc; oc.height = R;
    const og = oc.getContext('2d'), id = og.createImageData(Cc, R);
    for (let i = 0; i < R; i++) for (let j = 0; j < Cc; j++) {
      const c = jet(map[i][j] / (mx || 1)), o = 4 * (i * Cc + j);
      id.data[o] = c[0]; id.data[o + 1] = c[1]; id.data[o + 2] = c[2]; id.data[o + 3] = 255;
    }
    og.putImageData(id, 0, 0);
    g.globalAlpha = +$('opacity').value;
    g.imageSmoothingEnabled = $('smooth').checked; g.imageSmoothingQuality = 'high';
    g.drawImage(oc, 0, 0, cv.width, cv.height);
    g.globalAlpha = 1;
    d._stat.textContent = `scale 0 to max ${mx.toPrecision(4)} (uniform = ${(1 / (R * Cc)).toPrecision(3)}); image share of raw attention ${(d.shares.image * 100).toFixed(1)}%`;
  }
}
window.addEventListener('resize', () => { clearTimeout(window._rt); window._rt = setTimeout(render, 200); });
init();
