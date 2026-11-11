// Headless check of the qf viewer page's script against a minimal DOM stub (no browser here); stub as in r20_viewer_check.js.
// Usage: node qf_viewer_check.js <viewer folder>
const fs = require('fs'), path = require('path');
const dir = process.argv[2];
const html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];

class Node_ { constructor(t) { this.nodeType = 3; this.textContent_ = t; } get textContent() { return this.textContent_; } }
class El {
  constructor(tag) { this.tagName = tag; this.children = []; this.attrs = {}; this.dataset = {}; this.style = {}; this.listeners = {}; this.className = ''; this.value = ''; this.checked = false; this.nodeType = 1;
    const self = this; this.classList = { toggle(c, on) { const s = new Set(self.className.split(/\s+/).filter(Boolean)); (on === undefined ? !s.has(c) : on) ? s.add(c) : s.delete(c); self.className = [...s].join(' '); }, contains(c) { return self.className.split(/\s+/).includes(c); } }; }
  get firstChild() { return this.children[0] || null; }
  get childNodes() { return this.children; }
  appendChild(c) { this.children.push(c); return c; }
  removeChild(c) { this.children.splice(this.children.indexOf(c), 1); return c; }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  fire(t, extra) { for (const f of this.listeners[t] || []) f(Object.assign({target: this}, extra)); }
  set textContent(t) { this.children = [new Node_(String(t))]; }
  get textContent() { return this.children.map(c => c.nodeType === 3 ? c.textContent_ : c.textContent).join(''); }
  all(pred, out = []) { for (const c of this.children) if (c.nodeType === 1) { if (pred(c)) out.push(c); c.all(pred, out); } return out; }
}
const reg = {};
for (const m of html.split('<script>')[0].matchAll(/<(\w+)([^>]*?)\sid="(\w+)"([^>]*)>/g)) {
  const e = new El(m[1]); const cls = (m[2] + m[4]).match(/class="([^"]*)"/); e.className = cls ? cls[1] : ''; if (/checked/.test(m[2] + m[4])) e.checked = true; reg[m[3]] = e;
}
const body = new El('body');
const document = { getElementById: id => reg[id], createElement: t => new El(t), createTextNode: t => new Node_(t), body, addEventListener() {} };
const fetch = async p => { const f = path.join(dir, p); if (!fs.existsSync(f)) return {ok: false, status: 404}; const txt = fs.readFileSync(f, 'utf8'); return {ok: true, status: 200, json: async () => JSON.parse(txt)}; };
const window = {innerWidth: 1600};
const errors = [];
(async () => {
  new Function('document', 'fetch', 'window', 'setTimeout', script)(document, fetch, window, (f) => setImmediate(f));
  const wait = async () => { for (let i = 0; i < 25; i++) await new Promise(r => setTimeout(r, 30)); };
  await wait();
  if (body.children.length) throw new Error('page error: ' + body.textContent);
  const D = JSON.parse(fs.readFileSync(path.join(dir, 'data/candidates.json')));
  const imgs = id => reg[id].all(e => e.tagName === 'img').length;
  const ids = id => reg[id].all(e => e.className === 'id').map(e => e.textContent);
  const check = (tag, got, want) => { if (JSON.stringify(got) !== JSON.stringify(want)) throw new Error(tag + ': ' + JSON.stringify(got) + ' != ' + JSON.stringify(want)); };
  console.log('header rules', reg.rules.children.length, '| counts lines', reg.counts.children.length);
  if (!reg.rules.children.length || reg.counts.children.length < 4) throw new Error('header empty');
  check('ids a', ids('mainA'), D.panel_a.map(c => c.id));
  console.log('A: candidates', D.panel_a.length, 'imgs', imgs('mainA'));
  reg.tabB.fire('click'); await wait();
  check('ids b', ids('mainB'), D.panel_b.map(c => c.id));
  console.log('B: candidates', D.panel_b.length, 'imgs', imgs('mainB'), 'groups', reg.mainB.all(e => e.className === 'grp').length);
  console.log((reg.mainB.children[0] ? reg.mainB.children[0].textContent : ''));
  reg.tabC.fire('click'); await wait();
  for (const a of ['emotion', 'style', 'genre']) {
    reg.cAspect.value = a; reg.cAspect.fire('change');
    check('ids c ' + a, ids('mainC'), D.clusters[a].map(r => r.id));
    console.log('C', a, 'rows', D.clusters[a].length, 'imgs', imgs('mainC'), 'in figure', reg.mainC.all(e => /\bfig\b/.test(e.className)).length);
  }
  const marks = reg.mainC.all(e => e.className === 'ok').length;
  console.log('green marks (genre tab)', marks);
  if (body.children.length) throw new Error('page error: ' + body.textContent);
  console.log('OK');
})().catch(e => { console.error('FAILED', e); process.exit(1); });
