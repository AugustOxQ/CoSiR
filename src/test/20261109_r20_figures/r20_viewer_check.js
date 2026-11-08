// Headless check of the viewer page's script against a minimal DOM stub (no browser here).
// Usage: node r20_viewer_check.js <viewer folder>
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
  const wait = async () => { for (let i = 0; i < 20; i++) await new Promise(r => setTimeout(r, 30)); };
  await wait();
  const rows = id => reg[id].children.length;
  const imgs = id => reg[id].all(e => e.tagName === 'img').length;
  const report = (tag) => console.log(tag, 'cstar rows', rows('colCstar'), 'imgs', imgs('colCstar'), '| plain rows', rows('colPlain'), 'imgs', imgs('colPlain'));
  report('initial');
  if (body.children.length) throw new Error('page error: ' + body.textContent);
  let combos = 0;
  for (const set of ['artelingo', 'mmart']) {
    reg.cSet.value = set; reg.cSet.fire('change'); await wait();
    for (const a of ['emotion', 'style', 'genre']) {
      reg.cAspect.value = a; reg.cAspect.fire('change');
      const ks = reg.cK.children.map(o => o.attrs.value);
      for (const k of ks) { reg.cK.value = k; reg.cK.fire('change');
        for (const cr of ['per_class', 'purity', 'random']) { reg.cCR.value = cr; reg.cCR.fire('change');
          for (const pr of ['centre', 'random']) { reg.cPR.value = pr; reg.cPR.fire('change');
            for (const n of ['3', '6']) { reg.cN.value = n; reg.cN.fire('change'); combos++;
              if (!rows('colCstar') || !rows('colPlain')) throw new Error(`empty panel ${set} ${a} ${k} ${cr} ${pr}`);
            } } } }
    }
    report(set + ' last');
    reg.tabT.fire('click');
    const T = JSON.parse(fs.readFileSync(path.join(dir, 'data/candidates.json'))).sets[set].teaser.length;
    for (let i = 0; i < T; i++) { reg.strip.children[i].fire('click'); for (const n of ['3', '5']) { reg.tN.value = n; reg.tN.fire('change'); } }
    reg.tPlain.checked = false; reg.tPlain.fire('change'); reg.tPlain.checked = true; reg.tPlain.fire('change');
    console.log(set, 'teaser candidates', T, 'strip', reg.strip.children.length, 'blocks', reg.teaser.children.length, 'imgs', reg.teaser.all(e => e.tagName === 'img').length);
    console.log((reg.teaser.children[1] ? reg.teaser.children[1].textContent : '').slice(0, 200));
    reg.tabC.fire('click');
  }
  console.log('cluster combinations rendered:', combos, 'OK');
})().catch(e => { console.error('FAILED', e); process.exit(1); });
