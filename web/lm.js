// The trained neural language model (and the bigram baseline), running in the browser.
function mulberry32(seed) {
  return () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

class NameModel {
  constructor(m) { this.m = m; this.chars = m.chars; this.block = m.block; }

  // P(next char | previous `block` chars) for the neural model (same maths as training/model.py)
  probsMLP(ctx) {
    const { C, W1, b1, W2, b2 } = this.m;
    const e = ctx.flatMap(i => C[i]);
    const h = b1.map((b, j) => { let s = b; for (let i = 0; i < e.length; i++) s += e[i] * W1[i][j]; return Math.tanh(s); });
    const logits = b2.map((b, k) => { let s = b; for (let j = 0; j < h.length; j++) s += h[j] * W2[j][k]; return s; });
    const mx = Math.max(...logits), ex = logits.map(z => Math.exp(z - mx)), sum = ex.reduce((a, b) => a + b, 0);
    return ex.map(v => v / sum);
  }
  probsBigram(ctx) {                                // renormalise: the exported table is rounded to 6 decimals
    const row = this.m.bigram[ctx[ctx.length - 1]], s = row.reduce((a, b) => a + b, 0);
    return row.map(v => v / s);
  }
  probs(ctx, kind) { return kind === "bigram" ? this.probsBigram(ctx) : this.probsMLP(ctx); }

  // Generate one name. Temperature < 1 = safer, more common names; > 1 = more creative / chaotic.
  generate(kind, { temperature = 1, prefix = "", rand = Math.random, maxLen = 20 } = {}) {
    let ctx = new Array(this.block).fill(0), name = "";
    const steps = [];
    for (const ch of prefix) {                       // feed the user's starting letters first
      const ix = this.chars.indexOf(ch); if (ix < 1) continue;
      steps.push({ ctx: ctx.slice(), probs: this.probs(ctx, kind), chosen: ix, forced: true });
      name += ch; ctx = [...ctx.slice(1), ix];
    }
    while (name.length < maxLen) {
      const p = this.probs(ctx, kind);
      const scaled = p.map(v => Math.pow(v, 1 / temperature)), z = scaled.reduce((a, b) => a + b, 0);
      let r = rand() * z, ix = 0;
      while (ix < scaled.length - 1 && (r -= scaled[ix]) > 0) ix++;
      steps.push({ ctx: ctx.slice(), probs: p, chosen: ix, forced: false });
      if (ix === 0) break;                            // '.' = end of name
      name += this.chars[ix]; ctx = [...ctx.slice(1), ix];
    }
    return { name, steps };
  }

  // How "name-like" is a word? Average -log P per letter (incl. the end), lower = more name-like.
  score(word, kind) {
    let ctx = new Array(this.block).fill(0), total = 0;
    const letters = [];
    for (const ch of word + ".") {
      const ix = this.chars.indexOf(ch); if (ix < 0) continue;
      const p = this.probs(ctx, kind)[ix];
      letters.push({ ch, p }); total += -Math.log(p);
      ctx = [...ctx.slice(1), ix];
    }
    return { letters, loss: total / Math.max(1, letters.length) };
  }
}

if (typeof module !== "undefined") module.exports = { NameModel, mulberry32 };
