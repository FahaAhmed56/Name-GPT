// Run with: node web/test.js
const fs = require("fs"), path = require("path"), vm = require("vm"), assert = require("assert");
const { NameModel, mulberry32 } = require("./lm.js");
const ctx = {}; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, "model.js"), "utf8") + "\nthis.M = MODEL;", ctx);
const model = new NameModel(JSON.parse(JSON.stringify(ctx.M)));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "test_fixtures.json"), "utf8"));
let passed = 0; const test = (n, f) => { f(); passed++; console.log("  ✓ " + n); };

test("JS neural network gives the same probabilities as Python", () => {
  let worst = 0;
  fx.contexts.forEach((c, i) => model.probsMLP(c).forEach((p, k) => (worst = Math.max(worst, Math.abs(p - fx.probs[i][k])))));
  console.log(`    (largest difference ${worst.toExponential(1)})`);
  assert.ok(worst < 1e-5);
});
test("probabilities always sum to 1", () => {
  for (const c of [[0, 0, 0], [5, 13, 13], [26, 26, 26]]) for (const k of ["mlp", "bigram"])
    assert.ok(Math.abs(model.probs(c, k).reduce((a, b) => a + b, 0) - 1) < 1e-9);
});
test("generated names only use a-z, and respect the prefix", () => {
  const r = mulberry32(3);
  for (let i = 0; i < 200; i++) {
    const { name } = model.generate("mlp", { prefix: "fa", rand: r });
    assert.match(name, /^fa[a-z]*$/);
  }
});
test("same seed -> same names (reproducible)", () => {
  const run = () => { const r = mulberry32(9); return Array.from({ length: 5 }, () => model.generate("mlp", { rand: r }).name); };
  assert.deepStrictEqual(run(), run());
});
test("real names score as more name-like than keyboard mashing", () => {
  for (const real of ["emma", "olivia", "fahad", "james"]) for (const junk of ["xqzt", "kkkkk", "zxcvb"])
    assert.ok(model.score(real, "mlp").loss < model.score(junk, "mlp").loss, `${real} vs ${junk}`);
});
test("low temperature produces more real (common) names than high temperature", () => {
  const ctx2 = {}; vm.createContext(ctx2);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "names.js"), "utf8") + "\nthis.N = NAMES;", ctx2);
  const real = t => { const r = mulberry32(1); let s = 0;
    for (let i = 0; i < 300; i++) s += ctx2.N.has(model.generate("mlp", { temperature: t, rand: r }).name); return s / 300; };
  const lo = real(0.5), hi = real(1.5);
  console.log(`    (real-name rate: temperature 0.5 -> ${(lo * 100).toFixed(0)}%, 1.5 -> ${(hi * 100).toFixed(0)}%)`);
  assert.ok(lo > hi);
});
console.log(`\nAll ${passed} tests passed ✓`);
