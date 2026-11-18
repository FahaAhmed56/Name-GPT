const $ = id => document.getElementById(id);
const model = new NameModel(MODEL);
const show = c => (c === "." ? "END" : c);
let kind = "mlp", generated = [];

$("lossMlp").textContent = MODEL.test_loss.toFixed(3);
$("lossBi").textContent = MODEL.bigram_test_loss.toFixed(3);

function generate() {
  const temperature = +$("temp").value;
  const prefix = $("prefix").value.toLowerCase().replace(/[^a-z]/g, "");
  generated = Array.from({ length: 12 }, () => model.generate(kind, { temperature, prefix }));
  const isNew = g => g.name && !NAMES.has(g.name);
  $("names").innerHTML = generated.map((g, i) =>
    `<button class="name ${isNew(g) ? "new" : "real"}" data-i="${i}">${isNew(g) ? "✨" : "📖"} ${g.name || "<i>(empty)</i>"}</button>`).join("");
  const n = generated.filter(isNew).length;
  $("novelty").textContent = `${n} of 12 are new inventions`;
  showSteps(0);
}

function showSteps(i) {
  const g = generated[i]; if (!g) return;
  document.querySelectorAll(".name").forEach(b => b.classList.toggle("sel", +b.dataset.i === i));
  $("stepName").textContent = g.name;
  $("steps").innerHTML = g.steps.map(s => {
    const ctxText = s.ctx.map(c => (c === 0 ? "·" : model.chars[c])).join("");
    const top = s.probs.map((p, k) => [k, p]).sort((a, b) => b[1] - a[1]).slice(0, 5);
    if (!top.some(([k]) => k === s.chosen)) top[4] = [s.chosen, s.probs[s.chosen]];   // always show the chosen letter
    return `<div class="step">
      <div class="ctx"><code>${ctxText}</code> →</div>
      <div class="opts">${top.map(([k, p]) => `
        <div class="opt ${k === s.chosen ? "chosen" : ""}"><b>${show(model.chars[k])}</b>
          <span class="track"><span class="fill" style="width:${Math.max(2, p * 100)}%"></span></span><em>${p < 0.01 ? "<1" : (p * 100).toFixed(0)}%</em></div>`).join("")}
      </div>${s.forced ? '<div class="forced">your prefix</div>' : ""}</div>`;
  }).join("");
}

function scoreWord() {
  const w = $("tryWord").value.toLowerCase().replace(/[^a-z]/g, "");
  if (!w) { $("scoreOut").innerHTML = ""; return; }
  const s = model.score(w, kind);
  // map loss to a friendly 0-100 score: typical real names ~2.0, keyboard mashing ~4.5+
  const score = Math.round(Math.max(0, Math.min(100, (4.5 - s.loss) / (4.5 - 1.5) * 100)));
  const verdict = score >= 70 ? "Very name-like" : score >= 45 ? "Could be a name" : "Not very name-like";
  $("scoreOut").innerHTML = `
    <div class="verdict"><strong>${score}</strong><span>/100 · ${verdict}${NAMES.has(w) ? " · 📖 in the dataset" : ""}</span></div>
    <div class="letters">${s.letters.map(l => `<span title="P = ${(l.p * 100).toFixed(1)}%" style="--a:${Math.min(1, l.p * 3)}">${show(l.ch)}<small>${(l.p * 100).toFixed(0)}%</small></span>`).join("")}</div>
    <p class="info">Each letter shows how likely the model thought it was, given the letters before it. Darker = more expected.</p>`;
}

document.querySelectorAll(".kind").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll(".kind").forEach(x => x.classList.remove("active"));
  b.classList.add("active"); kind = b.dataset.kind; generate(); scoreWord();
}));
$("temp").addEventListener("input", () => ($("tOut").textContent = (+$("temp").value).toFixed(1)));
$("temp").addEventListener("change", generate);
$("gen").addEventListener("click", generate);
$("prefix").addEventListener("keydown", e => e.key === "Enter" && generate());
$("names").addEventListener("click", e => { const b = e.target.closest(".name"); if (b) showSteps(+b.dataset.i); });
$("tryWord").addEventListener("input", scoreWord);
generate(); scoreWord();
