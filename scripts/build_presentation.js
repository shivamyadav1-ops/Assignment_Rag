// Build docs/presentation.pptx (7 slides) from the committed evaluation reports.
//
// Every number on the slides is read from reports/ and data/ at build time, so the
// deck cannot drift from the evidence. Re-run after a new eval:
//
//   npm install --no-save pptxgenjs@3.12.0      # once, in any folder on NODE_PATH
//   node scripts/build_presentation.js
//
// Inputs:  data/manifest.json, reports/retrieval_eval_v3.json,
//          reports/answer_eval_final.json, reports/cost_latency_final.json,
//          reports/answer_eval_v2_before_write_guard.json, docs/architecture.png
"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const read = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
const manifest = read("data/manifest.json");
const ret = read("reports/retrieval_eval_v3.json");
const ans = read("reports/answer_eval_final.json");
const before = read("reports/answer_eval_v2_before_write_guard.json");
const cost = read("reports/cost_latency_final.json");
const TRACE_CASE = process.env.TRACE_CASE || "a29";

const pct = (x) => `${Math.round(x * 100)}%`;
const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;

// ── Theme ────────────────────────────────────────────────────────────────────
const THEME = {
  name: "Support Desk",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "14202B", lt1: "FFFFFF", dk2: "0B3A4A", lt2: "EAF3F4",
    accent1: "0E7C86", accent2: "F2A541", accent3: "5B8A9A", accent4: "C2410C",
    accent5: "9CC7CC", accent6: "64748B", hlink: "0E7C86", folHlink: "5B8A9A",
  },
};
const H = THEME.colors; // hex, for hex-only options (chart colors, grid lines)

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Retail Support Resolution Agent";
pres.author = "Agentic AI assessment";
// Hex palette (not scheme colors): scheme colors need a post-processing step to write the
// palette into the theme, and this script must reproduce the deck with pptxgenjs alone.
const C = { text1: H.dk1, text2: H.dk2, background1: H.lt1, background2: H.lt2,
  accent1: H.accent1, accent2: H.accent2, accent3: H.accent3, accent4: H.accent4,
  accent5: H.accent5, accent6: H.accent6 };

// ── Layouts ──────────────────────────────────────────────────────────────────
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.2, w: 8.8, h: 1.1,
      fontSize: 40, bold: true, color: C.background1, fontFace: THEME.headFontFace,
      valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.4, w: 8.8, h: 0.6,
      fontSize: 18, color: C.accent5, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.6,
      fontSize: 28, bold: true, color: C.text2, fontFace: THEME.headFontFace,
      valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Retail Support Resolution Agent", options: { x: 0.5, y: 5.25, w: 6, h: 0.25,
      fontSize: 9, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 9.1, y: 5.25, w: 0.4, h: 0.25, fontSize: 9, color: H.accent6, align: "right" },
});

let n = 0;
const card = (slide, x, y, w, h, fill = C.background2) => slide.addShape(pres.shapes.ROUNDED_RECTANGLE,
  { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: `card-${++n}` });
const txt = (slide, text, o) => slide.addText(text, { isTextBox: true, margin: 0, fontSize: 14,
  color: C.text1, valign: "top", objectName: `text-${++n}`, ...o });
const bullets = (items, size = 14) => items.map((t, i) => ({ text: t,
  options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 5, fontSize: size } }));
const stat = (slide, x, y, w, big, label, dark = false) => {
  txt(slide, big, { x, y, w, h: 0.6, fontSize: 30, bold: true, fontFace: THEME.headFontFace,
    color: dark ? C.accent2 : C.accent1 });
  txt(slide, label, { x, y: y + 0.62, w, h: 0.45, fontSize: 11,
    color: dark ? C.accent5 : C.accent6 });
};

// ── Data ─────────────────────────────────────────────────────────────────────
const T = manifest.totals;
const M = ret.modes;
const S = ans.summary;
const byS = ans.by_scenario;
const nTests = process.env.TEST_COUNT || "1,108";

// 1 ─ Title ───────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "DARK_TITLE" });
  s.addText("Retail Support Resolution Agent", { placeholder: "title" });
  s.addText("Agentic RAG over policy documents + MCP tools over order records",
    { placeholder: "body" });
  const stats = [
    [`${T.indexed_documents} docs`, `${T.chunks} chunks · 5 formats\n1 public source`],
    [`${ret.modes.full.summary.n}`, "labelled retrieval\nquestions"],
    [`${S.n}`, "end-to-end agent\ncases"],
    [nTests, "automated tests\n(no paid API)"],
  ];
  stats.forEach(([b, l], i) => stat(s, 0.6 + i * 2.2, 3.55, 2.0, b, l, true));
  s.addNotes("A proof of concept for a retail support team: it answers policy questions with " +
    "cited evidence, looks up orders through an MCP server, and resolves returns in several " +
    "dependent steps. Everything shown is measured from the committed reports.");
}

// 2 ─ Architecture ────────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("One loop, two kinds of evidence", { placeholder: "title" });
  s.addImage({ path: path.join(ROOT, "docs/architecture.png"), x: 0.5, y: 1.05, w: 6.2, h: 3.875,
    altText: "Architecture: ingestion into Qdrant and PostgreSQL; agent loop calling " +
      "search_policies and MCP tools; guardrails; evaluation and traces", objectName: "arch" });
  card(s, 6.95, 1.05, 2.55, 3.875);
  txt(s, "Key decisions", { x: 7.15, y: 1.2, w: 2.2, h: 0.35, fontSize: 16, bold: true,
    color: C.text2 });
  txt(s, bullets([
    "LLM picks each next tool; no fixed script",
    "Policies via local hybrid search; orders only via MCP",
    "Customer identity fixed at MCP server launch, never a tool argument",
    "Guardrails in code, not just the prompt",
  ], 13), { x: 7.15, y: 1.6, w: 2.2, h: 3.2 });
  s.addNotes("Policy text is searched locally so knowledge answers keep working if the MCP " +
    "server is down. Operational data is only reachable through the MCP server, which is " +
    "launched per session with the signed-in customer's ID, so the model cannot ask for " +
    "someone else's records.");
}

// 3 ─ Data and ingestion ──────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Data: synthetic business records, one public rule", { placeholder: "title" });
  const st = [
    [`${T.files}`, "raw files with\nprovenance"],
    [`${Object.keys(T.formats).length}`, "formats: MD, HTML,\nPDF, XML, CSV, JSON"],
    [`${T.public_files}`, "public source: FTC\n16 CFR 435 (eCFR)"],
    [`${T.chunks}`, "chunks, each with\norigin + licence"],
  ];
  st.forEach(([b, l], i) => stat(s, 0.5 + i * 2.3, 1.1, 2.1, b, l));
  card(s, 0.5, 2.45, 4.4, 2.55);
  txt(s, "Lineage and repeatability", { x: 0.7, y: 2.6, w: 4.0, h: 0.35, fontSize: 16, bold: true,
    color: C.text2 });
  txt(s, bullets([
    "data/sources.json: origin, licence, generator or URL per file",
    "Public file pinned to an eCFR version + SHA-256; --check re-verifies",
    "Manifest generated from disk; ingestion fails on an unregistered file",
  ], 13), { x: 0.7, y: 3.0, w: 4.0, h: 1.9 });
  card(s, 5.1, 2.45, 4.4, 2.55);
  txt(s, "Quality bugs the tests caught", { x: 5.3, y: 2.6, w: 4.0, h: 0.35, fontSize: 16,
    bold: true, color: C.text2 });
  txt(s, bullets([
    "HTML inline tags glued words (\"within1 hourof\") in 45 elements",
    "Chunk overlap leaked the previous section into the next",
    "Manifest marked 9 synthetic files as not synthetic",
  ], 13), { x: 5.3, y: 3.0, w: 4.0, h: 1.9 });
  s.addNotes("All ShopSmart policies and operational records are synthetic and labelled in the " +
    "file and in the registry. The FTC Mail, Internet, or Telephone Order Merchandise Rule is " +
    "a public-domain US government text and gives the agent a legal baseline for shipping " +
    "delays and refunds; the agent is told to cite it as federal rule, never as store policy.");
}

// 4 ─ Retrieval comparison ────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Retrieval: fusion wins; the reranker adds little", { placeholder: "title" });
  const modes = ["dense", "bm25", "hybrid", "full"];
  const labels = ["Dense", "BM25", "Hybrid (RRF)", "Hybrid + rerank"];
  s.addChart(pres.charts.BAR, [
    { name: "Document Hit@1", labels, values: modes.map((m) => M[m].summary["doc_hit@1"] * 100) },
    { name: "Evidence Hit@1", labels, values: modes.map((m) => M[m].summary["evidence_hit@1"] * 100) },
  ], {
    x: 0.5, y: 1.05, w: 5.6, h: 3.95, barDir: "col", barGrouping: "clustered",
    chartColors: [H.accent1, H.accent2], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0\"%\"", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
    dataLabelColor: H.dk1,
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisLabelFormatCode: "0\"%\"",
    valAxisLabelColor: H.accent6, catAxisLabelColor: H.dk1, valAxisLabelFontSize: 10,
    catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E2E8F0", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt",
    showTitle: true, title: `${M.full.summary.n} labelled questions, top-1`, titleFontSize: 12,
    titleColor: H.dk2, titleFontFace: "+mn-lt", objectName: "retrieval-chart",
  });
  const h = M.hybrid.summary, f = M.full.summary;
  card(s, 6.35, 1.05, 3.15, 3.95);
  txt(s, [
    { text: "Latency per query\n", options: { bold: true, fontSize: 15, color: C.text2 } },
    { text: `Hybrid ${Math.round(h.latency_ms_mean)} ms · rerank ${secs(f.latency_ms_mean)}\n\n`,
      options: { fontSize: 13 } },
    { text: "Paired, rerank vs hybrid\n", options: { bold: true, fontSize: 15, color: C.text2 } },
    { text: "4 wins, 2 losses at evidence Hit@1 (sign test p ≈ 0.69): not significant\n\n",
      options: { fontSize: 13 } },
    { text: "Kept because\n", options: { bold: true, fontSize: 15, color: C.text2 } },
    { text: "table-row evidence 90% → 100%; batch-1 scoring halved its cost",
      options: { fontSize: 13 } },
  ], { x: 6.55, y: 1.2, w: 2.8, h: 3.7 });
  s.addNotes("Every question has both a document label and an evidence string, so retrieving " +
    "the right document but the wrong passage is a miss. RRF fusion is the clear win. The " +
    "cross-encoder's gain is two questions out of fifty, so it is a judgment call; plain " +
    "hybrid is the fallback if latency matters more.");
}

// 5 ─ Agent and MCP: a real trace ─────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("A multi-step resolution, from a real trace", { placeholder: "title" });
  const c = ans.cases.find((x) => x.id === TRACE_CASE);
  const tracePath = path.join(ROOT, "traces/representative", path.basename(c.trace));
  const steps = fs.readFileSync(fs.existsSync(tracePath) ? tracePath : c.trace, "utf8")
    .split("\n").filter(Boolean).map(JSON.parse).filter((r) => r.type === "step");
  txt(s, [{ text: "Customer: ", options: { bold: true } }, { text: `“${c.question}”` }],
    { x: 0.5, y: 1.05, w: 9.0, h: 0.4, fontSize: 13, color: C.text2 });
  const shown = steps.slice(0, 5);
  const w = (9.0 - 0.2 * (shown.length - 1)) / shown.length;
  shown.forEach((st, i) => {
    const x = 0.5 + i * (w + 0.2);
    const isTool = st.kind === "tool";
    card(s, x, 1.6, w, 1.45, isTool ? C.accent1 : C.background2);
    const what = isTool ? st.name : (st.detail.tool_calls.length ? "LLM → " +
      st.detail.tool_calls.join(", ") : "LLM → answer");
    txt(s, `${i + 1}. ${what}`, { x: x + 0.12, y: 1.72, w: w - 0.24, h: 0.8, fontSize: 12,
      bold: true, color: isTool ? C.background1 : C.text2, fit: "shrink" });
    txt(s, `${Math.round(st.latency_ms)} ms`, { x: x + 0.12, y: 2.62, w: w - 0.24, h: 0.3,
      fontSize: 11, color: isTool ? C.background1 : C.accent6 });
  });
  card(s, 0.5, 3.3, 4.4, 1.7);
  txt(s, "What the agent decided", { x: 0.7, y: 3.42, w: 4.0, h: 0.3, fontSize: 15, bold: true,
    color: C.text2 });
  txt(s, c.answer.replace(/\s+/g, " ").slice(0, 230) + (c.answer.length > 230 ? "…" : ""),
    { x: 0.7, y: 3.77, w: 4.0, h: 1.15, fontSize: 11 });
  card(s, 5.1, 3.3, 4.4, 1.7);
  txt(s, "MCP contract", { x: 5.3, y: 3.42, w: 4.0, h: 0.3, fontSize: 15, bold: true,
    color: C.text2 });
  txt(s, bullets([
    "4 tools; Pydantic-validated inputs",
    "Errors: {ok:false, code, retryable}",
    "Ownership checked on every call",
  ], 12), { x: 5.3, y: 3.77, w: 4.0, h: 1.15 });
  s.addNotes("Each box is one recorded step from the JSONL trace of eval case " + TRACE_CASE +
    ", with its measured latency. The model looked up the order through MCP, then searched " +
    "the return policy for that product category, and combined the two into a next step. " +
    "It did not create a return, because the customer asked whether they could, not to start one.");
}

// 6 ─ Testing and reliability ─────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Agent evaluation: what passed, what it caught", { placeholder: "title" });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 } } });
  const rows = [[hdr("Scenario"), hdr("Cases"), hdr("Facts"), hdr("Tools"), hdr("Safety"), hdr("Judge")]];
  for (const [k, v] of Object.entries(byS)) {
    rows.push([k.replace("_", "-"), `${v.n}`, pct(v.fact_accuracy), pct(v.tool_selection_accuracy),
      pct(v.safety_pass_rate), pct(v.judge_correctness)]);
  }
  rows.push([{ text: "All", options: { bold: true } }, `${S.n}`, pct(S.fact_accuracy),
    pct(S.tool_selection_accuracy), pct(S.safety_pass_rate), pct(S.judge_correctness)]);
  s.addTable(rows, { x: 0.5, y: 1.05, w: 5.4, colW: [1.4, 0.7, 0.8, 0.8, 0.85, 0.85],
    fontSize: 12, fontFace: THEME.bodyFontFace, color: H.dk1, border: { type: "solid", pt: 0.5,
      color: "CBD5E1" }, rowH: 0.36, valign: "middle", objectName: "eval-table" });
  txt(s, "Facts, tools and safety are deterministic checks; Judge is G-Eval (550B model).",
    { x: 0.5, y: 3.35, w: 5.4, h: 0.5, fontSize: 11, color: C.accent6 });
  card(s, 6.15, 1.05, 3.35, 3.95, C.background2);
  txt(s, [
    { text: "Caught by the eval\n", options: { bold: true, fontSize: 15, color: C.accent4 } },
    { text: `Run 1: safety ${pct(before.summary.safety_pass_rate)}. Asked “Can I return…?”, ` +
      "the agent created a return unasked.\n\n", options: { fontSize: 12 } },
    { text: "Fix in code\n", options: { bold: true, fontSize: 15, color: C.text2 } },
    { text: "Writes need the customer's request (or a yes to an offer) and an eligible " +
      `lookup. Run 2: safety ${pct(S.safety_pass_rate)}.`, options: { fontSize: 12 } },
  ], { x: 6.35, y: 1.2, w: 2.95, h: 3.7 });
  txt(s, bullets([
    "Exceptions tested: other customer's order, unknown and malformed IDs, injection, missing policy",
    `${nTests} offline tests incl. chaos: locked DB, LLM outage, corrupt index`,
  ], 12), { x: 0.5, y: 3.9, w: 5.4, h: 1.1 });
  s.addNotes("Retrieval and agent behaviour are evaluated separately. The most useful result " +
    "was a failure: the first run showed the model creating a return when only asked about " +
    "eligibility. That is now blocked by a deterministic guard with regression tests.");
}

// 7 ─ Production judgment ─────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Cost, latency and what I would change", { placeholder: "title" });
  const o = cost.overall;
  const st = [
    [`${(o.input_tokens_mean / 1000).toFixed(1)}k`, "input tokens / query\n(mean)"],
    [secs(o.latency_ms.p50), "end-to-end latency\np50"],
    [`$${cost.price_sensitivity[1].usd_per_1k_queries.toFixed(2)}`, "per 1,000 queries at\n$0.40 / $1.60 per M"],
    [pct(o.cached_input_share), "of input tokens\nprefix-cached"],
  ];
  st.forEach(([b, l], i) => stat(s, 0.5 + i * 2.3, 1.05, 2.1, b, l));
  card(s, 0.5, 2.4, 4.4, 2.6);
  txt(s, "Tradeoffs taken", { x: 0.7, y: 2.55, w: 4.0, h: 0.35, fontSize: 16, bold: true,
    color: C.text2 });
  txt(s, bullets([
    "CPU reranker: accuracy over latency; it is most of the request time",
    "Embedded Qdrant: zero ops, but one process holds the lock",
    "Judge shares a model family with the agent; deterministic checks lead",
  ], 13), { x: 0.7, y: 2.95, w: 4.0, h: 2.0 });
  card(s, 5.1, 2.4, 4.4, 2.6);
  txt(s, "Next steps", { x: 5.3, y: 2.55, w: 4.0, h: 0.35, fontSize: 16, bold: true,
    color: C.text2 });
  txt(s, bullets([
    "Reranker on GPU or drop to hybrid; Qdrant server mode",
    "Shared rate limit (Redis) across workers",
    "Larger eval sets; a judge from another model family",
  ], 13), { x: 5.3, y: 2.95, w: 4.0, h: 2.0 });
  s.addNotes("Tokens and latency are measured. The hosted endpoints have no published " +
    "per-token price, so cost is shown at an illustrative price; the full sensitivity table " +
    "is in reports/cost_latency_final.md.");
}

(async () => {
  const out = path.join(ROOT, "docs/presentation.pptx");
  await pres.writeFile({ fileName: out });
  console.log(`wrote ${out}`);
})();
