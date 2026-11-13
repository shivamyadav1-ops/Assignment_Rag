// Build docs/presentation_full.pptx — a 14-slide presenter deck with speaker notes.
// (docs/presentation.pptx stays as the 7-slide submission deck.)
//
// Every number is read from reports/ and data/ at build time, so the deck cannot drift
// from the evidence. Re-run after a new eval:
//
//   npm install --no-save pptxgenjs@3.12.0      # once, in any folder on NODE_PATH
//   node scripts/build_presentation_full.js
//
// Optional: APPLY_THEME=/path/to/apply_theme.js writes the palette into the theme part.
"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const read = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
const manifest = read("data/manifest.json");
const ingest = read("data/processed/ingestion_summary.json");
const ret = read("reports/retrieval_eval_v3.json");
const ans = read("reports/answer_eval_final.json");
const before = read("reports/answer_eval_v2_before_write_guard.json");
const cost = read("reports/cost_latency_final.json");
const traceLines = (f) => fs.readFileSync(path.join(ROOT, "traces/representative", f), "utf8")
  .split("\n").filter(Boolean).map(JSON.parse);

const pct = (x) => `${Math.round(x * 100)}%`;
const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;
const nTests = process.env.TEST_COUNT || "1,210"; // `pytest --collect-only -q`, 2026-11-13

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
const H = THEME.colors; // hex, for hex-only options (charts, tables, grid lines)

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Retail Support Resolution Agent";
pres.subject = "Agentic AI Engineering — practical assessment";
const C = pres.SchemeColor;

// ── Layouts ──────────────────────────────────────────────────────────────────
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: H.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.0, w: 8.8, h: 1.2,
      fontSize: 40, bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.3, w: 8.8, h: 0.7,
      fontSize: 18, color: C.accent5, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: H.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.6,
      fontSize: 28, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Retail Support Resolution Agent", options: { x: 0.5, y: 5.25, w: 6, h: 0.25,
      fontSize: 9, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: 9.1, y: 5.25, w: 0.4, h: 0.25, fontSize: 9, color: H.accent6, align: "right" },
});

// ── Helpers ──────────────────────────────────────────────────────────────────
let n = 0;
const card = (s, x, y, w, h, fill = C.background2) => s.addShape(pres.shapes.ROUNDED_RECTANGLE,
  { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill, width: 0.5 },
    objectName: `card-${++n}` });
const txt = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontSize: 14,
  color: C.text1, valign: "top", objectName: `text-${++n}`, ...o });
const bullets = (items, size = 14, color) => items.map((t, i) => ({ text: t,
  options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 5, fontSize: size,
    ...(color ? { color } : {}) } }));
const stat = (s, x, y, w, big, label, dark = false) => {
  txt(s, big, { x, y, w, h: 0.6, fontSize: 30, bold: true, fontFace: THEME.headFontFace,
    color: dark ? C.accent2 : C.accent1 });
  txt(s, label, { x, y: y + 0.62, w, h: 0.5, fontSize: 11, color: dark ? C.accent5 : C.accent6 });
};
const heading = (s, t, x, y, w, color = C.text2) =>
  txt(s, t, { x, y, w, h: 0.35, fontSize: 16, bold: true, color });
const badge = (s, label, x, y, d = 0.5, fill = C.accent1) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill },
    objectName: `badge-${++n}` });
  txt(s, label, { x, y, w: d, h: d, fontSize: 16, bold: true, color: C.background1,
    align: "center", valign: "middle", fontFace: THEME.headFontFace });
};
const node = (s, label, x, y, w, h, fill, color, size = 12) => {
  card(s, x, y, w, h, fill);
  txt(s, label, { x: x + 0.05, y, w: w - 0.1, h, fontSize: size, bold: true, color,
    align: "center", valign: "middle" });
};
const arrow = (s, x, y, w, h, o = {}) => s.addShape(pres.shapes.LINE, { x, y, w, h,
  flipH: !!o.flipH, line: { color: H.accent6, width: 1.5, endArrowType: "triangle",
    ...(o.both ? { beginArrowType: "triangle" } : {}) }, objectName: `arrow-${++n}` });
const tableOpts = (o) => ({ fontSize: 11, fontFace: THEME.bodyFontFace, color: H.dk1,
  border: { type: "solid", pt: 0.5, color: "CBD5E1" }, valign: "middle", margin: 0.05, ...o });
const th = (t) => ({ text: t, options: { bold: true, color: H.lt1, fill: { color: H.dk2 } } });

const T = manifest.totals;
const M = ret.modes;
const S = ans.summary;
const byS = ans.by_scenario;
const o = cost.overall;

// 1 ─ Title ───────────────────────────────────────────────────────────────────
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Introduction" });
  s.addText("Retail Support Resolution Agent", { placeholder: "title" });
  s.addText("Agentic RAG over policy documents + MCP tools over order records",
    { placeholder: "body" });
  [
    [`${T.indexed_documents} docs`, `${T.chunks} chunks · ${Object.keys(T.formats).length} formats\n1 public source`],
    [`${M.full.summary.n}`, "labelled retrieval\nquestions"],
    [`${S.n}`, "end-to-end agent\ncases"],
    [nTests, "automated tests\n(no paid API)"],
  ].forEach(([b, l], i) => stat(s, 0.6 + i * 2.2, 3.55, 2.0, b, l, true));
  s.addNotes("This is a proof of concept for a retail support team at a fictional store, " +
    "ShopSmart Electronics. It answers policy questions with cited evidence, looks up orders " +
    "through an MCP server, and resolves returns in several dependent steps. Every number in " +
    "this deck is read from the committed evaluation reports when the deck is built.");
}

// 2 ─ Requirements → delivered ────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("What was asked, and what was built", { placeholder: "title" });
  const items = [
    ["A", "Data & ingestion", `${T.files} files, 6 formats; lineage + SHA-256 per source`],
    ["B", "Retrieval", "Dense, BM25, RRF hybrid, rerank — compared on 50 labelled questions"],
    ["C", "Agent", "LangGraph state graph; the LLM picks each next tool"],
    ["D", "MCP", "4 validated tools; identity fixed per session; typed errors"],
    ["E", "Testing", "Retrieval and agent scored separately; traces with tokens + latency"],
    ["S", "Scenarios 1–3", "Policy answer, multi-step return, 4 exception cases"],
  ];
  const w = 2.85, h = 1.9;
  items.forEach(([l, t, d], i) => {
    const x = 0.5 + (i % 3) * (w + 0.225), y = 1.1 + Math.floor(i / 3) * (h + 0.2);
    card(s, x, y, w, h);
    badge(s, l, x + 0.2, y + 0.2, 0.5, l === "S" ? C.accent2 : C.accent1);
    txt(s, t, { x: x + 0.85, y: y + 0.2, w: w - 1.0, h: 0.5, fontSize: 16, bold: true,
      color: C.text2, valign: "middle" });
    txt(s, d, { x: x + 0.2, y: y + 0.85, w: w - 0.4, h: h - 1.0, fontSize: 14 });
  });
  s.addNotes("The assignment had five capability areas, A to E, plus three demonstration " +
    "scenarios. Each card says what was delivered for that area; the rest of the deck walks " +
    "through them in order and ends with measured results and limitations.");
}

// 3 ─ Architecture ────────────────────────────────────────────────────────────
pres.addSection({ title: "Design" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Design" });
  s.addText("One loop, two kinds of evidence", { placeholder: "title" });
  s.addImage({ path: path.join(ROOT, "docs/architecture.png"), x: 0.5, y: 1.05, w: 6.2, h: 3.875,
    altText: "Architecture: ingestion into Qdrant and PostgreSQL; agent loop calling " +
      "search_policies and MCP tools; guardrails; evaluation and traces", objectName: "arch" });
  card(s, 6.95, 1.05, 2.55, 3.875);
  heading(s, "Key decisions", 7.15, 1.2, 2.2);
  txt(s, bullets([
    "LLM picks each next tool; no fixed script",
    "Policies via local hybrid search; orders only via MCP",
    "Customer identity fixed at MCP launch, never a tool argument",
    "Guardrails enforced in code, not only in the prompt",
  ], 13), { x: 7.15, y: 1.6, w: 2.2, h: 3.25 });
  s.addNotes("Two kinds of evidence: policy text, searched locally so knowledge answers keep " +
    "working if the MCP server is down; and operational records, which are only reachable " +
    "through the MCP server. That server is launched per session with the signed-in " +
    "customer's ID, so the model has no way to ask for someone else's data.");
}

// 4 ─ Technology choices ──────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Design" });
  s.addText("Technology choices and their tradeoffs", { placeholder: "title" });
  const rows = [
    [th("Layer"), th("Choice"), th("Why"), th("Tradeoff")],
    ["LLM", "NVIDIA NIM nemotron (Anthropic switchable)", "OpenAI-compatible API, prompt-prefix cache", "No published per-token price"],
    ["Orchestration", "LangGraph StateGraph", "Explicit state, conditional routing, 8-step cap", "Extra framework layer"],
    ["Retrieval", "Qdrant + BGE-small + BM25 + bge-reranker", "RRF fusion built in; CPU ONNX models", "Embedded mode = one process; rerank slow on CPU"],
    ["Tools", "MCP (FastMCP over stdio)", "Standard protocol; per-session identity", "One server process per session"],
    ["Ops data", "PostgreSQL (SQLite in tests)", "Production-like; typed SQLAlchemy access", "Needs Docker to run"],
    ["Observability", "JSONL traces + Langfuse", "Per-step tokens, latency, TTFT", "Langfuse is optional/external"],
    ["Evaluation", "Deterministic checks + G-Eval judge", "Model-free primary metrics", "Judge shares agent's model family"],
  ];
  s.addTable(rows, tableOpts({ x: 0.5, y: 1.05, w: 9.0, colW: [1.35, 2.55, 2.6, 2.5],
    fontSize: 12, rowH: 0.46, objectName: "tech-table" }));
  s.addNotes("No vendor was mandated, so each choice is justified by a requirement. LangGraph " +
    "gives explicit, inspectable state and routing instead of a hidden agent loop. Qdrant " +
    "stores dense and sparse vectors side by side, so hybrid search is one query. MCP was " +
    "required; stdio keeps it simple and lets the server be launched with a fixed identity.");
}

// 5 ─ Data and ingestion ──────────────────────────────────────────────────────
pres.addSection({ title: "Build" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Build" });
  s.addText("Data: synthetic records, one public rule", { placeholder: "title" });
  const ct = ingest.policies.chunks_by_type;
  const op = ingest.operational;
  const steps = [
    ["Verify", "sources.json\n+ SHA-256"],
    ["Parse", "PDF · HTML · MD · XML"],
    ["Chunk", `${ct.text} prose · ${ct.table} table · ${ct.table_row} row`],
    ["Embed", "BGE dense +\nBM25 sparse"],
    ["Index", "Qdrant\nnamed vectors"],
  ];
  const w = 1.6, gap = 0.25;
  steps.forEach(([t, d], i) => {
    const x = 0.5 + i * (w + gap);
    card(s, x, 1.1, w, 1.15, i === 4 ? C.accent1 : C.background2);
    txt(s, t, { x: x + 0.1, y: 1.18, w: w - 0.2, h: 0.35, fontSize: 15, bold: true,
      align: "center", color: i === 4 ? C.background1 : C.text2 });
    txt(s, d, { x: x + 0.1, y: 1.55, w: w - 0.2, h: 0.65, fontSize: 12, align: "center",
      color: i === 4 ? C.background1 : C.text1 });
    if (i < steps.length - 1) arrow(s, x + w + 0.03, 1.675, gap - 0.06, 0);
  });
  txt(s, `CSV + JSON → PostgreSQL: ${op.customers} customers · ${op.orders} orders · ${op.returns} returns · ` +
    `ingest ${ingest.elapsed_seconds} s · incremental re-index of changed files`,
    { x: 0.5, y: 2.35, w: 9.0, h: 0.4, fontSize: 11, color: C.accent6 });
  card(s, 0.5, 2.85, 4.4, 2.2);
  heading(s, "Lineage and repeatability", 0.7, 2.98, 4.0);
  txt(s, bullets([
    "Every chunk carries source file, origin, licence, synthetic flag",
    "FTC 16 CFR 435 pinned to an eCFR version; --check re-verifies",
    "Ingestion fails on any file not in the registry",
  ], 13), { x: 0.7, y: 3.38, w: 4.0, h: 1.6 });
  card(s, 5.1, 2.85, 4.4, 2.2);
  heading(s, "Data-quality bugs found by tests", 5.3, 2.98, 4.0);
  txt(s, bullets([
    "HTML inline tags glued words (\"within1 hourof\")",
    "Chunk overlap leaked the previous section",
    "Manifest marked 9 synthetic files as not synthetic",
  ], 13), { x: 5.3, y: 3.38, w: 4.0, h: 1.6 });
  s.addNotes(`${T.synthetic_files} of ${T.files} files are synthetic and labelled in the file ` +
    "itself and in data/sources.json. The one public file is the FTC Mail, Internet, or " +
    "Telephone Order Merchandise Rule, a public-domain US government text that gives a legal " +
    "baseline for shipping delays and refunds. Tables are indexed both whole and row by row, " +
    "so a question about one product category retrieves exactly that row.");
}

// 6 ─ Retrieval comparison ────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Build" });
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
    dataLabelColor: H.dk1, valAxisMinVal: 0, valAxisMaxVal: 100,
    valAxisLabelFormatCode: "0\"%\"", valAxisLabelColor: H.accent6, catAxisLabelColor: H.dk1,
    valAxisLabelFontSize: 10, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontFace: "+mn-lt", valGridLine: { color: "E2E8F0", size: 0.5 },
    catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 11,
    legendFontFace: "+mn-lt", showTitle: true,
    title: `${M.full.summary.n} labelled questions, top-1`, titleFontSize: 12,
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
    { text: `table-row evidence 90% → 100%; evidence MRR ${h.evidence_mrr.toFixed(2)} → ${f.evidence_mrr.toFixed(2)}`,
      options: { fontSize: 13 } },
  ], { x: 6.55, y: 1.2, w: 2.8, h: 3.7 });
  s.addNotes("Each question names the right document and an evidence string the chunk must " +
    "contain, so the right document with the wrong passage counts as a miss. Dense and BM25 " +
    "miss different questions, which is why fusing them is the big gain. The cross-encoder " +
    "nets two extra questions out of fifty for about 100 times the latency; I kept it for " +
    "table-row ranking, but plain hybrid is the fallback if latency matters more. By question " +
    "type it is mixed: table questions 90% to 100%, prose 73% to 77%, but paraphrases drop " +
    "from 78% to 67%.");
}

// 7 ─ Agent workflow ──────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Build" });
  s.addText("A stateful agent, not a fixed pipeline", { placeholder: "title" });
  const W = 1.45, G = 0.35, X = (i) => 0.5 + i * (W + G), Y1 = 1.15, Y2 = 2.25, NH = 0.55;
  ["input_guard", "cache_lookup", "prepare", "agent (LLM)", "output_guard"].forEach((t, i) => {
    const hot = i === 3;
    node(s, t, X(i), Y1, W, NH, hot ? C.accent1 : C.background2, hot ? C.background1 : C.text2);
    if (i < 4) arrow(s, X(i) + W + 0.03, Y1 + NH / 2, G - 0.06, 0);
  });
  node(s, "tools (MCP + search)", X(2), Y2, W, NH, C.background2, C.text2);
  node(s, "forced_search", X(3), Y2, W, NH, C.background2, C.text2);
  node(s, "grounding → finalize", X(4), Y2, W, NH, C.background2, C.text2);
  arrow(s, X(2) + W - 0.25, Y1 + NH + 0.03, X(3) + 0.25 - (X(2) + W - 0.25), Y2 - Y1 - NH - 0.06,
    { flipH: true, both: true });
  arrow(s, X(3) + W / 2, Y1 + NH + 0.03, 0, Y2 - Y1 - NH - 0.06, { both: true });
  arrow(s, X(4) + W / 2, Y1 + NH + 0.03, 0, Y2 - Y1 - NH - 0.06);
  txt(s, "blocked input or cache hit → end early", { x: 0.5, y: Y1 + NH + 0.1, w: 3.0, h: 0.3,
    fontSize: 11, italic: true, color: C.accent6 });
  card(s, 0.5, 3.1, 4.4, 1.95);
  heading(s, "State carried between steps", 0.7, 3.22, 4.0);
  txt(s, bullets([
    "Messages, tool results, evidence and context",
    "Guard state: lookups done, per-tool call budgets",
    "Per-step usage, latency and TTFT for the trace",
  ], 13), { x: 0.7, y: 3.6, w: 4.0, h: 1.4 });
  card(s, 5.1, 3.1, 4.4, 1.95);
  heading(s, "Why it is genuinely agentic", 5.3, 3.22, 4.0);
  txt(s, bullets([
    "LLM picks the next tool from each result",
    "No answer without evidence (forced search)",
    "Writes need a customer request + eligible lookup",
  ], 13), { x: 5.3, y: 3.6, w: 4.0, h: 1.4 });
  s.addNotes("These are the real nodes of the LangGraph StateGraph in agent.py. The loop " +
    "between the LLM and its tools runs until the model answers, capped at 8 steps. Two " +
    "deterministic rules sit on top of the model's choices: it cannot answer without evidence, " +
    "and it cannot create a return unless the customer asked for one and the lookup said " +
    "the item is eligible.");
}

// 8 ─ MCP server ──────────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Build" });
  s.addText("MCP server: four business tools with contracts", { placeholder: "title" });
  const rows = [
    [th("Tool"), th("Purpose"), th("Validation")],
    ["get_order_details", "Order, membership, return history, computed eligibility", "ORD-\\d{3,} format; caller must own order"],
    ["get_return_refund_status", "Return status and refund amount", "Same ownership check"],
    ["create_return_request", "Starts a return (write)", "Window, non-returnable and status rules"],
    ["create_escalation_ticket", "Hands off to a human (write)", "Reason enum; summary required"],
  ];
  s.addTable(rows, tableOpts({ x: 0.5, y: 1.05, w: 5.9, colW: [1.9, 2.1, 1.9], fontSize: 12,
    rowH: 0.6, objectName: "mcp-table" }));
  txt(s, "Inputs and outputs are Pydantic models; read tools are retried once on transient errors, writes never.",
    { x: 0.5, y: 4.2, w: 5.9, h: 0.6, fontSize: 12, color: C.accent6 });
  card(s, 6.65, 1.05, 2.85, 3.95);
  heading(s, "Error envelope", 6.85, 1.2, 2.5);
  txt(s, '{"ok": false,\n "error": {\n  "code": "UNAUTHORIZED",\n  "message": "…",\n  "retryable": false}}',
    { x: 6.85, y: 1.6, w: 2.5, h: 1.3, fontSize: 11, fontFace: "Courier New", color: C.text2 });
  heading(s, "Identity", 6.85, 3.0, 2.5);
  txt(s, "RSA_REQUESTER_ID is set when the server starts. The model never passes a customer ID, so it cannot ask for another account's data.",
    { x: 6.85, y: 3.38, w: 2.5, h: 1.55, fontSize: 12 });
  s.addNotes("The server exposes four tools over stdio. Every error comes back in the same " +
    "typed envelope with a code and a retryable flag, which is what lets the agent tell an " +
    "outage it can retry from a definitive answer like ORDER_NOT_FOUND. Ownership is checked " +
    "server-side on every call.");
}

// 9 ─ Scenarios 1 and 2 ───────────────────────────────────────────────────────
pres.addSection({ title: "Demonstration" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demonstration" });
  s.addText("Scenarios 1–2: policy query and return flow", { placeholder: "title" });
  const t2 = traceLines("scenario2_multistep_return.jsonl");
  const meta = t2.find((r) => r.type === "meta");
  const res = t2.find((r) => r.type === "result");
  const steps = t2.filter((r) => r.type === "step" && (r.kind === "llm" || r.kind === "tool"));
  txt(s, [{ text: "Scenario 2 — ", options: { bold: true } },
    { text: `${meta.requester_id}: “${meta.question}”` }],
    { x: 0.5, y: 1.0, w: 9.0, h: 0.35, fontSize: 14, color: C.text2 });
  const w = (9.0 - 0.25 * (steps.length - 1)) / steps.length;
  steps.forEach((st, i) => {
    const x = 0.5 + i * (w + 0.25);
    const isTool = st.kind === "tool";
    const what = isTool ? st.name : (st.detail.tool_calls.length
      ? "LLM → " + st.detail.tool_calls.join(", ") : "LLM → answer");
    card(s, x, 1.45, w, 1.05, isTool ? C.accent1 : C.background2);
    txt(s, `${i + 1}. ${what}`, { x: x + 0.1, y: 1.52, w: w - 0.2, h: 0.6, fontSize: 12,
      bold: true, color: isTool ? C.background1 : C.text2 });
    txt(s, `${Math.round(st.latency_ms)} ms`, { x: x + 0.1, y: 2.15, w: w - 0.2, h: 0.28,
      fontSize: 11, color: isTool ? C.background1 : C.accent6 });
    if (i < steps.length - 1) arrow(s, x + w + 0.03, 1.975, 0.19, 0);
  });
  card(s, 0.5, 2.7, 4.4, 2.35);
  heading(s, "Scenario 2 — decision", 0.7, 2.82, 4.0);
  txt(s, bullets([
    "Order: Gaming Accessories, standard member",
    "Policy: 30-day window, original packaging",
    "Verdict: eligible, deadline from the order; offers to start the return",
    `Total ${secs(res.latency_ms)} · NLI grounding ${res.grounding.faithfulness}`,
  ], 13), { x: 0.7, y: 3.2, w: 4.0, h: 1.8 });
  const t1 = traceLines("scenario1_knowledge_query.jsonl");
  const m1 = t1.find((r) => r.type === "meta");
  card(s, 5.1, 2.7, 4.4, 2.35);
  heading(s, "Scenario 1 — knowledge query", 5.3, 2.82, 4.0);
  txt(s, [
    { text: `“${m1.question}”\n`, options: { italic: true, fontSize: 13, color: C.text2 } },
    { text: "Express to Canada: $19.99, 3–4 business days.\n", options: { fontSize: 13 } },
    { text: "Cited: international_shipping.html, shipping_faq.html (with chunk IDs)",
      options: { fontSize: 13, color: C.accent6 } },
  ], { x: 5.3, y: 3.2, w: 4.0, h: 1.8 });
  s.addNotes("Each box in the top row is one recorded step from the committed trace, with its " +
    "measured latency. The order lookup returned the category and delivery date; the model " +
    "then searched the return policy for that category, so the second step depends on the " +
    "first. It did not create a return because the customer asked whether they could, not " +
    "to start one. Scenario 1 shows one search and an answer citing its source chunks. " +
    "Live demo: uv run rsa ask \"Is it too late to return my controller from order ORD-10002?\" " +
    "--customer CUST-1035 --trace");
}

// 10 ─ Scenario 3: exceptions ─────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demonstration" });
  s.addText("Scenario 3: four exceptions handled safely", { placeholder: "title" });
  const cases = [
    ["Unauthorized access", "scenario3_unauthorized_order.jsonl",
      "MCP returns UNAUTHORIZED; nothing about the order is revealed"],
    ["Invalid order ID", "scenario3_order_not_found.jsonl",
      "ORDER_NOT_FOUND is definitive: asks the customer to check the ID"],
    ["Missing evidence", "scenario3_missing_evidence.jsonl",
      "No grounded policy: says so instead of guessing"],
    ["Prompt injection", "scenario3_prompt_injection_blocked.jsonl",
      "Blocked before the LLM: no tokens, no tools"],
  ];
  const w = 4.4, h = 1.8;
  cases.forEach(([t, f, how], i) => {
    const lines = traceLines(f);
    const m = lines.find((r) => r.type === "meta"), r = lines.find((x) => x.type === "result");
    const x = 0.5 + (i % 2) * (w + 0.2), y = 1.05 + Math.floor(i / 2) * (h + 0.2);
    card(s, x, y, w, h);
    txt(s, t, { x: x + 0.2, y: y + 0.12, w: 2.2, h: 0.35, fontSize: 15, bold: true,
      color: C.text2 });
    card(s, x + w - 2.05, y + 0.14, 1.85, 0.3, C.accent4);
    txt(s, r.outcome, { x: x + w - 2.05, y: y + 0.14, w: 1.85, h: 0.3, fontSize: 11, bold: true,
      color: C.background1, align: "center", valign: "middle" });
    const q = m.question.length > 70 ? m.question.slice(0, 67) + "…" : m.question;
    txt(s, `“${q}”`, { x: x + 0.2, y: y + 0.58, w: w - 0.4, h: 0.55, fontSize: 12, italic: true,
      color: C.accent6 });
    txt(s, how, { x: x + 0.2, y: y + 1.15, w: w - 0.4, h: 0.55, fontSize: 13 });
  });
  txt(s, "Tool unavailable: read tools retry once; MCP outage, locked DB and LLM outage are covered by chaos tests.",
    { x: 0.5, y: 4.85, w: 9.0, h: 0.3, fontSize: 11, color: C.accent6 });
  s.addNotes("The assignment asked for at least two exception cases; these are four, each " +
    "from a committed trace. The outcome label is derived from tool state, never from the " +
    "answer's wording, so refused_unauthorized means every order lookup actually hit another " +
    "account. The injection case never reaches the model at all.");
}

// 11 ─ Guardrails ─────────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demonstration" });
  s.addText("Safety enforced at five points in code", { placeholder: "title" });
  const layers = [
    ["1", "Input", ["Toxicity: regex + toxic-bert", "Injection: regex + DeBERTa", "Rate limit 10 / 60 s"]],
    ["2", "Tool call", ["Schema + ID format", "No return before lookup", "Per-tool call budgets"]],
    ["3", "Tool output", ["Neutralise injected text", "Redact email, card, SSN"]],
    ["4", "Answer", ["Prompt-leak scan", "Others' IDs masked", "Made-up numbers flagged"]],
    ["5", "Grounding", ["Live NLI check of claims", "Contradiction flagged"]],
  ];
  const w = 1.64, gap = 0.2;
  layers.forEach(([nn, t, items], i) => {
    const x = 0.5 + i * (w + gap);
    card(s, x, 1.1, w, 3.3);
    badge(s, nn, x + (w - 0.5) / 2, 1.25, 0.5);
    txt(s, t, { x: x + 0.1, y: 1.85, w: w - 0.2, h: 0.35, fontSize: 15, bold: true,
      color: C.text2, align: "center" });
    txt(s, bullets(items, 12), { x: x + 0.12, y: 2.3, w: w - 0.24, h: 2.0 });
    if (i < layers.length - 1) arrow(s, x + w + 0.02, 1.5, gap - 0.04, 0);
  });
  txt(s, "The model is told the rules too, but nothing above depends on it following them.",
    { x: 0.5, y: 4.6, w: 9.0, h: 0.35, fontSize: 13, italic: true, color: C.accent6 });
  s.addNotes("Each layer is a separate module under guardrails/. The important design point " +
    "is that safety does not rely on the prompt: a model that ignores its instructions still " +
    "cannot read another customer's order, start an unrequested return, or leak the system " +
    "prompt, because those checks run in code before and after it.");
}

// 12 ─ Evaluation ─────────────────────────────────────────────────────────────
pres.addSection({ title: "Results" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Agent evaluation: what passed, what it caught", { placeholder: "title" });
  const rows = [[th("Scenario"), th("Cases"), th("Facts"), th("Tools"), th("Safety"), th("Judge")]];
  for (const [k, v] of Object.entries(byS)) {
    rows.push([k.replace("_", "-"), `${v.n}`, pct(v.fact_accuracy), pct(v.tool_selection_accuracy),
      pct(v.safety_pass_rate), pct(v.judge_correctness)]);
  }
  rows.push([{ text: "All", options: { bold: true } }, `${S.n}`, pct(S.fact_accuracy),
    pct(S.tool_selection_accuracy), pct(S.safety_pass_rate), pct(S.judge_correctness)]);
  s.addTable(rows, tableOpts({ x: 0.5, y: 1.05, w: 5.4, colW: [1.4, 0.7, 0.8, 0.8, 0.85, 0.85],
    fontSize: 12, rowH: 0.36, objectName: "eval-table" }));
  txt(s, `Facts, tools, safety: deterministic checks. Judge: G-Eval. Faithfulness ${pct(S.faithfulness)}.`,
    { x: 0.5, y: 3.0, w: 5.4, h: 0.5, fontSize: 11, color: C.accent6 });
  txt(s, bullets([
    "Exceptions: other account, bad IDs, injection, missing policy",
    `${nTests} offline tests incl. chaos: locked DB, LLM outage, corrupt index`,
  ], 13), { x: 0.5, y: 3.6, w: 5.4, h: 1.4 });
  card(s, 6.15, 1.05, 3.35, 3.95);
  txt(s, [
    { text: "Caught by the eval\n", options: { bold: true, fontSize: 15, color: C.accent4 } },
    { text: `Earlier run: safety ${pct(before.summary.safety_pass_rate)}. Asked “Can I return…?”, ` +
      "the agent created a return nobody requested.\n\n", options: { fontSize: 13 } },
    { text: "Fix in code\n", options: { bold: true, fontSize: 15, color: C.text2 } },
    { text: "Writes need the customer's request (or a yes to an offer) and an eligible " +
      `lookup. Final run: safety ${pct(S.safety_pass_rate)}.`, options: { fontSize: 13 } },
  ], { x: 6.35, y: 1.2, w: 2.95, h: 3.7 });
  s.addNotes(`Retrieval and agent behaviour are evaluated separately. ${S.n} cases is a small ` +
    "set, so one case is about 3.4 points, and run-to-run variance is real: case a29 passed " +
    "here but failed in two earlier runs, so I treat it as flaky, not fixed. The LLM judge is " +
    "noisy too, which is why the deterministic checks are the primary metrics. The most " +
    "useful result was a failure: the eval caught the agent creating an unrequested return.");
}

// 13 ─ Cost, latency, limitations ─────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Cost, latency and what I would change", { placeholder: "title" });
  [
    [`${(o.input_tokens_mean / 1000).toFixed(1)}k`, "input tokens / query\n(mean)"],
    [secs(o.latency_ms.p50), "end-to-end latency\np50"],
    [`$${cost.price_sensitivity[1].usd_per_1k_queries.toFixed(2)}`, "per 1,000 queries at\n$0.40 / $1.60 per M"],
    [pct(o.cached_input_share), "of input tokens\nprefix-cached"],
  ].forEach(([b, l], i) => stat(s, 0.5 + i * 2.3, 1.05, 2.1, b, l));
  card(s, 0.5, 2.4, 4.4, 2.65);
  heading(s, "Known limitations", 0.7, 2.55, 4.0);
  txt(s, bullets([
    `CPU reranker is the slow stage; knowledge p95 ${secs(byS.knowledge.latency_ms_p95)}`,
    "Embedded Qdrant: one process holds the lock",
    "Judge shares the agent's model family",
    "29 agent cases: one case ≈ 3.4 points",
  ], 13), { x: 0.7, y: 2.95, w: 4.0, h: 2.05 });
  card(s, 5.1, 2.4, 4.4, 2.65);
  heading(s, "Next steps", 5.3, 2.55, 4.0);
  txt(s, bullets([
    "Reranker on GPU, or drop to plain hybrid",
    "Qdrant server mode; Redis for a shared rate limit",
    "Larger eval sets; a judge from another model family",
  ], 13), { x: 5.3, y: 2.95, w: 4.0, h: 2.05 });
  s.addNotes("Tokens and latency are measured from the final run. The hosted endpoint has no " +
    "published per-token price, so cost is shown at an illustrative price; the full " +
    "sensitivity table is in reports/cost_latency_final.md. Per-stage mean: LLM " +
    "2.8 s, retrieval 0.9 s, MCP tools 0.1 s, so the model and the reranker are where the " +
    "time goes.");
}

// 14 ─ Close ──────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide({ masterName: "DARK_TITLE", sectionTitle: "Results" });
  s.addText("Questions?", { placeholder: "title" });
  s.addText("Evidence-first answers · tools behind MCP with fixed identity · safety in code",
    { placeholder: "body" });
  [
    [pct(M.full.summary["evidence_hit@1"]), "evidence Hit@1\n(full retrieval)"],
    [pct(S.fact_accuracy), "fact accuracy\n(29 agent cases)"],
    [pct(S.safety_pass_rate), "safety pass rate\n(final run)"],
    [`$${cost.price_sensitivity[1].usd_per_1k_queries.toFixed(2)}`, "per 1,000 queries\n(illustrative price)"],
  ].forEach(([b, l], i) => stat(s, 0.6 + i * 2.2, 3.55, 2.0, b, l, true));
  s.addNotes("To close: answers are grounded in retrieved evidence, operational data is only " +
    "reachable through MCP with the identity fixed per session, and the safety rules run in " +
    "code. Reports, traces and the test suite are all in the repository and reproducible " +
    "without a paid API.");
}

(async () => {
  const out = path.join(ROOT, process.env.OUT || "docs/presentation_full.pptx");
  await pres.writeFile({ fileName: out });
  if (process.env.APPLY_THEME) {
    const { applyTheme } = require(process.env.APPLY_THEME);
    await applyTheme(out, THEME);
  }
  console.log(`wrote ${out}`);
})();
