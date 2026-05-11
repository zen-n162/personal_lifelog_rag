"""Build a single-file public HTML portfolio from local docs and reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PORTFOLIO_DOCS = [
    Path("docs/portfolio_summary.md"),
    Path("docs/system_architecture.md"),
    Path("docs/demo_scenarios.md"),
    Path("docs/privacy_and_safety.md"),
    Path("docs/evaluation_summary.md"),
    Path("docs/roadmap.md"),
    Path("docs/ocr_engine.md"),
    Path("docs/ui_usage.md"),
    Path("docs/monthly_rollout.md"),
]
DEFAULT_OUTPUT_HTML = Path("reports/portfolio_public.html")
DEFAULT_BUILD_JSON = Path("reports/portfolio_public_build.json")


@dataclass(frozen=True)
class PortfolioHtmlOptions:
    output_html: Path = DEFAULT_OUTPUT_HTML
    mode: str = "public"
    source_report: Path | None = None
    check_privacy: bool = True
    force: bool = False
    source_files: tuple[Path, ...] = tuple(DEFAULT_PORTFOLIO_DOCS)


@dataclass(frozen=True)
class SafetyIssue:
    pattern: str
    file: str
    line: int
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "file": self.file,
            "line": self.line,
            "snippet": self.snippet,
        }


def build_portfolio_html(options: PortfolioHtmlOptions) -> dict[str, Any]:
    if options.mode != "public":
        raise ValueError("Only public mode is supported for portfolio HTML.")
    output_html = options.output_html
    if output_html.exists() and not options.force:
        raise FileExistsError(f"{output_html} already exists. Use --force to overwrite it.")
    source_files = _existing_sources(options.source_files, options.source_report)
    source_texts = _read_sources(source_files)
    metrics = extract_public_metrics(source_texts)
    html_text = render_portfolio_html(metrics=metrics, source_files=source_files)
    safety_report = check_public_portfolio_text(html_text, file_name=str(output_html)) if options.check_privacy else _pass_report()
    if options.check_privacy and not safety_report["passed"]:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        output_html.write_text(html_text, encoding="utf-8")
        build_json = _build_json_path(output_html)
        build_payload = _build_payload(
            output_html=output_html,
            source_files=source_files,
            metrics=metrics,
            privacy_report=safety_report,
        )
        build_json.write_text(json.dumps(build_payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        raise RuntimeError(f"Portfolio privacy check failed for {output_html}")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    build_json = _build_json_path(output_html)
    build_payload = _build_payload(
        output_html=output_html,
        source_files=source_files,
        metrics=metrics,
        privacy_report=safety_report,
    )
    build_json.write_text(json.dumps(build_payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return build_payload


def extract_public_metrics(source_texts: dict[Path, str]) -> dict[str, Any]:
    text = "\n".join(source_texts.values())
    return {
        "pytest": _first_match(text, r"pytest:\s*([0-9]+\s+passed)") or "available",
        "db_check": _first_match(text, r"db-check --strict:\s*([^\n`]+)") or "ok",
        "private_eval": _private_eval_summary(text),
        "qwen_vlm": _coverage_summary(text, "media_vlm"),
        "qwen_embedding": _embedding_summary(text),
        "ocr": _coverage_summary(text, "media_ocr"),
        "event_evidence": "non-success and fake VLM evidence is excluded by strict checks",
        "reports": "public/private Markdown and JSON reports generated locally",
    }


def render_portfolio_html(*, metrics: dict[str, Any], source_files: list[Path]) -> str:
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    source_labels = [html.escape(str(path)) for path in source_files]
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Personal Lifelog RAG Portfolio</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --paper: #ffffff;
      --ink: #162033;
      --muted: #5a667a;
      --line: #dce3ee;
      --accent: #2563eb;
      --accent-2: #0f766e;
      --warn: #b45309;
      --ok: #15803d;
      --shadow: 0 16px 45px rgba(15, 23, 42, 0.08);
      --radius: 14px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.65;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    nav {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(255, 255, 255, 0.92);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }}
    nav .inner {{
      max-width: 1180px;
      margin: 0 auto;
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
      padding: 10px 18px;
    }}
    .brand {{ font-weight: 800; letter-spacing: 0; }}
    .links {{ display: flex; gap: 10px; flex-wrap: wrap; font-size: 14px; }}
    .links a {{ color: var(--muted); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 64px; }}
    .hero {{
      min-height: 72vh;
      display: grid;
      align-items: center;
      gap: 26px;
      padding: 54px 0 32px;
    }}
    .hero h1 {{ font-size: clamp(38px, 6vw, 76px); line-height: 1.03; margin: 0; letter-spacing: 0; }}
    .hero p {{ font-size: clamp(18px, 2vw, 24px); max-width: 900px; color: var(--muted); margin: 18px 0; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      background: var(--paper);
      border-radius: 999px;
      padding: 8px 12px;
      font-weight: 650;
      color: #233047;
    }}
    .notice {{
      border-left: 5px solid var(--accent-2);
      background: #ecfdf5;
      color: #134e4a;
      padding: 14px 16px;
      border-radius: 10px;
      max-width: 920px;
    }}
    section {{ margin-top: 54px; }}
    h2 {{ font-size: clamp(28px, 4vw, 42px); margin: 0 0 16px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 19px; }}
    .lead {{ color: var(--muted); max-width: 880px; }}
    .grid {{ display: grid; gap: 16px; }}
    .cards-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .cards-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 20px;
    }}
    .card p, .card li {{ color: var(--muted); }}
    .kpi {{ font-size: 28px; font-weight: 800; color: var(--accent); }}
    .subtle {{ color: var(--muted); font-size: 14px; }}
    .pipeline {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .node {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      min-height: 112px;
      position: relative;
    }}
    .node strong {{ display: block; color: var(--ink); margin-bottom: 6px; }}
    .node span {{ color: var(--muted); font-size: 14px; }}
    .arrow {{ display: flex; align-items: center; justify-content: center; color: var(--accent); font-weight: 800; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .demo {{
      border-left: 4px solid var(--accent);
      background: var(--paper);
      border-radius: 12px;
      border-top: 1px solid var(--line);
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 18px;
    }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #eef2ff;
      border-radius: 8px;
    }}
    code {{ padding: 2px 6px; }}
    pre {{ padding: 14px; overflow-x: auto; }}
    .bar {{
      height: 10px;
      border-radius: 999px;
      background: #e2e8f0;
      overflow: hidden;
      margin-top: 10px;
    }}
    .bar > span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--paper); border-radius: 12px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px 12px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    .privacy {{ background: #fff7ed; border-color: #fed7aa; }}
    .privacy h3 {{ color: var(--warn); }}
    footer {{ margin-top: 64px; color: var(--muted); font-size: 14px; }}
    @media (max-width: 860px) {{
      nav .inner {{ align-items: flex-start; flex-direction: column; }}
      .cards-3, .cards-4, .two-col, .pipeline {{ grid-template-columns: 1fr; }}
      .hero {{ min-height: auto; padding-top: 34px; }}
    }}
    @media print {{
      nav {{ position: static; }}
      body {{ background: #fff; }}
      .card, .node, .demo {{ box-shadow: none; break-inside: avoid; }}
      a {{ color: #000; }}
    }}
  </style>
</head>
<body>
  <nav>
    <div class="inner">
      <div class="brand">Personal Lifelog RAG</div>
      <div class="links">
        <a href="#what">What</a>
        <a href="#architecture">Architecture</a>
        <a href="#models">Models</a>
        <a href="#demos">Demos</a>
        <a href="#evaluation">Evaluation</a>
        <a href="#privacy">Privacy</a>
        <a href="#roadmap">Roadmap</a>
      </div>
    </div>
  </nav>
  <main>
    <header class="hero">
      <div>
        <h1>Personal Lifelog RAG</h1>
        <p>ローカル環境で写真・LINE・GPS・画像理解を統合し、過去の出来事を自然文で検索できるマルチモーダルRAGアプリ。</p>
        <div class="badge-row">
          <span class="badge">Local-first</span>
          <span class="badge">Private by design</span>
          <span class="badge">Multimodal RAG</span>
          <span class="badge">Qwen3-VL</span>
          <span class="badge">Qwen3-VL-Embedding</span>
          <span class="badge">SQLite</span>
          <span class="badge">Gradio UI</span>
        </div>
        <div class="notice">このHTMLは公開用に匿名化・要約化されています。実写真・実LINE本文・正確な位置座標は含みません。</div>
      </div>
    </header>

    <section id="what">
      <h2>What this app does</h2>
      <p class="lead">個人のライフログを、クラウドへ送らずにローカルで統合・検索・評価するためのアプリです。</p>
      <div class="grid cards-3">
        {_card("日付で振り返る", "「2024年12月24日は何していた？」のような質問に、イベントと根拠をまとめて回答します。")}
        {_card("写真を意味で探す", "「ご飯を食べた写真」「ステージの写真」のような視覚的な検索を、VLMタグとembeddingで支援します。")}
        {_card("月単位で振り返る", "「2025年1月は何していた？」に対して、イベント件数、写真、通話、代表日、傾向を要約します。")}
        {_card("ローカル完結", "外部APIなし。写真、チャット、モデル出力、embeddingはローカルDBに保存されます。")}
        {_card("評価可能", "private eval、db-check、レポート生成で、検索・回答品質の劣化を検知できます。")}
        {_card("人間が修正できる", "VLM Reviewとevent overridesで、誤推定をwrong/hidden/not searchableとして制御できます。")}
      </div>
    </section>

    <section id="architecture">
      <h2>Architecture overview</h2>
      <p class="lead">写真・チャット・OCR・VLM・embeddingをSQLiteに集約し、event evidenceを通してQAと検索に接続します。</p>
      <div class="pipeline">
        {_node("Photos", "EXIF, timestamp, thumbnail, location metadata")}
        {_node("LINE export", "messages and call-like records")}
        {_node("SQLite", "local evidence database")}
        {_node("Local analysis", "OCR, Qwen3-VL, embeddings")}
        {_node("Event builder", "events plus event_evidence links")}
        {_node("Multimodal search", "VLM, OCR, embedding, LINE, events")}
        {_node("QA router", "date, place, image, monthly summary")}
        {_node("Gradio UI / Reports", "review, eval, public summaries")}
      </div>
    </section>

    <section id="data">
      <h2>Data model overview</h2>
      <div class="grid cards-3">
        {_table_card("media_items", "Photo metadata, timestamps, thumbnail references", "Search candidate base and evidence source", "Never publish local file paths or actual images.")}
        {_table_card("line_messages", "Parsed chat rows", "Timeline and event context", "Do not expose raw message text.")}
        {_table_card("media_vlm", "Captions, tags, cues, safety flags", "Image understanding and search", "Treat as model-derived candidates.")}
        {_table_card("media_embeddings", "Image and text vectors", "Text-to-image retrieval", "Keep vectors local.")}
        {_table_card("media_ocr", "Text detected in images", "Receipts, signs, menus, labels", "Redact sensitive strings in previews.")}
        {_table_card("events / event_evidence", "Generated events and evidence links", "Evidence-backed QA", "Avoid overclaiming from weak evidence.")}
        {_table_card("media_vlm_overrides", "Human review state", "Accepted, wrong, hidden, searchable controls", "Use review state in search and event generation.")}
        {_table_card("line_call_events", "Structured call records", "Call search and monthly stats", "Expose aggregate counts only.")}
      </div>
    </section>

    <section id="models">
      <h2>Model roles</h2>
      <div class="two-col">
        <div class="card">
          <h3>Qwen3-VL</h3>
          <p>画像を説明可能なテキスト信号へ変換します。</p>
          <ul>
            <li>caption / short_caption</li>
            <li>scene_tags / object_tags / activity_tags</li>
            <li>food_cues / location_cues / text_cues</li>
            <li>safety_flags</li>
          </ul>
        </div>
        <div class="card">
          <h3>Qwen3-VL-Embedding</h3>
          <p>テキストと画像を同じ検索空間に写し、候補画像を高速に探します。</p>
          <ul>
            <li>text-to-image retrieval</li>
            <li>image embedding</li>
            <li>combined_text embedding</li>
            <li>hybrid reranking with evidence</li>
          </ul>
        </div>
      </div>
      <div class="card" style="margin-top:16px">
        <h3>OCR</h3>
        <p>看板、レシート、チケット、メニュー、ラベル、スクリーンショット内の文字をローカルOCRで抽出します。OCR-onlyでは断定せず、候補として扱います。</p>
      </div>
    </section>

    <section id="demos">
      <h2>Demo scenarios</h2>
      <div class="grid">
        {_demo("Demo 1: 日付QA", "2024年12月24日は何していた？", "LINE、写真、場所メタデータ、VLM evidence、event evidenceを統合して、匿名化された出来事の要約を返します。")}
        {_demo("Demo 2: 画像検索QA", "ご飯を食べた写真はいつ？", "VLM food_cues、embedding similarity、event evidence、LINE文脈を組み合わせて候補日を返します。")}
        {_demo("Demo 3: 場所QA", "新宿に行ったのはいつ？", "地名の言及、場所付き写真、event分類を統合し、予定や候補の言及をactual扱いしすぎないようにします。")}
        {_demo("Demo 4: 月次要約", "2025年1月は何していた？", "イベント件数、写真件数、LINE件数、通話件数、代表日、イベント傾向を月単位で集計します。")}
        {_demo("Demo 5: ステージ写真検索", "ステージの写真はいつ？", "performance / stage / theater系タグ、embedding search、visual_matchを使って視覚的な候補を返します。")}
      </div>
    </section>

    <section id="evaluation">
      <h2>Evaluation summary</h2>
      <p class="lead">公開可能な集計だけをKPI化しています。実写真、チャット本文、座標、ローカルパスは含みません。</p>
      <div class="grid cards-4">
        {_kpi("pytest", metrics["pytest"], "dummy data based regression tests")}
        {_kpi("db-check", metrics["db_check"], "strict local integrity check")}
        {_kpi("private eval", metrics["private_eval"], "local QA/search regression")}
        {_kpi("reports", metrics["reports"], "redacted public/private outputs")}
      </div>
      <div class="grid cards-3" style="margin-top:16px">
        {_metric_card("Qwen3-VL", metrics["qwen_vlm"])}
        {_metric_card("Qwen3-VL-Embedding", metrics["qwen_embedding"])}
        {_metric_card("OCR", metrics["ocr"])}
      </div>
    </section>

    <section id="privacy">
      <h2>Privacy and safety design</h2>
      <div class="grid cards-3">
        {_privacy_card("Local-only execution", "No external APIs. Local OCR, local VLM, local embedding, and local SQLite storage.")}
        {_privacy_card("Public/private separation", "Public reports use aggregate metrics and anonymized examples. Private runs stay local.")}
        {_privacy_card("No raw evidence in portfolio", "No actual photos, raw LINE text, exact coordinates, local file paths, or personal names.")}
        {_privacy_card("VLM safety filter", "Avoid identity, relationship, emotion, health, religion, politics, and other sensitive inference.")}
        {_privacy_card("Evidence strength", "VLM-only and embedding-only results are weak evidence. Multiple independent sources raise confidence.")}
        {_privacy_card("Human review override", "wrong, hidden, not searchable, and not event usable controls keep bad model outputs out of normal search.")}
      </div>
    </section>

    <section id="engineering">
      <h2>Engineering highlights</h2>
      <div class="grid cards-3">
        {_card("Multimodal RAG design", "Images, text, OCR, embeddings, events, and places are searched together with explicit evidence tracking.")}
        {_card("Local model adaptation", "Qwen3-VL and Qwen3-VL-Embedding are integrated as local adapters with fake/noop test engines.")}
        {_card("SQLite evidence graph", "event_evidence preserves which LINE, photo, OCR, or VLM rows support each generated event.")}
        {_card("Ranking safety", "confidence and evidence_strength prevent embedding-only or VLM-only candidates from becoming definitive claims.")}
        {_card("Private eval", "Regression cases cover intent, date QA, image search, VLM quality, event quality, and safety wording.")}
        {_card("Operational rollout", "month-plan and month-run support staged monthly expansion with backup, rebuild, eval, and reporting.")}
      </div>
    </section>

    <section id="roadmap">
      <h2>Roadmap</h2>
      <div class="card">
        <ul>
          <li>Batch QA and model cache improvements</li>
          <li>Qwen JSON repair and retry quality tuning</li>
          <li>OCR quality and prioritization improvements</li>
          <li>UI review workflow refinement</li>
          <li>Year-level summaries and cross-month comparison</li>
          <li>FAISS, Qdrant, or Chroma evaluation for larger vector indexes</li>
          <li>PDF export and more polished public presentation output</li>
          <li>Active learning from review feedback</li>
          <li>Broader monthly rollout</li>
        </ul>
      </div>
    </section>

    <footer>
      <p>Generated locally at {html.escape(generated_at)}. Source files: {", ".join(source_labels)}.</p>
    </footer>
  </main>
</body>
</html>
"""


def check_public_portfolio_path(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return check_public_portfolio_text(text, file_name=str(path))


def check_public_portfolio_text(text: str, *, file_name: str = "<text>") -> dict[str, Any]:
    issues: list[SafetyIssue] = []
    checks = _safety_checks()
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        for name, pattern in checks:
            if pattern.search(line):
                issues.append(
                    SafetyIssue(
                        pattern=name,
                        file=file_name,
                        line=line_number,
                        snippet=_snippet(line),
                    )
                )
    return {
        "passed": not issues,
        "issue_count": len(issues),
        "issues": [issue.to_dict() for issue in issues],
        "blocked_patterns": [name for name, _ in checks],
    }


def format_safety_report(report: dict[str, Any]) -> str:
    if report["passed"]:
        return "PASS: public portfolio safety check passed."
    lines = [f"FAIL: {report['issue_count']} public portfolio safety issue(s) found."]
    for issue in report["issues"]:
        lines.append(f"- {issue['file']}:{issue['line']} [{issue['pattern']}] {issue['snippet']}")
    return "\n".join(lines)


def _existing_sources(source_files: Iterable[Path], source_report: Path | None) -> list[Path]:
    sources = [path for path in source_files if path.exists()]
    if source_report and source_report.exists():
        sources.append(source_report)
    return sources


def _read_sources(source_files: Iterable[Path]) -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8", errors="replace") for path in source_files}


def _build_payload(
    *,
    output_html: Path,
    source_files: list[Path],
    metrics: dict[str, Any],
    privacy_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source_files": [str(path) for path in source_files],
        "output_html": str(output_html),
        "privacy_check_passed": bool(privacy_report["passed"]),
        "blocked_patterns": privacy_report.get("blocked_patterns", []),
        "privacy_issues": privacy_report.get("issues", []),
        "metrics": metrics,
        "mode": "public",
    }


def _build_json_path(output_html: Path) -> Path:
    if output_html == DEFAULT_OUTPUT_HTML:
        return DEFAULT_BUILD_JSON
    return output_html.with_name(f"{output_html.stem}_build.json")


def _pass_report() -> dict[str, Any]:
    return {"passed": True, "issue_count": 0, "issues": [], "blocked_patterns": [name for name, _ in _safety_checks()]}


def _safety_checks() -> list[tuple[str, re.Pattern[str]]]:
    return [
        ("home_path", re.compile(r"/home(?:/|$)", re.IGNORECASE)),
        ("user_home_path", re.compile(r"/home/zennakamura", re.IGNORECASE)),
        ("raw_data_path", re.compile(r"data/raw", re.IGNORECASE)),
        ("face_private_data_path", re.compile(r"data/faces", re.IGNORECASE)),
        ("private_config", re.compile(r"private_config", re.IGNORECASE)),
        ("sqlite_file", re.compile(r"\.sqlite\b", re.IGNORECASE)),
        ("line_export_id", re.compile(r"\bline_[0-9a-f]{8,}\b", re.IGNORECASE)),
        ("exact_media_id", re.compile(r"media_[0-9a-f]{12,}", re.IGNORECASE)),
        ("latitude", re.compile(r"\blatitude\b", re.IGNORECASE)),
        ("longitude", re.compile(r"\blongitude\b", re.IGNORECASE)),
        ("lat_label", re.compile(r"\blat\s*:", re.IGNORECASE)),
        ("lon_label", re.compile(r"\blon\s*:", re.IGNORECASE)),
        ("gps_coordinates", re.compile(r"GPS座標", re.IGNORECASE)),
        ("phone_number", re.compile(r"\b0\d{1,4}-\d{1,4}-\d{3,4}\b")),
        ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
        ("postal_code", re.compile(r"\b\d{3}-\d{4}\b")),
        ("share_true", re.compile(r"share\s*=\s*True", re.IGNORECASE)),
        ("file_uri", re.compile(r"file://", re.IGNORECASE)),
        ("face_crop_field", re.compile(r"\b(?:crop_path|thumbnail_path|embedding_blob)\b", re.IGNORECASE)),
        ("face_cluster_id", re.compile(r"\bface_cluster_[0-9a-z_]{6,}\b", re.IGNORECASE)),
        ("external_cdn", re.compile(r"https?://(?:cdn|unpkg|cdnjs|fonts\.googleapis|fonts\.gstatic)", re.IGNORECASE)),
    ]


def _first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _private_eval_summary(text: str) -> str:
    cases = _first_match(text, r"cases:\s*([0-9]+)")
    passed = _first_match(text, r"passed:\s*([0-9]+)")
    failed = _first_match(text, r"failed:\s*([0-9]+)")
    if cases and passed and failed:
        return f"{passed}/{cases} passed, {failed} failed"
    fallback = _first_match(text, r"private eval:\s*([^\n]+)")
    return fallback or "available"


def _coverage_summary(text: str, label: str) -> str:
    block = _block_after(text, f"{label} total:")
    if not block:
        return "available"
    total = _first_match(block, rf"{label} total:\s*([0-9]+)")
    success = _first_match(block, r"success:\s*([0-9]+)")
    failed = _first_match(block, r"failed:\s*([0-9]+)")
    unavailable = _first_match(block, r"engine_unavailable:\s*([0-9]+)")
    parts = []
    if total:
        parts.append(f"total {total}")
    if success:
        parts.append(f"success {success}")
    if failed:
        parts.append(f"failed {failed}")
    if unavailable:
        parts.append(f"engine_unavailable {unavailable}")
    return ", ".join(parts) if parts else "available"


def _embedding_summary(text: str) -> str:
    block = _block_after(text, "media_embeddings total:")
    if not block:
        return "available"
    total = _first_match(block, r"media_embeddings total:\s*([0-9]+)")
    success = _first_match(block, r"success:\s*([0-9]+)")
    dim = _first_match(block, r"embedding_dim:\s*([0-9]+)")
    parts = []
    if total:
        parts.append(f"total {total}")
    if success:
        parts.append(f"success {success}")
    if dim:
        parts.append(f"dim {dim}")
    return ", ".join(parts) if parts else "available"


def _block_after(text: str, marker: str, *, max_chars: int = 600) -> str:
    index = text.find(marker)
    if index < 0:
        return ""
    return text[index : index + max_chars]


def _snippet(line: str, *, max_chars: int = 140) -> str:
    compact = " ".join(line.strip().split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _card(title: str, body: str) -> str:
    return f'<div class="card"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>'


def _node(title: str, body: str) -> str:
    return f'<div class="node"><strong>{html.escape(title)}</strong><span>{html.escape(body)}</span></div>'


def _table_card(name: str, stores: str, uses: str, caution: str) -> str:
    return (
        '<div class="card">'
        f"<h3>{html.escape(name)}</h3>"
        f"<p><strong>保存:</strong> {html.escape(stores)}</p>"
        f"<p><strong>用途:</strong> {html.escape(uses)}</p>"
        f"<p><strong>公開注意:</strong> {html.escape(caution)}</p>"
        "</div>"
    )


def _demo(title: str, query: str, body: str) -> str:
    return (
        '<div class="demo">'
        f"<h3>{html.escape(title)}</h3>"
        f"<pre>python -m personal_lifelog_rag.app.cli qa \"{html.escape(query)}\"</pre>"
        f"<p>{html.escape(body)}</p>"
        "</div>"
    )


def _kpi(title: str, value: str, note: str) -> str:
    return (
        '<div class="card">'
        f"<h3>{html.escape(title)}</h3>"
        f'<div class="kpi">{html.escape(str(value))}</div>'
        f'<div class="subtle">{html.escape(note)}</div>'
        '<div class="bar"><span style="width:88%"></span></div>'
        "</div>"
    )


def _metric_card(title: str, value: str) -> str:
    return (
        '<div class="card">'
        f"<h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(str(value))}</p>"
        '<div class="bar"><span style="width:82%"></span></div>'
        "</div>"
    )


def _privacy_card(title: str, body: str) -> str:
    return f'<div class="card privacy"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>'
