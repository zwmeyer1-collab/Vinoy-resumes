#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
import argparse
from pathlib import Path
from typing import List

import extract_latest_resumes as m

DOCS_DIR = Path(__file__).parent / "docs"
INDEX_HTML = DOCS_DIR / "index.html"


def build_rows(pdf: Path, fast: bool) -> List[m.EventRow]:
    if fast:
        # Skip OCR for faster builds; may reduce accuracy.
        m.ocr_search_over_event = lambda *_args, **_kwargs: ""  # type: ignore
        m.ocr_crop = lambda *_args, **_kwargs: ""  # type: ignore
    pages_text = m.read_pdf_pages_text(pdf)
    events = m.group_pages_into_events(pdf, pages_text, debug=False)

    rows: List[m.EventRow] = []
    for ev in events:
        org = m.first_group(m.RE_ORG, ev.text) or ""
        name = m.first_group(m.RE_NAME, ev.text) or m.first_group(m.RE_NAME2, ev.text) or ""
        label = m.normalize_event_label(org, name)
        if label == "—":
            label = m.fallback_event_label(ev.text) or "—"

        arr, dep = m.extract_dates(ev.text, pdf, ev.pages)
        drive, fly = m.extract_drive_fly(ev.text, pdf, ev.pages)
        valet = m.extract_valet(ev.text, pdf, ev.pages)
        billing = m.extract_billing(ev.text, pdf, ev.pages)

        if label == "—" and arr == "—" and dep == "—" and drive == "—" and fly == "—" and valet == "—" and billing == "—":
            continue
        rows.append(m.EventRow(label, arr, dep, drive, fly, valet, billing))
    return rows


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_table(rows: List[m.EventRow]) -> str:
    headers = [
        "Event / Group",
        "Main Arrival",
        "Main Departure",
        "Drive-In %",
        "Fly-In %",
        "Valet Pricing",
        "Billing",
    ]

    def td(val: str) -> str:
        return f"<td>{html_escape(val)}</td>"

    body_rows = []
    for r in rows:
        body_rows.append(
            "<tr>"
            + td(r.event_group)
            + td(r.main_arrival)
            + td(r.main_departure)
            + td(r.drive_in)
            + td(r.fly_in)
            + td(r.valet_pricing)
            + td(r.billing)
            + "</tr>"
        )

    head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    body = "\n".join(body_rows)
    return f"<table><thead>{head}</thead><tbody>{body}</tbody></table>"


def build_html(pdf: Path, rows: List[m.EventRow], fast: bool) -> str:
    updated = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = "FAST (no OCR)" if fast else "FULL"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Event Resume Summary</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f4ef;
      --ink: #1b1b1b;
      --muted: #6b6b6b;
      --accent: #1e3a8a;
      --card: #ffffff;
      --rule: #e5e1d8;
    }}
    body {{
      margin: 0;
      font-family: "Georgia", "Times New Roman", serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1100px;
      margin: 24px auto 64px;
      padding: 0 16px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 6px;
      letter-spacing: 0.3px;
    }}
    .meta {{
      font-size: 14px;
      color: var(--muted);
      margin-bottom: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--rule);
      border-radius: 10px;
      padding: 8px 10px;
      box-shadow: 0 1px 0 rgba(0,0,0,0.04);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .table-wrap {{
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }}
    .table-wrap table {{
      min-width: 820px;
    }}
    thead th {{
      text-align: left;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--muted);
      padding: 10px 8px;
      border-bottom: 1px solid var(--rule);
      position: sticky;
      top: 0;
      background: var(--card);
    }}
    tbody td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--rule);
      vertical-align: top;
    }}
    tbody tr:hover {{
      background: #faf8f4;
    }}
    @media (max-width: 720px) {{
      h1 {{ font-size: 22px; }}
      table {{ font-size: 13px; }}
      thead th {{ font-size: 11px; }}
      .wrap {{ margin-top: 16px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Event Resume Summary</h1>
    <div class="meta">PDF: {html_escape(pdf.name)} · Updated {updated} · Mode: {mode}</div>
    <div class="card">
      <div class="table-wrap">
        {render_table(rows)}
      </div>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="Skip OCR for faster builds (less accurate).")
    args = ap.parse_args()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    pdf = m.pick_latest_pdf(Path(m.DEFAULT_RESUMES_DIR))
    rows = build_rows(pdf, fast=args.fast)
    INDEX_HTML.write_text(build_html(pdf, rows, fast=args.fast), encoding="utf-8")
    print(f"Wrote {INDEX_HTML}")


if __name__ == "__main__":
    main()
