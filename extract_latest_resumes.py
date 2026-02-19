#!/usr/bin/env python3
from __future__ import annotations

import re
import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# -----------------------------
# Config
# -----------------------------
DEFAULT_RESUMES_DIR = Path(
    "/Users/Zachary/Library/CloudStorage/OneDrive-TheUniversityofTampa/Senior year/Resumes"
)

OCR_DPI = 250
MAX_OCR_PAGES_PER_FIELD = 12  # small bump helps multi-page events

# -----------------------------
# Stable header label lines (FIXED)
# -----------------------------
RE_DATE_LABEL     = re.compile(r"(?mi)\bDATE\s*:")
RE_MASTER_ACCOUNT = re.compile(r"(?mi)\bMASTER\s+ACCOUNT\s+NUMBER\s*:")
RE_GROUP_CODE     = re.compile(r"(?mi)\bGROUP\s+CODE\s*:")
RE_EVENTS_MANAGER = re.compile(r"(?mi)\bEVENTS\s+MANAGER\s*:")
RE_SALES_MANAGER  = re.compile(r"(?mi)\bSALES\s+MANAGER\s*:")
RE_SENIOR_RS      = re.compile(r"(?mi)\bSENIOR\s+RESERVATIONS\s+SPECIALIST\s*:")

LABELS = [
    ("DATE", RE_DATE_LABEL),
    ("MASTER", RE_MASTER_ACCOUNT),
    ("GROUP", RE_GROUP_CODE),
    ("EV_MGR", RE_EVENTS_MANAGER),
    ("SALES", RE_SALES_MANAGER),
    ("SRS", RE_SENIOR_RS),
]

# -----------------------------
# Other fields
# -----------------------------
FIELD_STOP = (
    r"(?=\b(?:ORGANIZATION|NAME\s*OF\s*(?:MEETING|EVENT)|DATE|MASTER\s+ACCOUNT\s+NUMBER|GROUP\s+CODE|"
    r"EVENTS\s+MANAGER|SALES\s+MANAGER|SENIOR\s+RESERVATIONS\s+SPECIALIST|MAIN\s+ARRIVAL\s+DATE|"
    r"MAIN\s+DEPARTURE\s+DATE|ARRIVAL\s+DATE|DEPARTURE\s+DATE|EVENT\s+DATE|ROOM\s+BLOCK\s+START\s+DATE|"
    r"ROOM\s+BLOCK\s+END\s+DATE)\b\s*:|$)"
)

RE_ORG   = re.compile(rf"(?mis)\bORGANIZATION\s*:\s*(.+?){FIELD_STOP}")
RE_NAME  = re.compile(rf"(?mis)\bNAME\s*OF\s*(?:MEETING|EVENT)\s*:?\s*(.+?){FIELD_STOP}")
RE_NAME2 = re.compile(rf"(?mis)\bNAMEOF(?:MEETING|EVENT)\s*:?\s*(.+?){FIELD_STOP}")
RE_MASTER_DESC = re.compile(r"(?mis)\bMASTER\s+ACCOUNT\s+NUMBER\s*:\s*\d+\s+(.+?)(?=\bGROUP\s+CODE\b|$)")
RE_GROUP_VALUE = re.compile(r"(?mis)\bGROUP\s+CODE\s*:\s*([A-Za-z0-9][A-Za-z0-9() .&'/-]{1,80})")

RE_MAIN_ARRIVAL = re.compile(rf"(?mis)\bMAIN\s+ARRIVAL\s+DATE\s*:?\s*(.+?){FIELD_STOP}")
RE_MAIN_DEPART  = re.compile(rf"(?mis)\bMAIN\s+DEPARTURE\s+DATE\s*:?\s*(.+?){FIELD_STOP}")
RE_EVENT_DATE   = re.compile(rf"(?mis)\bEVENT\s+DATE\s*:?\s*(.+?){FIELD_STOP}")
RE_ARRIVAL_GENERIC = re.compile(rf"(?mis)\bARRIVAL\s+DATE\s*:?\s*(.+?){FIELD_STOP}")
RE_DEPART_GENERIC  = re.compile(rf"(?mis)\bDEPARTURE\s+DATE\s*:?\s*(.+?){FIELD_STOP}")

# Room block fallback dates (very common)
RE_ROOM_BLOCK_START = re.compile(r"(?mi)\bRoom\s+Block\s+Start\s+Date\b\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})")
RE_ROOM_BLOCK_END   = re.compile(r"(?mi)\bRoom\s+Block\s+End\s+Date\b\s*([0-9]{1,2}[/-][0-9]{1,2}[/-][0-9]{2,4})")

RE_ARRIVAL_METHOD = re.compile(
    r"(?mi)\bARRIVAL\s*METHOD\b.*?\[?(\d{1,3})\]?\s*%?\s*Drive\s*in.*?\[?(\d{1,3})\]?\s*%?\s*Fly\s*in",
    re.S
)
RE_DRIVE_FLY_LOOSE = re.compile(
    r"(?mi)\bDrive[-\s]*in\b[^0-9]{0,80}(\d{1,3})\s*%?\b.*?\bFly[-\s]*in\b[^0-9]{0,80}(\d{1,3})\s*%?\b",
    re.S
)
RE_ARRIVAL_METHOD_DRIVE_ONLY = re.compile(r"(?mi)\bARRIVAL\s*METHOD\b[^\n]{0,140}?\[?(\d{1,3})\]?\s*%?\s*Drive[-\s]*in\b")
RE_ARRIVAL_METHOD_FLY_ONLY   = re.compile(r"(?mi)\bARRIVAL\s*METHOD\b[^\n]{0,140}?\[?(\d{1,3})\]?\s*%?\s*Fly[-\s]*in\b")
RE_DRIVE_ONLY = re.compile(r"(?mi)\bDrive[-\s]*in\b[^0-9]{0,40}([\-]?\d{1,3})\s*%\b")
RE_FLY_ONLY   = re.compile(r"(?mi)\bFly[-\s]*in\b[^0-9]{0,40}([\-]?\d{1,3})\s*%\b")

RE_VAL_REDUCED_TO = re.compile(
    r"(?mi)\breduced\s+to\b.*?\$?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)\s*overnight\s*/\s*\$?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)\s*(?:day|daily)"
)
RE_VALET_PARKING_LINE = re.compile(
    r"(?mi)\bVALET\s+PARKING\b.*?\bOvernight\s*:\s*\$?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)\b.*?\bDaily\s*:\s*\$?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)\b",
    re.S
)
RE_DAILY_VALET_PARKING     = re.compile(r"(?mi)\bDAILY\s+VALET\s+PARKING\b[^0-9$]{0,60}\$?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)")
RE_OVERNIGHT_VALET_PARKING = re.compile(r"(?mi)\bOVERNIGHT\s+VALET\s+PARKING\b[^0-9$]{0,60}\$?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)")
RE_VALET_SERVICES_DAILY    = re.compile(r"(?mi)\bVALET\s+SERVICES\b.*?\bDaily\s*:\s*\$?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)\b", re.S)
RE_DAILY_VALET_MONEY_FIRST = re.compile(r"(?mi)\$?\s*([0-9]{1,3}(?:\.[0-9]{1,2})?)\s*(?:Daily\s+)?Valet\s+Parking\b")

RE_DISCOUNT_HINT = re.compile(r"(?mi)\b(discount(?:ed)?|reduced|concession)\b")
RE_BILL_TO_EPO   = re.compile(r"(?mi)\bBill\s+to\s+EPO\b")
RE_BILL_TO_MASTER = re.compile(
    r"(?mi)\bBill(?:ed)?\s+to\s+Master\s+Account\b|"
    r"\bto\s+Master\s+Account\b|"
    r"\bMaster\s+Accounts?\s+Master\s+Account\s*#\b|"
    r"\bMaster\s+Account\s+All\b"
)
RE_GUEST_TO_PAY = re.compile(r"(?mi)\bGuest\s+to\s+Pay\b")
RE_BILLING_INSTRUCTIONS = re.compile(r"(?mi)\bBilling\s+Instructions\b")
RE_XXX_MARK = re.compile(r"(?mi)\bX{2,}\b")

# Header-only fallback signals
RE_ORG_LINE  = re.compile(r"(?mi)\bORGANIZATION\s*:")
RE_NAME_LINE = re.compile(r"(?mi)\bNAME\s+OF\s+(?:MEETING|EVENT)\s*:")

# -----------------------------
# Date parsing
# -----------------------------
NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
TEXTUAL_DATE_RE = re.compile(
    r"\b(?:(?:Mon|Tue|Tues|Wed|Thu|Thur|Fri|Sat|Sun|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*)?"
    r"((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))\s+"
    r"(\d{1,2})(?:st|nd|rd|th)?"
    r"(?:,\s*(\d{4}))?\b",
    re.I,
)
MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# -----------------------------
# Candidate regions (fractions)
# ADD top_header so OCR can see DATE/MASTER on pages like your page 24.
# -----------------------------
CANDIDATE_REGIONS = [
    ("top_header",      (0.02, 0.00, 0.98, 0.22)),  # <-- NEW
    ("upper_mid_full",  (0.02, 0.12, 0.98, 0.65)),
    ("right_table_high",(0.28, 0.18, 0.98, 0.70)),
    ("right_table_mid", (0.28, 0.25, 0.98, 0.80)),
    ("mid_band_full",   (0.02, 0.22, 0.98, 0.82)),
    ("mid_band_right",  (0.40, 0.22, 0.98, 0.82)),
]

# Field crops (OCR fallback)
CROP_DATES_SECTION  = (0.02, 0.08, 0.98, 0.58)
CROP_ARRIVAL_METHOD = (0.02, 0.30, 0.80, 0.70)
CROP_VALET_SECTION  = (0.02, 0.48, 0.98, 0.96)
CROP_BILLING_SECTION = (0.02, 0.30, 0.98, 0.96)

# -----------------------------
# Helpers
# -----------------------------
def stitch_broken_words(text: str) -> str:
    if not text:
        return ""
    cur = text.replace("\r\n", "\n").replace("\r", "\n")
    for _ in range(8):
        # Join likely split words, but never across line breaks.
        new = re.sub(r"\b([A-Za-z]{2,})[ \t]+([a-z]{1,})\b", r"\1\2", cur)
        if new == cur:
            break
        cur = new
    cur = re.sub(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), cur)
    return cur

def first_group(rx: re.Pattern, text: str, group: int = 1) -> Optional[str]:
    m = rx.search(text or "")
    if not m:
        return None
    val = m.group(group).strip()
    val = re.sub(r"[ \t]+", " ", val).strip(" :|-")
    return val or None

def pick_latest_pdf(folder: Path) -> Path:
    pdfs = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {folder}")
    return max(pdfs, key=lambda p: p.stat().st_mtime)

def _make_date(y: int, m: int, d: int) -> Optional[datetime]:
    try:
        return datetime(y, m, d)
    except Exception:
        return None

def extract_first_date(text: str) -> Optional[datetime]:
    if not text:
        return None
    m = NUMERIC_DATE_RE.search(text)
    if m:
        mm, dd, yy = m.groups()
        try:
            mi, di, yi = int(mm), int(dd), int(yy)
            if yi < 100:
                yi = 2000 + yi
            return _make_date(yi, mi, di)
        except Exception:
            pass
    m = TEXTUAL_DATE_RE.search(text)
    if m:
        mon, d1, y = m.groups()
        if not y:
            return None
        key = mon.lower()
        key = key if key in MONTHS else key[:3]
        mm = MONTHS.get(key)
        if mm:
            return _make_date(int(y), mm, int(d1))
    return None

def extract_all_dates(text: str) -> List[datetime]:
    out: List[datetime] = []
    if not text:
        return out

    for m in NUMERIC_DATE_RE.finditer(text):
        mm, dd, yy = m.groups()
        try:
            mi, di, yi = int(mm), int(dd), int(yy)
            if yi < 100:
                yi = 2000 + yi
            dt = _make_date(yi, mi, di)
            if dt:
                out.append(dt)
        except Exception:
            continue

    for m in TEXTUAL_DATE_RE.finditer(text):
        mon, d1, y = m.groups()
        if not y:
            continue
        key = mon.lower()
        key = key if key in MONTHS else key[:3]
        mm = MONTHS.get(key)
        if not mm:
            continue
        dt = _make_date(int(y), mm, int(d1))
        if dt:
            out.append(dt)

    # Deduplicate while preserving chronological sort.
    uniq = {(d.year, d.month, d.day): d for d in out}
    return sorted(uniq.values())

def fmt_short(dt: Optional[datetime]) -> str:
    if not dt:
        return "—"
    return dt.strftime("%a %-m/%-d/%y")

def label_hits(text: str) -> List[str]:
    hits = []
    for name, rx in LABELS:
        if rx.search(text or ""):
            hits.append(name)
    return hits

def is_new_event_start(hits: List[str], header_text: str) -> bool:
    hs = set(hits or [])

    # Primary: DATE or MASTER + another label
    if (("DATE" in hs) or ("MASTER" in hs)) and len(hs) >= 2:
        return True

    # Secondary: ORG+NAME in header + at least one manager-ish label
    ht = header_text or ""
    has_org = bool(RE_ORG_LINE.search(ht))
    has_name = bool(RE_NAME_LINE.search(ht))
    has_some_label = any(k in hs for k in ("GROUP", "EV_MGR", "SALES", "SRS"))
    return has_org and has_name and has_some_label

def has_event_identity(text: str) -> bool:
    t = text or ""
    return bool(RE_ORG.search(t) or RE_NAME.search(t) or RE_NAME2.search(t))

# -----------------------------
# OCR (cached)
# -----------------------------
def _require_ocr_deps():
    try:
        from pdf2image import convert_from_path  # noqa
    except ImportError:
        raise SystemExit("Missing dependency: pdf2image\nInstall: pip3 install pdf2image")
    try:
        import pytesseract  # noqa
    except ImportError:
        raise SystemExit("Missing dependency: pytesseract\nInstall: pip3 install pytesseract")

_image_cache: Dict[int, object] = {}
_ocr_cache: Dict[tuple, str] = {}

def render_page_image(pdf_path: Path, page_index: int):
    if page_index in _image_cache:
        return _image_cache[page_index]
    _require_ocr_deps()
    from pdf2image import convert_from_path  # type: ignore
    imgs = convert_from_path(
        str(pdf_path),
        dpi=OCR_DPI,
        first_page=page_index + 1,
        last_page=page_index + 1,
    )
    img = imgs[0] if imgs else None
    _image_cache[page_index] = img
    return img

def ocr_image(img, psm: int) -> str:
    import pytesseract  # type: ignore
    config = f"--oem 3 --psm {psm}"
    return stitch_broken_words(pytesseract.image_to_string(img, config=config))

def ocr_crop(pdf_path: Path, page_index: int, box_frac: Tuple[float, float, float, float], psm: int = 6) -> str:
    key = (page_index, box_frac, psm)
    if key in _ocr_cache:
        return _ocr_cache[key]

    img = render_page_image(pdf_path, page_index)
    if img is None:
        _ocr_cache[key] = ""
        return ""

    w, h = img.size
    x0, y0, x1, y1 = box_frac
    crop = img.crop((int(x0*w), int(y0*h), int(x1*w), int(y1*h)))

    # two OCR modes; one can catch table-ish headers better
    t1 = ocr_image(crop, psm=6)
    t2 = ocr_image(crop, psm=4)
    text = (t1 + "\n" + t2).strip()

    _ocr_cache[key] = text
    return text

# -----------------------------
# pdfplumber (fast when PDF has real text)
# -----------------------------
def plumber_region_text(pdf_path: Path, page_index: int, box_frac: Tuple[float,float,float,float]) -> str:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return ""
    with pdfplumber.open(str(pdf_path)) as pdf:
        page = pdf.pages[page_index]
        W = float(page.width)
        H = float(page.height)
        x0f, y0f, x1f, y1f = box_frac
        bbox = (x0f*W, y0f*H, x1f*W, y1f*H)
        try:
            cropped = page.crop(bbox)
            words = cropped.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
        except Exception:
            return ""
        if not words:
            return ""
        parts = [w["text"] for w in words if w.get("text")]
        return stitch_broken_words(" ".join(parts))

def read_pdf_pages_text(pdf_path: Path) -> List[str]:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        raise SystemExit("Missing dependency: pdfplumber\nInstall: pip3 install pdfplumber")
    out: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for p in pdf.pages:
            out.append(stitch_broken_words(p.extract_text() or ""))
    return out

# -----------------------------
# Header detection (text-first, OCR fallback)
# -----------------------------
def best_header_text(pdf_path: Path, page_index: int, debug: bool=False) -> Tuple[str, List[str], str]:
    best_text = ""
    best_hits: List[str] = []
    best_name = "none"
    best_letters = -1

    # 1) plumber
    for region_name, frac in CANDIDATE_REGIONS:
        ptxt = plumber_region_text(pdf_path, page_index, frac)
        phits = label_hits(ptxt)
        pletters = len(re.findall(r"[A-Za-z]", ptxt))

        if (len(phits) > len(best_hits)) or (len(phits) == len(best_hits) and pletters > best_letters):
            best_text, best_hits, best_name, best_letters = ptxt, phits, f"plumber:{region_name}", pletters

    # 2) OCR if weak
    need_ocr = (len(best_hits) < 2) or (best_letters < 40)
    if need_ocr:
        o_best_text = ""
        o_best_hits: List[str] = []
        o_best_name = "none"
        o_best_letters = -1

        for region_name, frac in CANDIDATE_REGIONS:
            otxt = ocr_crop(pdf_path, page_index, frac, psm=6)
            ohits = label_hits(otxt)
            oletters = len(re.findall(r"[A-Za-z]", otxt))

            if (len(ohits) > len(o_best_hits)) or (len(ohits) == len(o_best_hits) and oletters > o_best_letters):
                o_best_text, o_best_hits, o_best_name, o_best_letters = otxt, ohits, f"ocr:{region_name}", oletters

        if (len(o_best_hits) > len(best_hits)) or (len(o_best_hits) == len(best_hits) and o_best_letters > best_letters):
            best_text, best_hits, best_name = o_best_text, o_best_hits, o_best_name

    if debug and best_text:
        sample = best_text.replace("\n", " ")[:140]
        print(f"      [DBG] best_header_source={best_name} sample='{sample}...'")

    return best_text, best_hits, best_name

# -----------------------------
# Group pages into events
# -----------------------------
@dataclass
class EventBlock:
    start_page: int
    pages: List[int]
    text: str

def group_pages_into_events(pdf_path: Path, pages_text: List[str], debug: bool=False) -> List[EventBlock]:
    events: List[EventBlock] = []
    buf_text: List[str] = []
    buf_pages: List[int] = []
    start_idx: Optional[int] = None

    for i, base in enumerate(pages_text):
        header_text, hits, src = best_header_text(pdf_path, i, debug=debug)
        combined = (base + "\n" + header_text).strip()

        # IMPORTANT: start decision based on HEADER TEXT ONLY
        start = is_new_event_start(hits, header_text)

        if debug:
            print(f"[DBG] Page {i+1:02d} start={start} hits={','.join(hits) if hits else '-'} src={src}")

        if start:
            if buf_text and start_idx is not None:
                events.append(EventBlock(start_idx, buf_pages[:], "\n".join(buf_text).strip()))
            buf_text = [combined]
            buf_pages = [i]
            start_idx = i
        else:
            if buf_text:
                buf_text.append(combined)
                buf_pages.append(i)

    if buf_text and start_idx is not None:
        events.append(EventBlock(start_idx, buf_pages[:], "\n".join(buf_text).strip()))

    return [e for e in events if e.text.strip()]

# -----------------------------
# Extraction
# -----------------------------
@dataclass
class EventRow:
    event_group: str
    main_arrival: str
    main_departure: str
    drive_in: str
    fly_in: str
    valet_pricing: str
    billing: str

def normalize_event_label(org: str, name: str) -> str:
    org = (org or "").strip()
    name = (name or "").strip()
    if org and name:
        o, n = org.lower(), name.lower()
        if n.startswith(o) or o in n:
            return name
        return f"{org} – {name}"
    return org or name or "—"

def fallback_event_label(text: str) -> Optional[str]:
    master_desc = first_group(RE_MASTER_DESC, text) or ""
    group_val = first_group(RE_GROUP_VALUE, text) or ""

    master_desc = re.sub(r"\s+", " ", master_desc).strip(" :|-")
    group_val = re.sub(r"\s+", " ", group_val).strip(" :|-")

    # Remove staff/attendee parenthetical tags from fallback labels.
    master_desc = re.sub(r"\((?:ATTENDEES|STAFF)\)", "", master_desc, flags=re.I).strip(" -")
    group_val = re.sub(r"\((?:ATTENDEES|STAFF)\)", "", group_val, flags=re.I).strip(" -")

    if master_desc and len(master_desc) > 2:
        return master_desc
    if group_val and len(group_val) > 2:
        return group_val
    return None

def _iter_ocr_pages(pages: List[int]) -> List[int]:
    return pages[:MAX_OCR_PAGES_PER_FIELD]

def ocr_search_over_event(pdf_path: Path, page_indices: List[int], crop: Tuple[float,float,float,float], psm: int) -> str:
    chunks: List[str] = []
    for pi in _iter_ocr_pages(page_indices):
        t = ocr_crop(pdf_path, pi, crop, psm=psm)
        if t.strip():
            chunks.append(t)
    return "\n".join(chunks).strip()

def extract_dates(event_text: str, pdf_path: Path, event_pages: List[int]) -> Tuple[str,str]:
    # text-first
    arr_raw = (
        first_group(RE_MAIN_ARRIVAL, event_text)
        or first_group(RE_ARRIVAL_GENERIC, event_text)
        or first_group(RE_EVENT_DATE, event_text)
    )
    dep_raw = (
        first_group(RE_MAIN_DEPART, event_text)
        or first_group(RE_DEPART_GENERIC, event_text)
        or first_group(RE_EVENT_DATE, event_text)
    )

    # room block fallback
    if not arr_raw:
        m = RE_ROOM_BLOCK_START.search(event_text)
        arr_raw = m.group(1) if m else arr_raw
    if not dep_raw:
        m = RE_ROOM_BLOCK_END.search(event_text)
        dep_raw = m.group(1) if m else dep_raw

    arr_dt = extract_first_date(arr_raw or "")
    dep_dt = extract_first_date(dep_raw or "")

    # OCR fallback across event pages
    if not arr_dt or not dep_dt:
        dates_ocr = ocr_search_over_event(pdf_path, event_pages, CROP_DATES_SECTION, psm=6)

        arr_raw2 = (
            first_group(RE_MAIN_ARRIVAL, dates_ocr)
            or first_group(RE_ARRIVAL_GENERIC, dates_ocr)
            or first_group(RE_EVENT_DATE, dates_ocr)
        )
        dep_raw2 = (
            first_group(RE_MAIN_DEPART, dates_ocr)
            or first_group(RE_DEPART_GENERIC, dates_ocr)
            or first_group(RE_EVENT_DATE, dates_ocr)
        )

        if not arr_raw2:
            m = RE_ROOM_BLOCK_START.search(dates_ocr)
            arr_raw2 = m.group(1) if m else arr_raw2
        if not dep_raw2:
            m = RE_ROOM_BLOCK_END.search(dates_ocr)
            dep_raw2 = m.group(1) if m else dep_raw2

        arr_dt = arr_dt or extract_first_date(arr_raw2 or "")
        dep_dt = dep_dt or extract_first_date(dep_raw2 or "")

    # Last fallback: use any dates found in the event block.
    if not arr_dt or not dep_dt:
        all_dates = extract_all_dates(event_text)
        if not all_dates:
            dates_ocr = ocr_search_over_event(pdf_path, event_pages, CROP_DATES_SECTION, psm=6)
            all_dates = extract_all_dates(dates_ocr)
        if all_dates:
            arr_dt = arr_dt or all_dates[0]
            dep_dt = dep_dt or all_dates[-1]

    return fmt_short(arr_dt), fmt_short(dep_dt)

def extract_drive_fly(event_text: str, pdf_path: Path, event_pages: List[int]) -> Tuple[str,str]:
    def to_pct(raw: Optional[str]) -> Optional[int]:
        if not raw:
            return None
        m = re.search(r"\d{1,3}", raw)
        if not m:
            return None
        v = int(m.group(0))
        return v if 0 <= v <= 100 else None

    def extract_pair(text: str) -> Tuple[Optional[int], Optional[int]]:
        m = RE_ARRIVAL_METHOD.search(text) or RE_DRIVE_FLY_LOOSE.search(text)
        if m:
            return to_pct(m.group(1)), to_pct(m.group(2))
        # Single-sided values on ARRIVAL METHOD lines (e.g., "ARRIVAL METHOD 100 % Drive in")
        d = to_pct(first_group(RE_ARRIVAL_METHOD_DRIVE_ONLY, text))
        f = to_pct(first_group(RE_ARRIVAL_METHOD_FLY_ONLY, text))
        if d is not None or f is not None:
            return d, f
        return to_pct(first_group(RE_DRIVE_ONLY, text)), to_pct(first_group(RE_FLY_ONLY, text))

    drive, fly = extract_pair(event_text)
    if drive is None or fly is None:
        am_ocr = ocr_search_over_event(pdf_path, event_pages, CROP_ARRIVAL_METHOD, psm=4)
        d2, f2 = extract_pair(am_ocr)
        drive = drive if drive is not None else d2
        fly = fly if fly is not None else f2

    # Wider OCR fallback for pages where arrival method isn't in the expected crop.
    if drive is None or fly is None:
        full_ocr = ocr_search_over_event(pdf_path, event_pages, (0.02, 0.02, 0.98, 0.95), psm=6)
        d3, f3 = extract_pair(full_ocr)
        drive = drive if drive is not None else d3
        fly = fly if fly is not None else f3

    # If only one side is present, infer the other when values are percentage shares.
    if drive is not None and fly is None:
        if 0 <= drive <= 100:
            fly = 100 - drive
    elif fly is not None and drive is None:
        if 0 <= fly <= 100:
            drive = 100 - fly

    return (f"{drive}%" if drive is not None else "—", f"{fly}%" if fly is not None else "—")

def _to_money(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def extract_valet(event_text: str, pdf_path: Path, event_pages: List[int]) -> str:
    discounted = bool(RE_DISCOUNT_HINT.search(event_text))
    bill_to_epo = bool(RE_BILL_TO_EPO.search(event_text))

    def ok(v: Optional[float]) -> bool:
        return v is not None and v >= 1.0

    def notes():
        n = []
        if discounted: n.append("discounted")
        if bill_to_epo: n.append("Bill to EPO")
        return f" ({'; '.join(n)})" if n else ""

    m = RE_VAL_REDUCED_TO.search(event_text)
    if m:
        on, day = _to_money(m.group(1)), _to_money(m.group(2))
        if ok(on) and ok(day):
            return f"${on:g} Overnight / ${day:g} Daily" + notes()

    m = RE_VALET_PARKING_LINE.search(event_text)
    if m:
        on, day = _to_money(m.group(1)), _to_money(m.group(2))
        if ok(on) and ok(day):
            return f"${on:g} Overnight / ${day:g} Daily" + notes()

    on = _to_money(first_group(RE_OVERNIGHT_VALET_PARKING, event_text) or "")
    day = _to_money(first_group(RE_DAILY_VALET_PARKING, event_text) or "")
    if ok(on) and ok(day):
        return f"${on:g} Overnight / ${day:g} Daily" + notes()
    if ok(day):
        return f"${day:g} Daily Only" + notes()
    day_mf = _to_money(first_group(RE_DAILY_VALET_MONEY_FIRST, event_text) or "")
    if ok(day_mf):
        return f"${day_mf:g} Daily Only" + notes()

    valet_ocr = ocr_search_over_event(pdf_path, event_pages, CROP_VALET_SECTION, psm=4)
    if valet_ocr:
        discounted = discounted or bool(RE_DISCOUNT_HINT.search(valet_ocr))
        bill_to_epo = bill_to_epo or bool(RE_BILL_TO_EPO.search(valet_ocr))

        m = RE_VALET_PARKING_LINE.search(valet_ocr)
        if m:
            on2, day2 = _to_money(m.group(1)), _to_money(m.group(2))
            if ok(on2) and ok(day2):
                return f"${on2:g} Overnight / ${day2:g} Daily" + notes()

        on2 = _to_money(first_group(RE_OVERNIGHT_VALET_PARKING, valet_ocr) or "")
        day2 = _to_money(first_group(RE_DAILY_VALET_PARKING, valet_ocr) or "")
        if ok(on2) and ok(day2):
            return f"${on2:g} Overnight / ${day2:g} Daily" + notes()
        if ok(day2):
            return f"${day2:g} Daily Only" + notes()
        day2_mf = _to_money(first_group(RE_DAILY_VALET_MONEY_FIRST, valet_ocr) or "")
        if ok(day2_mf):
            return f"${day2_mf:g} Daily Only" + notes()

        m = RE_VALET_SERVICES_DAILY.search(valet_ocr)
        if m:
            day3 = _to_money(m.group(1))
            if ok(day3):
                return f"${day3:g} Daily Only" + notes()

    return "—"

def _billing_col_for_xxx(line: str, master_idx: int, guest_idx: int) -> Optional[str]:
    x_idx = line.find("XXX")
    if x_idx < 0 or master_idx < 0 or guest_idx < 0:
        return None
    return "master" if abs(x_idx - master_idx) <= abs(x_idx - guest_idx) else "guest"

def _extract_billing_from_layout(pdf_path: Path, event_pages: List[int]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    daily_col: Optional[str] = None
    overnight_col: Optional[str] = None
    parking_col: Optional[str] = None
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return daily_col, overnight_col, parking_col

    with pdfplumber.open(str(pdf_path)) as pdf:
        for pi in _iter_ocr_pages(event_pages):
            text = pdf.pages[pi].extract_text(layout=True) or ""
            lines = text.splitlines()
            master_idx = -1
            guest_idx = -1
            for ln in lines:
                u = ln.upper()
                if "CHARGES" in u and "MASTER ACCOUNT" in u and "GUEST TO PAY" in u:
                    master_idx = u.find("MASTER ACCOUNT")
                    guest_idx = u.find("GUEST TO PAY")
                    continue
                if master_idx < 0 or guest_idx < 0:
                    continue

                col = _billing_col_for_xxx(u, master_idx, guest_idx)
                if col is None:
                    continue
                if "DAILY VALET PARKING" in u:
                    daily_col = col
                elif "OVERNIGHT VALET PARKING" in u:
                    overnight_col = col
                elif re.search(r"\bPARKING\b", u):
                    parking_col = col
    return daily_col, overnight_col, parking_col

def extract_billing(event_text: str, pdf_path: Path, event_pages: List[int]) -> str:
    daily_col, overnight_col, parking_col = _extract_billing_from_layout(pdf_path, event_pages)

    # Fill missing valet rows with parking row billing when available.
    daily_col = daily_col or parking_col
    overnight_col = overnight_col or parking_col

    # Explicit wording in narrative sections can supplement table parsing.
    text_master = bool(RE_BILL_TO_MASTER.search(event_text) or RE_BILL_TO_EPO.search(event_text))
    text_guest = bool(RE_GUEST_TO_PAY.search(event_text) and RE_XXX_MARK.search(event_text))

    if daily_col == "master" and overnight_col == "master":
        return "Master Account"
    if daily_col == "guest" and overnight_col == "guest":
        return "Guest to Pay"
    if daily_col and overnight_col and daily_col != overnight_col:
        return "Split (Daily Master / Overnight Guest)" if daily_col == "master" else "Split (Daily Guest / Overnight Master)"
    if daily_col == "master" or overnight_col == "master":
        return "Master Account"
    if daily_col == "guest" or overnight_col == "guest":
        return "Guest to Pay"

    if text_master and text_guest:
        return "Mixed"
    if text_master:
        return "Master Account"
    if text_guest:
        return "Guest to Pay"

    # OCR fallback for image-heavy billing sections where layout text is sparse.
    bill_ocr = ocr_search_over_event(pdf_path, event_pages, CROP_BILLING_SECTION, psm=6)
    ocr_master = bool(RE_BILL_TO_MASTER.search(bill_ocr) or RE_BILL_TO_EPO.search(bill_ocr))
    ocr_guest = bool(RE_GUEST_TO_PAY.search(bill_ocr) and RE_XXX_MARK.search(bill_ocr))
    if ocr_master and ocr_guest:
        return "Mixed"
    if ocr_master:
        return "Master Account"
    if ocr_guest:
        return "Guest to Pay"
    return "—"

def _truncate(val: str, width: int) -> str:
    if len(val) <= width:
        return val
    if width <= 1:
        return val[:width]
    return val[: width - 1] + "…"

def print_table(rows: List[EventRow]) -> None:
    headers = ["Event / Group", "Main Arrival", "Main Departure", "Drive-In %", "Fly-In %", "Valet Pricing", "Billing"]
    base_mins = [30, 12, 14, 10, 9, 20, 10]
    max_cols = [44, 12, 14, 10, 9, 32, 26]
    all_rows = [
        [r.event_group, r.main_arrival, r.main_departure, r.drive_in, r.fly_in, r.valet_pricing, r.billing]
        for r in rows
    ]
    widths = []
    for i, h in enumerate(headers):
        max_cell = max([len(str(r[i])) for r in all_rows], default=0)
        widths.append(min(max(base_mins[i], len(h), max_cell), max_cols[i]))

    def fmt(vals):
        return " | ".join(_truncate(str(v), w).ljust(w) for v, w in zip(vals, widths))

    print(fmt(headers))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for r in rows:
        print(fmt([r.event_group, r.main_arrival, r.main_departure, r.drive_in, r.fly_in, r.valet_pricing, r.billing]))

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(DEFAULT_RESUMES_DIR))
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--pdf", default="", help="Optional: explicit PDF path (overrides --folder latest)")
    args = ap.parse_args()

    if args.pdf:
        pdf = Path(args.pdf).expanduser()
        if not pdf.exists():
            raise SystemExit(f"PDF not found: {pdf}")
    else:
        folder = Path(args.folder)
        pdf = pick_latest_pdf(folder)

    pages_text = read_pdf_pages_text(pdf)
    events = group_pages_into_events(pdf, pages_text, debug=args.debug)

    if args.debug:
        print(f"[DBG] Extracted {len(events)} event(s)")

    rows: List[EventRow] = []
    for ev in events:
        org = first_group(RE_ORG, ev.text) or ""
        name = first_group(RE_NAME, ev.text) or first_group(RE_NAME2, ev.text) or ""
        label = normalize_event_label(org, name)
        if label == "—":
            label = fallback_event_label(ev.text) or "—"

        arr, dep = extract_dates(ev.text, pdf, ev.pages)
        drive, fly = extract_drive_fly(ev.text, pdf, ev.pages)
        valet = extract_valet(ev.text, pdf, ev.pages)
        billing = extract_billing(ev.text, pdf, ev.pages)

        if label == "—" and arr == "—" and dep == "—" and drive == "—" and fly == "—" and valet == "—" and billing == "—":
            continue
        rows.append(EventRow(label, arr, dep, drive, fly, valet, billing))

    print(f"\nUsing PDF: {pdf.name}\n")
    print_table(rows)

if __name__ == "__main__":
    main()
