from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
LTTextContainer, LTTextLine, LTChar, LTFigure, LAParams, LTLine
)
import io
import os
import unicodedata
import re

# ── Optional RTL-support libraries ─────────────────────────────────────────
# Install with:  pip install arabic-reshaper python-bidi
try:
import arabic_reshaper
from bidi.algorithm import get_display
RTL_SUPPORT = True
except ImportError:
RTL_SUPPORT = False
print(
"WARNING: arabic-reshaper / python-bidi not installed. "
"Arabic & Urdu text may render in wrong order. "
"Run: pip install arabic-reshaper python-bidi"
)

app = FastAPI(title="PDF Extractor API")

# ── Unicode ranges that belong to RTL scripts ───────────────────────────────
RTL_RANGES = [
(0x0590, 0x05FF),   # Hebrew
(0x0600, 0x06FF),   # Arabic (covers Urdu too)
(0x0700, 0x074F),   # Syriac
(0x0750, 0x077F),   # Arabic Supplement
(0x0780, 0x07BF),   # Thaana
(0x08A0, 0x08FF),   # Arabic Extended-A
(0xFB00, 0xFB4F),   # Alphabetic Presentation Forms (Hebrew)
(0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
(0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
]

# Scripts that explicitly use Arabic-style glyphs (for reshaper)
ARABIC_SCRIPT_FONTS = re.compile(
r"(arabic|urdu|farsi|persian|pashto|sindhi|uyghur|naskh|nastaliq|amiri|scheherazade|lateef)",
re.IGNORECASE,
)


def is_rtl_char(ch: str) -> bool:
"""Return True if the character belongs to a RTL Unicode block."""
cp = ord(ch)
return any(lo <= cp <= hi for lo, hi in RTL_RANGES)


def text_is_rtl(text: str) -> bool:
"""
    Heuristic: if more than 30 % of the *letter* characters are RTL, treat the
    whole string as RTL.  This correctly handles mixed punctuation / digits.
    """
letters = [ch for ch in text if unicodedata.category(ch).startswith("L")]
if not letters:
return False
rtl_count = sum(1 for ch in letters if is_rtl_char(ch))
return (rtl_count / len(letters)) > 0.30


def reshape_rtl(text: str) -> str:
"""
    Apply Arabic reshaping + BiDi algorithm so the string is in correct
    visual/logical display order.  Falls back gracefully when libraries are
    missing.
    """
if not text or not text_is_rtl(text):
return text
if not RTL_SUPPORT:
return text
try:
reshaped = arabic_reshaper.reshape(text)
return get_display(reshaped)
except Exception:
return text


def clean_font_name(fontname: str) -> str:
"""Strip subset prefix like 'BAAAAA+' from embedded font names."""
if "+" in fontname:
return fontname.split("+", 1)[1]
return fontname


def get_font_info(char):
fontname = clean_font_name(getattr(char, "fontname", "Unknown"))
fontsize = getattr(char, "size", 10.0)
lower = fontname.lower()

# Bold detection — covers Latin *and* Arabic font naming conventions
is_bold = any(kw in lower for kw in ("bold", "heavy", "black", "semibold", "demibold", "bd"))

# Italic / oblique detection
is_italic = any(kw in lower for kw in ("italic", "oblique", "slanted", "it", "kursiv"))

return fontname, round(float(fontsize), 1), is_bold, is_italic, "#000000"


def collect_hlines(container):
"""
    Collect all horizontal LTLine elements from a container (page or figure).
    Returns list of (x0, x1, y) tuples.
    """
hlines = []
for el in container:
if isinstance(el, LTLine):
if abs(el.y1 - el.y0) < 2.0:
hlines.append((el.x0, el.x1, el.y0))
elif isinstance(el, LTFigure):
hlines.extend(collect_hlines(el))
return hlines


def is_underlined(line, hlines, tolerance=3.0):
text_x0 = line.x0
text_x1 = line.x1
text_y0 = line.y0

for hx0, hx1, hy in hlines:
y_diff = text_y0 - hy
if 0 <= y_diff <= tolerance:
overlap = min(text_x1, hx1) - max(text_x0, hx0)
if overlap > 5.0:
return True
return False


def detect_alignment(line, page_width: float, is_rtl: bool) -> str:
"""
    Detect text alignment taking script direction into account.

    For LTR text the default is "left"; for RTL the default is "right".
    Centred text is recognised for both directions.
    """
line_center = line.x0 + line.width / 2
page_center = page_width / 2

if abs(line_center - page_center) < 50:
return "center"

# Near the right margin → right-aligned (common for RTL body text)
near_right = (page_width - line.x1) < 50
near_left  = line.x0 < 50

if is_rtl:
if near_right:
return "right"
if near_left:
return "left"   # may be an LTR element on an RTL page
return "right"      # default for RTL
else:
if near_left:
return "left"
if near_right:
return "right"
return "left"       # default for LTR


def extract_text_blocks(container, page_height, page_width, page_num, hlines):
"""
    Extract one block per visual line (LTTextLine).
    Applies RTL reshaping where needed.
    Recurses into LTFigure for nested XObjects.
    """
blocks = []

for element in container:
if isinstance(element, LTFigure):
nested = extract_text_blocks(element, page_height, page_width, page_num, hlines)
blocks.extend(nested)
continue

if not isinstance(element, LTTextContainer):
continue

for line in element:
if not isinstance(line, LTTextLine):
continue

chars = [obj for obj in line if isinstance(obj, LTChar)]
if not chars:
continue

raw_text = line.get_text().strip()
font_family, font_size, is_bold, is_italic, color_hex = get_font_info(chars[0])

# ── Bullet glyph in Symbol font ──────────────────────────────
is_bullet = font_family == "Symbol" and not raw_text
if is_bullet:
raw_text = ""

if not raw_text and not is_bullet:
continue

# ── RTL detection & reshaping ─────────────────────────────────
# Check both the extracted text AND the font name
rtl_by_font = bool(ARABIC_SCRIPT_FONTS.search(font_family))
rtl_by_text = text_is_rtl(raw_text)
is_rtl       = rtl_by_font or rtl_by_text

display_text = reshape_rtl(raw_text) if is_rtl else raw_text

# ── Underline & alignment ─────────────────────────────────────
underline = is_underlined(line, hlines)
alignment = detect_alignment(line, page_width, is_rtl)

idx = len(blocks)
blocks.append({
"text":             display_text,
"rawText":          raw_text,           # original extraction (useful for debugging)
"x":                round(float(line.x0), 2),
"y":                round(float(page_height - line.y1), 2),
"width":            round(float(line.width), 2),
"height":           round(float(line.height), 2),
"fontSize":         font_size,
"fontFamily":       font_family,
"colorHex":         color_hex,
"isBold":           is_bold,
"isItalic":         is_italic,
"isUnderline":      underline,
"isBullet":         is_bullet,
"isRTL":            is_rtl,             # NEW — consumers can use this for layout
"blockId":          f"p{page_num}_b{idx:03d}",
"flowGroup":        "body",
"alignment":        alignment,
"paragraphSpacing": 6.0,
"isFixedPosition":  False,
"continuedFromId":  None,
"continuedToId":    None,
})

return blocks


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
if not file.filename or not file.filename.lower().endswith(".pdf"):
raise HTTPException(status_code=400, detail="Only PDF files are allowed")
try:
content = await file.read()
pages_data = []

laparams = LAParams(
line_margin=0.5,       # keep default grouping
word_margin=0.1,
char_margin=2.0,
boxes_flow=0.5,        # 0.5 works for both LTR and RTL column detection
detect_vertical=False,
)

for page_num, page_layout in enumerate(
extract_pages(io.BytesIO(content), laparams=laparams), start=1
):
page_w = float(page_layout.width)
page_h = float(page_layout.height)

hlines     = collect_hlines(page_layout)
text_blocks = extract_text_blocks(page_layout, page_h, page_w, page_num, hlines)

pages_data.append({
"page":       page_num,
"pageWidth":  round(page_w, 1),
"pageHeight": round(page_h, 1),
"textBlocks": text_blocks,
"rtlSupport": RTL_SUPPORT,   # lets callers know reshaping was active
})

return JSONResponse(content=pages_data)

except Exception as e:
raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/")
async def root():
return {
"message":    "PDF Extractor API is running. Use POST /extract-pdf",
"rtlSupport": RTL_SUPPORT,
}


if __name__ == "__main__":
import uvicorn
port = int(os.getenv("PORT", 8000))
uvicorn.run(app, host="0.0.0.0", port=port)