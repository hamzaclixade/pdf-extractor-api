from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LTTextContainer, LTTextLine, LTChar, LTFigure, LAParams,
    LTLine, LTRect, LTCurve, LTImage, LTLayoutContainer
)
import io
import os
import base64
import fitz          # PyMuPDF  — pip install pymupdf
import pdfplumber    # pip install pdfplumber

app = FastAPI(title="PDF Extractor API")


# ─────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────

def clean_font_name(fontname: str) -> str:
    """Strip subset prefix like 'BAAAAA+' from embedded font names."""
    if "+" in fontname:
        return fontname.split("+", 1)[1]
    return fontname


def get_font_info(char):
    fontname = clean_font_name(getattr(char, "fontname", "Unknown"))
    fontsize = getattr(char, "size", 10.0)
    lower = fontname.lower()
    is_bold = "bold" in lower
    is_italic = "italic" in lower or "oblique" in lower
    return fontname, round(float(fontsize), 1), is_bold, is_italic, "#000000"


def collect_hlines(container):
    """Collect all horizontal LTLine elements (for underline detection)."""
    hlines = []
    for el in container:
        if isinstance(el, LTLine):
            if abs(el.y1 - el.y0) < 2.0:
                hlines.append((el.x0, el.x1, el.y0))
        elif isinstance(el, LTFigure):
            hlines.extend(collect_hlines(el))
    return hlines


def is_underlined(line, hlines, tolerance=3.0):
    text_x0, text_x1, text_y0 = line.x0, line.x1, line.y0
    for hx0, hx1, hy in hlines:
        y_diff = text_y0 - hy
        if 0 <= y_diff <= tolerance:
            overlap = min(text_x1, hx1) - max(text_x0, hx0)
            if overlap > 5.0:
                return True
    return False


# ─────────────────────────────────────────────
#  1.  TEXT BLOCKS  (unchanged logic, cleaned up)
# ─────────────────────────────────────────────

def extract_text_blocks(container, page_height, page_width, page_num, hlines):
    blocks = []

    for element in container:
        if isinstance(element, LTFigure):
            blocks.extend(extract_text_blocks(element, page_height, page_width, page_num, hlines))
            continue

        if not isinstance(element, LTTextContainer):
            continue

        for line in element:
            if not isinstance(line, LTTextLine):
                continue

            chars = [obj for obj in line if isinstance(obj, LTChar)]
            if not chars:
                continue

            line_text = line.get_text().strip()
            font_family, font_size, is_bold, is_italic, color_hex = get_font_info(chars[0])

            is_bullet = font_family == "Symbol" and not line_text
            if is_bullet:
                line_text = ""

            if not line_text and not is_bullet:
                continue

            underline = is_underlined(line, hlines)

            alignment = "left"
            line_center = line.x0 + line.width / 2
            if abs(line_center - page_width / 2) < 50:
                alignment = "center"

            idx = len(blocks)
            blocks.append({
                "text": line_text,
                "x": round(float(line.x0), 2),
                "y": round(float(page_height - line.y1), 2),
                "width": round(float(line.width), 2),
                "height": round(float(line.height), 2),
                "fontSize": font_size,
                "fontFamily": font_family,
                "colorHex": color_hex,
                "isBold": is_bold,
                "isItalic": is_italic,
                "isUnderline": underline,
                "isBullet": is_bullet,
                "blockId": f"p{page_num}_b{idx:03d}",
                "flowGroup": "body",
                "alignment": alignment,
                "paragraphSpacing": 6.0,
                "isFixedPosition": False,
                "continuedFromId": None,
                "continuedToId": None,
            })

    return blocks


# ─────────────────────────────────────────────
#  2.  IMAGES  — via PyMuPDF
# ─────────────────────────────────────────────

def extract_images_from_page(fitz_page, page_num: int, page_height: float) -> list[dict]:
    """
    Extract all raster images embedded in the page.
    Returns a list of dicts with position info and base64-encoded image data.
    """
    images = []
    image_list = fitz_page.get_images(full=True)

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]                    # PDF object reference
        base_image = fitz_page.parent.extract_image(xref)
        if not base_image:
            continue

        img_bytes = base_image["image"]
        img_ext   = base_image.get("ext", "png")   # jpeg / png / jp2 …

        # Locate image on the page via its transform matrix
        bbox = None
        for item in fitz_page.get_image_rects(xref):
            bbox = item
            break

        entry = {
            "imageId":   f"p{page_num}_img{img_index:03d}",
            "extension": img_ext,
            "mimeType":  f"image/{img_ext.replace('jpg', 'jpeg')}",
            "dataBase64": base64.b64encode(img_bytes).decode("utf-8"),
            "width":      base_image.get("width"),
            "height":     base_image.get("height"),
            "colorspace": base_image.get("colorspace", 1),
            # Page-relative bounding box (top-left origin, PDF units)
            "bbox": {
                "x":      round(float(bbox.x0), 2) if bbox else None,
                "y":      round(float(page_height - bbox.y1), 2) if bbox else None,
                "width":  round(float(bbox.width), 2)  if bbox else None,
                "height": round(float(bbox.height), 2) if bbox else None,
            },
        }
        images.append(entry)

    return images


# ─────────────────────────────────────────────
#  3.  TABLES  — via pdfplumber
# ─────────────────────────────────────────────

# Fine-tune these settings if tables are missed or over-detected
TABLE_SETTINGS = {
    "vertical_strategy":   "lines",   # or "text"
    "horizontal_strategy": "lines",   # or "text"
    "snap_tolerance":      3,
    "join_tolerance":      3,
    "edge_min_length":     3,
    "min_words_vertical":  3,
    "min_words_horizontal": 1,
    "text_tolerance":      3,
    "text_x_tolerance":    3,
    "text_y_tolerance":    3,
}

def extract_tables_from_page(plumber_page, page_num: int, page_height: float) -> list[dict]:
    """
    Detect and extract tables using pdfplumber.
    Returns a list of table objects with cell-level data and bounding box.
    """
    tables_out = []
    try:
        found_tables = plumber_page.find_tables(TABLE_SETTINGS)
    except Exception:
        return tables_out

    for tbl_idx, tbl in enumerate(found_tables):
        # Bounding box in pdfplumber coords (bottom-left origin → convert)
        bb = tbl.bbox   # (x0, top, x1, bottom)  — already top-left in pdfplumber
        cells_raw = tbl.extract()   # list[list[str|None]]

        # Build structured row/cell output
        rows = []
        for r_idx, row in enumerate(cells_raw or []):
            cells = []
            for c_idx, cell_text in enumerate(row):
                cells.append({
                    "row":    r_idx,
                    "col":    c_idx,
                    "text":   (cell_text or "").strip(),
                    "isEmpty": not bool((cell_text or "").strip()),
                })
            rows.append(cells)

        tables_out.append({
            "tableId":   f"p{page_num}_tbl{tbl_idx:03d}",
            "numRows":   len(rows),
            "numCols":   max((len(r) for r in rows), default=0),
            "bbox": {
                "x":      round(float(bb[0]), 2),
                "y":      round(float(bb[1]), 2),
                "width":  round(float(bb[2] - bb[0]), 2),
                "height": round(float(bb[3] - bb[1]), 2),
            },
            "rows": rows,
        })

    return tables_out


# ─────────────────────────────────────────────
#  4.  DIAGRAMS  — vector graphic regions
# ─────────────────────────────────────────────

def _has_drawing_elements(container) -> bool:
    """True if container holds any LTLine / LTRect / LTCurve drawing primitives."""
    for el in container:
        if isinstance(el, (LTLine, LTRect, LTCurve)):
            return True
        if isinstance(el, LTFigure) and _has_drawing_elements(el):
            return True
    return False


def _has_meaningful_text(container, min_chars: int = 20) -> bool:
    """True if the combined text content inside a container exceeds min_chars."""
    total = 0
    for el in container:
        if isinstance(el, LTTextContainer):
            total += len(el.get_text().strip())
            if total >= min_chars:
                return True
        elif isinstance(el, LTFigure):
            if _has_meaningful_text(el, min_chars):
                return True
    return False


def _rasterize_region(fitz_page, bbox_pdf: tuple, page_height: float, dpi: int = 150) -> str | None:
    """
    Clip the given rectangle (x0, y0, x1, y1 in PDF coords, bottom-left origin)
    from the PyMuPDF page and return a base64-encoded PNG string.
    """
    try:
        rect = fitz.Rect(bbox_pdf)
        clip = fitz_page.get_pixmap(clip=rect, dpi=dpi)
        return base64.b64encode(clip.tobytes("png")).decode("utf-8")
    except Exception:
        return None


def detect_diagrams(
    pdfminer_page,
    fitz_page,
    page_height: float,
    page_num: int,
) -> list[dict]:
    """
    Identify diagram regions as LTFigure containers that:
      • contain drawing elements (lines / rects / curves), AND
      • contain little or no readable text  (≤ 20 chars total)

    Each detected region is rasterized to a PNG thumbnail.
    """
    diagrams = []
    idx = 0

    for element in pdfminer_page:
        if not isinstance(element, LTFigure):
            continue

        has_draw = _has_drawing_elements(element)
        has_text = _has_meaningful_text(element)

        if not has_draw or has_text:
            continue          # skip pure-text figures or empty figures

        # pdfminer coords: (x0, y0, x1, y1) — bottom-left origin
        x0, y0, x1, y1 = element.x0, element.y0, element.x1, element.y1
        width  = x1 - x0
        height = y1 - y0

        # Skip tiny artefacts
        if width < 20 or height < 20:
            continue

        # Convert for display (top-left origin)
        display_y = round(float(page_height - y1), 2)

        # Rasterize using PyMuPDF (pass bottom-left coords as-is)
        png_b64 = _rasterize_region(fitz_page, (x0, y0, x1, y1), page_height)

        diagrams.append({
            "diagramId": f"p{page_num}_diag{idx:03d}",
            "type":      "vector",          # always vector-origin for this detector
            "bbox": {
                "x":      round(float(x0), 2),
                "y":      display_y,
                "width":  round(float(width), 2),
                "height": round(float(height), 2),
            },
            "previewBase64": png_b64,       # PNG, may be None if rasterisation failed
            "previewMimeType": "image/png",
        })
        idx += 1

    return diagrams


# ─────────────────────────────────────────────
#  API endpoint
# ─────────────────────────────────────────────

@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        content = await file.read()
        pages_data = []

        # Open with both libraries once
        laparams    = LAParams()
        fitz_doc    = fitz.open(stream=content, filetype="pdf")
        plumber_doc = pdfplumber.open(io.BytesIO(content))

        pdfminer_pages  = list(extract_pages(io.BytesIO(content), laparams=laparams))
        plumber_pages   = plumber_doc.pages

        for page_num, (pm_page, plumb_page) in enumerate(
            zip(pdfminer_pages, plumber_pages), start=1
        ):
            fitz_page = fitz_doc[page_num - 1]

            page_w = float(pm_page.width)
            page_h = float(pm_page.height)

            hlines      = collect_hlines(pm_page)
            text_blocks = extract_text_blocks(pm_page, page_h, page_w, page_num, hlines)
            images      = extract_images_from_page(fitz_page, page_num, page_h)
            tables      = extract_tables_from_page(plumb_page, page_num, page_h)
            diagrams    = detect_diagrams(pm_page, fitz_page, page_h, page_num)

            pages_data.append({
                "page":       page_num,
                "pageWidth":  round(page_w, 1),
                "pageHeight": round(page_h, 1),
                "textBlocks": text_blocks,
                "images":     images,
                "tables":     tables,
                "diagrams":   diagrams,
            })

        fitz_doc.close()
        plumber_doc.close()

        return JSONResponse(content=pages_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/")
async def root():
    return {"message": "PDF Extractor API is running. Use POST /extract-pdf"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)