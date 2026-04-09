from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LTTextContainer, LTTextLine, LTChar, LTFigure, LAParams, LTLine
)
import io
import os

app = FastAPI(title="PDF Extractor API")


def clean_font_name(fontname: str) -> str:
    """Strip subset prefix like 'BAAAAA+' from embedded font names."""
    if '+' in fontname:
        return fontname.split('+', 1)[1]
    return fontname


def get_font_info(char):
    fontname = clean_font_name(getattr(char, 'fontname', 'Unknown'))
    fontsize = getattr(char, 'size', 10.0)
    lower = fontname.lower()
    is_bold = 'bold' in lower
    is_italic = 'italic' in lower or 'oblique' in lower
    return fontname, round(float(fontsize), 1), is_bold, is_italic, "#000000"


def collect_hlines(container):
    """
    Collect all horizontal LTLine elements from a container (page or figure).
    Returns list of (x0, x1, y) tuples.
    """
    hlines = []
    for el in container:
        if isinstance(el, LTLine):
            # Horizontal = y0 and y1 are nearly equal
            if abs(el.y1 - el.y0) < 2.0:
                hlines.append((el.x0, el.x1, el.y0))
        elif isinstance(el, LTFigure):
            hlines.extend(collect_hlines(el))
    return hlines


def is_underlined(line, hlines, tolerance=3.0):
    """
    Check if a text line has an underline by looking for a horizontal LTLine
    that sits just below the text baseline (within `tolerance` units).
    """
    text_x0 = line.x0
    text_x1 = line.x1
    text_y0 = line.y0  # bottom of text line (baseline area)

    for hx0, hx1, hy in hlines:
        # Must be below the text and within tolerance
        y_diff = text_y0 - hy
        if 0 <= y_diff <= tolerance:
            # Must horizontally overlap with the text
            overlap = min(text_x1, hx1) - max(text_x0, hx0)
            if overlap > 5.0:
                return True
    return False


def is_bullet_line(line):
    """
    Detect bullet point lines — Symbol font lines whose glyph
    renders as whitespace/empty after get_text().
    """
    chars = [obj for obj in line if isinstance(obj, LLChar := LTChar)]
    if not chars:
        return False
    font = clean_font_name(getattr(chars[0], 'fontname', ''))
    text = line.get_text().strip()
    return font == 'Symbol' and not text


def extract_text_blocks(container, page_height, page_width, page_num, hlines):
    """
    Extract one block per visual line (LTTextLine).
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

            line_text = line.get_text().strip()
            font_family, font_size, is_bold, is_italic, color_hex = get_font_info(chars[0])

            # Handle bullet glyphs: Symbol font with no readable text
            is_bullet = font_family == 'Symbol' and not line_text
            if is_bullet:
                line_text = ""  # keep as empty bullet marker

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
                "continuedToId": None
            })

    return blocks


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    try:
        content = await file.read()
        pages_data = []

        laparams = LAParams()
        for page_num, page_layout in enumerate(extract_pages(io.BytesIO(content), laparams=laparams), start=1):
            page_w = float(page_layout.width)
            page_h = float(page_layout.height)

            # Collect horizontal lines for underline detection
            hlines = collect_hlines(page_layout)

            text_blocks = extract_text_blocks(page_layout, page_h, page_w, page_num, hlines)

            pages_data.append({
                "page": page_num,
                "pageWidth": round(page_w, 1),
                "pageHeight": round(page_h, 1),
                "textBlocks": text_blocks
            })

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