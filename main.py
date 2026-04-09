from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LTTextContainer, LTTextLine, LTChar, LTFigure, LAParams
)
import io
import os

app = FastAPI(title="PDF Extractor API")


def get_font_info(char):
    fontname = getattr(char, 'fontname', 'Unknown')
    fontsize = getattr(char, 'size', 10.0)
    lower = fontname.lower()
    is_bold = 'bold' in lower
    is_italic = 'italic' in lower or 'oblique' in lower
    return fontname, round(float(fontsize), 1), is_bold, is_italic, "#000000"


def extract_text_blocks(container, page_height, page_width, block_id_counter, page_num):
    """Recursively extract text blocks from any container (page or LTFigure)."""
    blocks = []
    for element in container:
        # Recurse into figures (nested XObjects)
        if isinstance(element, LTFigure):
            nested = extract_text_blocks(element, page_height, page_width, block_id_counter + len(blocks), page_num)
            blocks.extend(nested)
            continue

        if not isinstance(element, LTTextContainer):
            continue

        text = element.get_text().strip()
        if not text:
            continue

        # Flatten chars: LTTextBox -> LTTextLine -> LTChar
        chars = []
        for line in element:
            if isinstance(line, LTTextLine):
                for obj in line:
                    if isinstance(obj, LTChar):
                        chars.append(obj)

        if chars:
            font_family, font_size, is_bold, is_italic, color_hex = get_font_info(chars[0])
        else:
            font_family, font_size, is_bold, is_italic, color_hex = "Unknown", 10.0, False, False, "#000000"

        alignment = "left"
        element_center = element.x0 + element.width / 2
        if abs(element_center - page_width / 2) < 50:
            alignment = "center"

        idx = block_id_counter + len(blocks)
        blocks.append({
            "text": text,
            "x": round(float(element.x0), 2),
            "y": round(float(page_height - element.y1), 2),
            "width": round(float(element.width), 2),
            "height": round(float(element.height), 2),
            "fontSize": font_size,
            "fontFamily": font_family,
            "colorHex": color_hex,
            "isBold": is_bold,
            "isItalic": is_italic,
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

            text_blocks = extract_text_blocks(page_layout, page_h, page_w, 0, page_num)

            pages_data.append({
                "page": page_num,
                "pageWidth": round(page_w, 1),
                "pageHeight": round(page_h, 1),
                "textBlocks": text_blocks
            })

        return JSONResponse(content=pages_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


# ── DEBUG endpoint ──────────────────────────────────────────────────────────────
@app.post("/debug-pdf")
async def debug_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    try:
        content = await file.read()
        report = []

        for page_num, page_layout in enumerate(extract_pages(io.BytesIO(content)), start=1):
            page_info = {
                "page": page_num,
                "pageWidth": round(float(page_layout.width), 1),
                "pageHeight": round(float(page_layout.height), 1),
                "elements": []
            }
            for element in page_layout:
                elem_type = type(element).__name__
                raw_text = element.get_text().strip() if isinstance(element, LTTextContainer) else ""
                page_info["elements"].append({
                    "type": elem_type,
                    "isTextContainer": isinstance(element, LTTextContainer),
                    "isFigure": isinstance(element, LTFigure),
                    "textPreview": raw_text[:80],
                    "x0": round(float(element.x0), 2),
                    "y0": round(float(element.y0), 2),
                })
            report.append(page_info)

        return JSONResponse(content=report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/")
async def root():
    return {"message": "PDF Extractor API is running. Use POST /extract-pdf"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)