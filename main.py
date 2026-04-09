from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LTTextContainer, LTTextLine, LTChar,
    LTTextBox, LTAnon, LTFigure, LTImage, LTRect, LTLine
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
    color = "#000000"
    return fontname, round(float(fontsize), 1), is_bold, is_italic, color


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    try:
        content = await file.read()
        pages_data = []

        for page_num, page_layout in enumerate(extract_pages(io.BytesIO(content)), start=1):
            text_blocks = []
            block_id_counter = 0

            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if not text:
                        continue

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
                    if abs(element_center - page_layout.width / 2) < 50:
                        alignment = "center"

                    block = {
                        "text": text,
                        "x": round(float(element.x0), 2),
                        "y": round(float(page_layout.height - element.y1), 2),
                        "width": round(float(element.width), 2),
                        "height": round(float(element.height), 2),
                        "fontSize": font_size,
                        "fontFamily": font_family,
                        "colorHex": color_hex,
                        "isBold": is_bold,
                        "isItalic": is_italic,
                        "blockId": f"p{page_num}_b{block_id_counter:03d}",
                        "flowGroup": "body",
                        "alignment": alignment,
                        "paragraphSpacing": 6.0,
                        "isFixedPosition": False,
                        "continuedFromId": None,
                        "continuedToId": None
                    }
                    text_blocks.append(block)
                    block_id_counter += 1

            pages_data.append({
                "page": page_num,
                "pageWidth": round(float(page_layout.width), 1),
                "pageHeight": round(float(page_layout.height), 1),
                "textBlocks": text_blocks
            })

        return JSONResponse(content=pages_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


# ── DEBUG endpoint ──────────────────────────────────────────────────────────────
# Call this first to see exactly what pdfminer finds in your PDF.
# It reports: element types, raw text, and char count per element.
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
                raw_text = ""
                text_stripped = ""
                char_count = 0
                line_count = 0

                if isinstance(element, LTTextContainer):
                    raw_text = element.get_text()
                    text_stripped = raw_text.strip()
                    for line in element:
                        if isinstance(line, LTTextLine):
                            line_count += 1
                            for obj in line:
                                if isinstance(obj, LTChar):
                                    char_count += 1

                page_info["elements"].append({
                    "type": elem_type,
                    "isTextContainer": isinstance(element, LTTextContainer),
                    "rawTextLength": len(raw_text),
                    "strippedTextLength": len(text_stripped),
                    "strippedTextPreview": text_stripped[:80] if text_stripped else "",
                    "lineCount": line_count,
                    "charCount": char_count,
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