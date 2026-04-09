from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LTTextContainer, LTTextLine, LTChar, LTFigure, LAParams, LTAnon
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


def extract_text_blocks(container, page_height, page_width, page_num):
    """
    Extract one block per visual line (LTTextLine), not per text box.
    This matches line-level granularity expected in the output.
    Recurses into LTFigure for nested XObjects.
    """
    blocks = []

    for element in container:
        # Recurse into figures
        if isinstance(element, LTFigure):
            nested = extract_text_blocks(element, page_height, page_width, page_num)
            blocks.extend(nested)
            continue

        if not isinstance(element, LTTextContainer):
            continue

        # Process each line individually instead of the whole text box
        for line in element:
            if not isinstance(line, LTTextLine):
                continue

            # Collect chars from this line
            chars = [obj for obj in line if isinstance(obj, LTChar)]
            if not chars:
                continue

            # Build text from chars + LTAnon (spaces/ligatures)
            line_text = ""
            for obj in line:
                if isinstance(obj, LTChar):
                    line_text += obj.get_text()
                elif isinstance(obj, LTAnon):
                    line_text += obj.get_text()

            line_text = line_text.strip()
            if not line_text:
                continue

            font_family, font_size, is_bold, is_italic, color_hex = get_font_info(chars[0])

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

            text_blocks = extract_text_blocks(page_layout, page_h, page_w, page_num)

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