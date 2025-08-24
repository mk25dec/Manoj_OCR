**Extract text & structured info from PDFs (vector or scanned) and images.**

**Features:**

- Reads same config.toml as other scripts (output_result, logs)
 - Uses PyMuPDF/pdfplumber for vector PDFs
 - Uses pytesseract + OpenCV preprocessing for scanned PDFs/images
 - Extracts raw text, emails, phones, URLs via regex
 - Optionally extracts tables (pdfplumber) to CSV
 - Saves outputs (txt/json) and logs everything


$ source /Users/manoj/coding/Env/venv-python/bin/activate
cd /Users/manoj/coding/scripts
(venv-python) $
(venv-python) $ python /Users/manoj/coding/scripts/prod/ocr.py -h
usage: ocr.py [-h] -i INPUT [INPUT ...] [--mode {auto,vector,ocr}]
              [--out-formats {txt,json} [{txt,json} ...]] [--tables] [--dpi DPI] [--auto-preprocess]
              [--ocr-lang OCR_LANG]

Extract text and structured data from PDFs/images (vector or scanned).

options:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        Input file(s) or glob pattern(s). Example: -i file.pdf or -i '/path/*.pdf'
                        '/path/*.png'
  --mode {auto,vector,ocr}
                        auto: try vector then OCR; vector: extract text from PDF; ocr: force OCR
                        (PDFs only)
  --out-formats {txt,json} [{txt,json} ...]
                        Which output formats to produce
  --tables              Attempt to extract tables (requires pdfplumber + pandas, PDFs only)
  --dpi DPI             DPI for rasterization/OCR (default 300 in auto/ocr, PDFs only)
  --auto-preprocess     Apply OpenCV preprocessing before OCR (deskew, denoise, contrast)
  --ocr-lang OCR_LANG   Tesseract OCR language code (default: eng)
(venv-python) $
