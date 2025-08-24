#!/usr/bin/env python3
"""
Extract text & structured info from PDFs (vector or scanned) and images.

Features:
 - Reads same config.toml as other scripts (output_result, logs)
 - Uses PyMuPDF/pdfplumber for vector PDFs
 - Uses pytesseract + OpenCV preprocessing for scanned PDFs/images
 - Extracts raw text, emails, phones, URLs via regex
 - Optionally extracts tables (pdfplumber) to CSV
 - Saves outputs (txt/json) and logs everything
"""

import os
import sys
import glob
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# Config path (same as your other scripts)
CONFIG_PATH = "/Users/manoj/coding/x_config/config.toml"

# Load config first to get library path
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

import toml
config = toml.load(CONFIG_PATH)

# Add library path to sys.path
if "output_path" in config and "lib" in config["output_path"]:
    lib_path = config["output_path"]["lib"]
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

# Import the logger module
try:
    from logger import ScriptLogger
    HAS_LOGGER_MODULE = True
except ImportError:
    # Fallback to basic logging if logger module is not available
    import logging
    HAS_LOGGER_MODULE = False
    
    # Create a simple fallback logger class
    class ScriptLogger:
        def __init__(self, script_name, logs_path):
            self.script_name = script_name
            self.logs_path = logs_path
            os.makedirs(logs_path, exist_ok=True)
            log_file = os.path.join(logs_path, f"{script_name}.log")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(log_file, mode="a", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            self.logger = logging.getLogger(script_name)
        
        def initialize(self, **kwargs):
            return self
        
        def start_execution(self, command_line_args):
            self.logger.info("="*80)
            self.logger.info(f"NEW RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Command line: {' '.join(command_line_args)}")
            self.logger.info("="*80)
        
        def log_error(self, error_message, exception=None):
            if exception:
                self.logger.error("%s: %s", error_message, str(exception), exc_info=True)
            else:
                self.logger.error(error_message)
        
        def log_warning(self, warning_message):
            self.logger.warning(warning_message)
        
        def log_info(self, info_message):
            self.logger.info(info_message)

# Import other required packages
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import cv2
import numpy as np
import pandas as pd

# Simple regex patterns for structured extraction
RE_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
RE_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
RE_URL   = re.compile(r"(?:https?://|www\.)[^\s'\"<>]+")

# Supported image formats for OCR
SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp'}

# Helper: load config and get paths
def load_config(config_path: str = CONFIG_PATH):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = toml.load(config_path)
    if "output_path" not in cfg:
        raise KeyError(f"[output_path] section missing in config file {config_path}.")
    paths = cfg["output_path"]
    required_keys = ["config", "logs", "output_result", "tmp"]
    for k in required_keys:
        if k not in paths:
            raise KeyError(f"Missing '{k}' in [output_path] of {config_path}.")
        os.makedirs(paths[k], exist_ok=True)
    return paths

# Detect whether a PDF has extractable (searchable) text using PyMuPDF or pdfplumber
def is_searchable_pdf(path: str) -> bool:
    try:
        doc = fitz.open(path)
        text_len = 0
        for page in doc:
            text_len += len(page.get_text("text") or "")
            if text_len > 50:  # heuristic: some text found
                doc.close()
                return True
        doc.close()
        return False
    except Exception:
        try:
            with pdfplumber.open(path) as pdf:
                text_len = sum(len(p.extract_text() or "") for p in pdf.pages[:3])
                return text_len > 50
        except Exception:
            return False

# Vector extraction: use PyMuPDF (fitz) primarily, fallback to pdfplumber
def extract_text_vector_pdf(path: str) -> str:
    try:
        doc = fitz.open(path)
        texts = []
        for page in doc:
            texts.append(page.get_text("text") or "")
        doc.close()
        return "\n".join(texts).strip()
    except Exception:
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages).strip()

# Table extraction using pdfplumber
def extract_tables_pdf(path: str) -> List[pd.DataFrame]:
    tables = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                try:
                    df = pd.DataFrame(table[1:], columns=table[0])
                except Exception:
                    # fallback: raw table -> DataFrame without header
                    df = pd.DataFrame(table)
                tables.append(df)
    return tables

# OpenCV preprocessing for OCR: deskew, denoise, contrast/brightness, adaptive threshold
def preprocess_image_for_ocr(pil_image: Image.Image,
                             deskew: bool = True,
                             denoise: bool = True,
                             brightness: Optional[float] = None,
                             contrast: Optional[float] = None,
                             threshold: bool = False) -> Image.Image:
    # Convert to grayscale and numpy
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Denoise
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Deskew: compute angle of text by getting edges and Hough lines or moments
    if deskew:
        coords = np.column_stack(np.where(gray < 250))
        if coords.size:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = gray.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Contrast / brightness adjustments using simple scaling
    if contrast is not None or brightness is not None:
        alpha = float(contrast) if contrast else 1.0  # contrast
        beta = int((brightness - 1.0) * 50) if brightness else 0  # simplistic
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Threshold (adaptive) - good for text-only pages
    if threshold:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 11)

    # Convert back to PIL
    processed = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    return processed

# OCR single image using pytesseract
def ocr_image(pil_image: Image.Image, lang: str = "eng", config: str = "") -> str:
    text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
    return text

# OCR raster PDF: convert pages to images then OCR
def ocr_pdf(path: str, dpi: int = 300, auto_preprocess: bool = True,
            lang: str = "eng", ocr_config: str = "") -> str:
    # Use pdfplumber to render images if available; else use pdf2image via subprocess? We'll use pdfplumber images if possible.
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                # get PIL image of page at given dpi
                pil_img = page.to_image(resolution=dpi).original
                if auto_preprocess:
                    pil_img = preprocess_image_for_ocr(pil_img, deskew=True, denoise=True, contrast=None, brightness=None, threshold=False)
                texts.append(ocr_image(pil_img, lang=lang, config=ocr_config))
    except Exception:
        # fallback: use PyMuPDF to render
        doc = fitz.open(path)
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            mode = "RGB" if pix.alpha == 0 else "RGBA"
            pil_img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            if auto_preprocess:
                pil_img = preprocess_image_for_ocr(pil_img, deskew=True, denoise=True)
            texts.append(ocr_image(pil_img, lang=lang, config=ocr_config))
        doc.close()
    return "\n".join(texts).strip()

# OCR image file: load and process image directly
def ocr_image_file(path: str, auto_preprocess: bool = True,
                   lang: str = "eng", ocr_config: str = "") -> str:
    # Open the image
    pil_img = Image.open(path)
    
    # Preprocess if requested
    if auto_preprocess:
        pil_img = preprocess_image_for_ocr(pil_img, deskew=True, denoise=True)
    
    # Perform OCR
    text = ocr_image(pil_img, lang=lang, config=ocr_config)
    return text

# Extract structured entities using regex
def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    emails = sorted(set(RE_EMAIL.findall(text)))
    phones = sorted(set(RE_PHONE.findall(text)))
    urls = sorted(set(RE_URL.findall(text)))
    return {"emails": emails, "phones": phones, "urls": urls}

# Save outputs
def save_text(output_folder: str, base_name: str, text: str) -> str:
    txt_path = os.path.join(output_folder, f"{base_name}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    return txt_path

def save_json(output_folder: str, base_name: str, data: Any) -> str:
    path = os.path.join(output_folder, f"{base_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path

def save_tables(output_folder: str, base_name: str, tables: List[Any]) -> List[str]:
    saved = []
    for idx, df in enumerate(tables):
        csv_path = os.path.join(output_folder, f"{base_name}_table_{idx+1}.csv")
        df.to_csv(csv_path, index=False)
        saved.append(csv_path)
    return saved

# Check if file is an image based on extension
def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_IMAGE_FORMATS

# Main processor class
class Extractor:
    def __init__(self, paths_cfg: Dict[str,str], args):
        self.paths = paths_cfg
        self.args = args
        self.script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        
        # Use the centralized logger
        self.logger = ScriptLogger(self.script_name, self.paths["logs"]).initialize()
        self.logger.start_execution(sys.argv)

    def process_path_list(self, input_patterns: List[str]):
        files = []
        for patt in input_patterns:
            files.extend(sorted(glob.glob(patt)))
        if not files:
            self.logger.log_error("No files matched input patterns.")
            return
        for p in files:
            try:
                self.process_file(p)
            except Exception as e:
                self.logger.log_error(f"Failed processing {p}", e)

    def process_file(self, path: str):
        base = os.path.splitext(os.path.basename(path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{base}_{timestamp}"
        self.logger.log_info(f"Processing: {path}")

        # Choose mode
        mode = self.args.mode.lower()
        raw_text = ""
        tables = []

        # Check if file is an image
        if is_image_file(path):
            self.logger.log_info("Detected image file - performing OCR.")
            raw_text = ocr_image_file(path, auto_preprocess=self.args.auto_preprocess, 
                                     lang=self.args.ocr_lang or "eng")
        # If it's a PDF
        elif path.lower().endswith(".pdf"):
            # If mode==auto, try vector first
            if mode == "auto":
                if is_searchable_pdf(path):
                    self.logger.log_info("Detected searchable (vector) PDF - extracting text directly.")
                    raw_text = extract_text_vector_pdf(path)
                    # optionally extract tables
                    if self.args.tables:
                        try:
                            tables = extract_tables_pdf(path)
                            self.logger.log_info(f"Extracted {len(tables)} tables.")
                        except Exception as e:
                            self.logger.log_error("Table extraction failed.", e)
                else:
                    self.logger.log_info("Performing OCR on raster/scanned PDF.")
                    raw_text = ocr_pdf(path, dpi=self.args.dpi or 300, 
                                      auto_preprocess=self.args.auto_preprocess, 
                                      lang=self.args.ocr_lang or "eng")
            elif mode == "vector":
                raw_text = extract_text_vector_pdf(path)
                if self.args.tables:
                    tables = extract_tables_pdf(path)
            elif mode == "ocr":
                raw_text = ocr_pdf(path, dpi=self.args.dpi or 300, 
                                  auto_preprocess=self.args.auto_preprocess, 
                                  lang=self.args.ocr_lang or "eng")
            else:
                raise ValueError("Unknown mode: " + mode)
        else:
            self.logger.log_warning(f"Unsupported file format: {path}")
            return

        # Structured extraction
        entities = extract_entities_from_text(raw_text)

        # Save outputs
        out_dir = self.paths["output_result"]
        txt_path = None
        json_path = None
        tables_paths = []

        if "txt" in self.args.out_formats:
            txt_path = save_text(out_dir, base, raw_text)
            self.logger.log_info(f"Saved text to {txt_path}")
        if "json" in self.args.out_formats:
            payload = {"source_file": path, "extracted_text": raw_text, "entities": entities}
            json_path = save_json(out_dir, base, payload)
            self.logger.log_info(f"Saved json to {json_path}")
        if self.args.tables and tables:
            tables_paths = save_tables(out_dir, base, tables)
            self.logger.log_info(f"Saved {len(tables_paths)} table(s).")

        self.logger.log_info(f"Extraction complete for {path} — emails:{len(entities['emails'])} phones:{len(entities['phones'])} urls:{len(entities['urls'])}")

# CLI
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract text and structured data from PDFs/images (vector or scanned).")
    parser.add_argument("-i", "--input", required=True, nargs="+", help="Input file(s) or glob pattern(s). Example: -i file.pdf or -i '/path/*.pdf' '/path/*.png'")
    parser.add_argument("--mode", choices=["auto","vector","ocr"], default="auto", help="auto: try vector then OCR; vector: extract text from PDF; ocr: force OCR (PDFs only)")
    parser.add_argument("--out-formats", nargs="+", default=["txt","json"], choices=["txt","json"], help="Which output formats to produce")
    parser.add_argument("--tables", action="store_true", help="Attempt to extract tables (requires pdfplumber + pandas, PDFs only)")
    parser.add_argument("--dpi", type=int, help="DPI for rasterization/OCR (default 300 in auto/ocr, PDFs only)")
    parser.add_argument("--auto-preprocess", action="store_true", help="Apply OpenCV preprocessing before OCR (deskew, denoise, contrast)")
    parser.add_argument("--ocr-lang", help="Tesseract OCR language code (default: eng)")
    args = parser.parse_args()

    # load config + setup extractor
    paths = load_config(CONFIG_PATH)
    extractor = Extractor(paths, args)
    extractor.process_path_list(args.input)

if __name__ == "__main__":
    main()