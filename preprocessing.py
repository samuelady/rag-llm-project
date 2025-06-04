import pdfplumber
import re
import unicodedata
from pathlib import Path
from collections import Counter

# --- Helper Functions (used internally by run_preprocessing) ---
def extract_text_from_pdf(pdf_path: Path) -> list[str]:
    """
    Extract raw text from each page of a PDF.
    Returns a list of page strings.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages

def detect_header_footer(pages: list[str]) -> tuple[str, str]:
    """
    Detect the most common first and last lines across pages as header/footer.
    """
    headers = [p.split('\n')[0].strip() for p in pages if p]
    footers = [p.split('\n')[-1].strip() for p in pages if p]
    header = Counter(headers).most_common(1)[0][0] if headers else ""
    footer = Counter(footers).most_common(1)[0][0] if footers else ""
    return header, footer

def clean_page(text: str, header: str, footer: str) -> str:
    """
    Remove header/footer, normalize whitespace, drop page numbers, fix ligatures.
    """
    lines = text.split('\n')
    if lines and lines[0].strip() == header:
        lines = lines[1:]
    if lines and lines[-1].strip() == footer:
        lines = lines[:-1]
    cleaned = []
    for line in lines:
        line = re.sub(r'\s+', ' ', line).strip()
        if re.fullmatch(r'(Page\\s*)?\\d+', line):
            continue
        cleaned.append(line)
    text_clean = ' '.join(cleaned)
    text_clean = text_clean.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
    return unicodedata.normalize('NFKC', text_clean)

# --- Main Preprocessing Function ---
def preprocess_pdf_to_text(pdf_input_path: Path, txt_output_path: Path):
    """
    Full pipeline for one PDF: extract, detect, clean, and save as .txt.
    """
    pages = extract_text_from_pdf(pdf_input_path)
    header, footer = detect_header_footer(pages)
    cleaned = [clean_page(p, header, footer) for p in pages]
    txt_output_path.write_text("\\n".join(cleaned), encoding='utf-8')

# --- Orchestration Function for Preprocessing Module ---
def run_preprocessing(input_dir: Path, output_dir: Path):
    """
    Processes all PDFs in input_dir, outputs cleaned .txt files in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}")
    for pdf_file_path in pdf_files: # Renamed variable for clarity
        out_file = output_dir / f"{pdf_file_path.stem}.txt"
        print(f"Processing {pdf_file_path.name} \\u2192 {out_file.name}")
        # The fix is here: Call preprocess_pdf_to_text with the correct arguments
        preprocess_pdf_to_text(pdf_file_path, out_file)
    print("Done preprocessing.")

if __name__ == "__main__":
    # This block runs only if the script is executed directly
    from config import DOCUMENTS_DIR, PROCESSED_DOCS_DIR
    print("Running preprocessing module directly (for testing)...")
    run_preprocessing(DOCUMENTS_DIR, PROCESSED_DOCS_DIR)