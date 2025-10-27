import streamlit as st
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import io
import re
import google.generativeai as genai
import numpy as np

# --- Configuration ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(['en'], gpu=False)

# --- Functions ---

def extract_text_from_pdf(pdf_bytes, max_pages=20):
    """
    Extracts text from a PDF, using OCR if direct text extraction is minimal.
    Limits OCR to first `max_pages` for memory efficiency.
    """
    all_text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(min(doc.page_count, max_pages)):
            page = doc.load_page(page_num)
            text = page.get_text("text")

            # Use OCR if text is too small
            if len(text.strip()) < 500 and not text.strip().lower().startswith("s=3"):
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # downscale for memory
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_result = ocr_reader.readtext(np.array(img))
                ocr_text = " ".join([t[1] for t in ocr_result])
                all_text += ocr_text + "\n"
            else:
                all_text += text + "\n"
        doc.close()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None
    return all_text

def clean_extracted_text(text):
    """Basic cleaning for LLM input."""
    if not text:
        return ""
    text = re.sub(r'(S=3|S31) प्रतिभूति और विनिमय बोर्ड\nSecurities and Exchange Board of India', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'Issued on: [A-Za-z]+ \d{1,2}, \d{4}', '', text)
    text = re.sub(r'Yours faithfully,', '', text)
    text = re.sub(r'\*{3,}', '', text)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    return text.strip()

def get_llm_analysis(document_text):
    """Sends the cleaned document text to Gemini for structured analysis."""
    if not document_text:
        return "No text extracted from PDF for analysis."

    prompt = f"""
    • The review draft must begin by classifying the consultation paper under the most
appropriate market division: Primary Markets, Secondary Markets, Commodity Markets, or
External Markets
1. Background / Regulatory Context/ Introduction
• Existing framework, evolution, pain point triggering reform
• Legislative / circular / regulatory history
2. Summary of Key Proposals Only include what SEBI has proposed
• Neutral narration, not opinion
3. Critical Analysis of the Proposals
The core of the review, structured proposal-by-proposal as they appear in the CP.
*For each distinct proposal in the CP (e.g., Proposal 1, Proposal 2...), use the following sub-
structure:
• Proposal Number: Title of Proposal from CP
• Concept Proposed: A clear, neutral summary of what SEBI is proposing.
• SEBI&#39;s Rationale: The reasoning and policy objectives behind the proposal as stated by
SEBI.
• Global Benchmarking: Analysis of how similar issues are regulated in key international
jurisdictions (e.g., US SEC, UK FCA, EU ESMA, Singapore MAS, Hong Kong
HKEX/SFC). (The global benchmarking should be country specific, and take only those
jurisdiction which suits India financial market) (Provide references link for this part only)
• Critical Assessment &amp; Recommendations:
o Our Stance: Clearly state the team&#39;s position (e.g., Accepted, Accepted with
Modification, Not Accepted).
o Supporting Rationale: Justification for the stance. Why is it good/bad? What are the
potential impacts?
o Proposed Modifications/Safeguards: If accepted with modification, provide specific,
actionable alternative language or suggestions (e.g., phased implementation, different
thresholds, added anti-avoidance clauses).

4. Conclusion and Overall Recommendations
5. List down the 5 relevant questions that the Finance Ministry should ask the regulator
(SEBI) about this consultation paper. (The questions must be critical, that would be
useful in enhancing the structure of the Indian market).
    {document_text}
    ---

    Ensure the output strictly adheres to the requested  main headings.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"

# --- Streamlit UI ---

st.set_page_config(page_title="SEBI PDF Analyzer", layout="centered")
st.title("SEBI Circular / Consultation Paper Analyzer")
st.markdown("Upload an official SEBI document (PDF) to extract, analyze, and provide comments.")

uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()  # Read once

    if "extracted_text" not in st.session_state:
        with st.spinner("🔍 Extracting text from PDF..."):
            st.session_state.extracted_text = extract_text_from_pdf(pdf_bytes)

    if st.session_state.extracted_text:
        cleaned_text = clean_extracted_text(st.session_state.extracted_text)
        st.success("Text extraction complete!")

        if st.button("Analyze with AI"):
            with st.spinner("Generating structured analysis..."):
                llm_analysis = get_llm_analysis(cleaned_text)
            st.markdown("## AI Analysis")
            st.markdown(llm_analysis)
    else:
        st.error("Failed to extract text from the uploaded PDF.")
else:
    st.info("Please upload a PDF to begin analysis.")
