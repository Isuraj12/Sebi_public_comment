import streamlit as st
import fitz  
import easyocr
from PIL import Image
import io
import re
import google.generativeai as genai

# --- Configuration ---

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])



model = genai.GenerativeModel("gemini-2.5-flash")


ocr_reader = easyocr.Reader(['en'], gpu=False)

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF, using OCR if direct text extraction is minimal.
    """
    all_text = ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text")

            # If the text is too small, use OCR
            if len(text.strip()) < 500 and not text.strip().lower().startswith("s=3"):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                ocr_result = ocr_reader.readtext(img)
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
    """
    Sends the cleaned document text to Gemini for structured analysis.
    """
    if not document_text:
        return "No text extracted from PDF for analysis."

    prompt = f"""
    Analyze the following official document (likely a consultation paper or draft circular from SEBI - Securities and Exchange Board of India).

    Please provide a concise summary in the following structure, identifying the key sections:

    ### Background & Objective
    [Summarize the background and the main objectives of the document. Keep it factual and directly based on the document's content.]

    ### Key Provisions/Proposals
    [List and briefly explain the main proposals or key provisions outlined in the document. Use bullet points or numbered lists if appropriate for clarity. Focus on the core changes or requirements.]

    ### Our Opinion
    [Provide an informed opinion or analysis of the document's proposals. Discuss potential positive implications, benefits, or significance for the stakeholders (e.g., investors, market participants, regulatory transparency). This should be a synthesized perspective based on the document's intent and common regulatory goals.]

    ---
    Document Text:
    {document_text}
    ---

    Ensure the output strictly adheres to the requested three main headings.
    
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"


# --- Streamlit UI ---

st.set_page_config(page_title="SEBI PDF Analyzer", layout="centered")

st.title("SEBI Circular / Consultation Paper Analyzer")
st.markdown("Upload an official SEBI document (PDF) to extract, analyze, and provide comments .")

uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_pdf:
    with st.spinner("🔍 Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_pdf)

    if extracted_text:
        cleaned_text = clean_extracted_text(extracted_text)
        st.success("Text extraction complete!")

        

        if st.button("Analyze with AI"):
            with st.spinner("Generating structured analysis..."):
                llm_analysis = get_llm_analysis(cleaned_text)

            st.markdown("## AI Analysis")
            st.markdown(llm_analysis)
    else:
        st.error(" Failed to extract text from the uploaded PDF.")
else:
    st.info(" Please upload a PDF to begin analysis.")
