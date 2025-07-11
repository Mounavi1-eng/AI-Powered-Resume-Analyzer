from fastapi import FastAPI, UploadFile, File
import spacy
from pdfminer.high_level import extract_text
import docx
import openai
import os

app = FastAPI()
nlp = spacy.load("en_core_web_sm")
openai.api_key = os.getenv("OPENAI_KEY")  # Set your API key

def extract_text_from_file(file: UploadFile):
    if file.filename.endswith(".pdf"):
        text = extract_text(file.file)
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file.file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")
    return text

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...), job_desc: str = ""):
    resume_text = extract_text_from_file(file)
    
    # Extract skills using spaCy
    doc = nlp(resume_text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    
    # OpenAI for ATS feedback (example)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a resume ATS optimizer."},
            {"role": "user", "content": f"Resume: {resume_text}\n\nJob Description: {job_desc}\n\nGive 3 tips to improve this resume for ATS."}
        ]
    )
    
    return {
        "skills": skills,
        "ats_feedback": response.choices[0].message["content"]
    }

# Run: uvicorn main:app --reload
