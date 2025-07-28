{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "66a23b10-cfa3-4b46-8ad1-4145fa8fbfc8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Run this Streamlit app with: streamlit run app.py\n",
    "\n",
    "import streamlit as st\n",
    "import docx\n",
    "import tempfile\n",
    "import os\n",
    "import openai\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from docx import Document\n",
    "\n",
    "# === Configuration ===\n",
    "openai.api_key = st.secrets[\"OPENAI_API_KEY\"]  # Add your OpenAI API key to Streamlit Secrets\n",
    "\n",
    "# === Helper Functions ===\n",
    "def extract_text_from_docx(doc_file):\n",
    "    doc = docx.Document(doc_file)\n",
    "    return \"\\n\".join([para.text for para in doc.paragraphs])\n",
    "\n",
    "def compute_ats_score(resume_text, job_desc):\n",
    "    vectorizer = CountVectorizer().fit_transform([resume_text, job_desc])\n",
    "    vectors = vectorizer.toarray()\n",
    "    score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]\n",
    "    return round(score * 100, 2)\n",
    "\n",
    "def tailor_with_llm(resume_text, job_desc):\n",
    "    prompt = f\"\"\"\n",
    "You are a professional resume editor. Modify the following resume to better match the given job description by:\n",
    "- Highlighting relevant skills and experiences\n",
    "- Rewriting descriptions to emphasize job-specific keywords\n",
    "- Keeping formatting and tone professional\n",
    "\n",
    "Resume:\n",
    "{resume_text}\n",
    "\n",
    "Job Description:\n",
    "{job_desc}\n",
    "\n",
    "Tailored Resume:\n",
    "\"\"\"\n",
    "    response = openai.ChatCompletion.create(\n",
    "        model=\"gpt-4\",\n",
    "        messages=[\n",
    "            {\"role\": \"system\", \"content\": \"You are an expert resume optimization assistant.\"},\n",
    "            {\"role\": \"user\", \"content\": prompt},\n",
    "        ],\n",
    "        temperature=0.7\n",
    "    )\n",
    "    return response[\"choices\"][0][\"message\"][\"content\"].strip()\n",
    "\n",
    "def save_as_docx(text):\n",
    "    doc = Document()\n",
    "    for line in text.split('\\n'):\n",
    "        doc.add_paragraph(line)\n",
    "    temp_path = tempfile.mktemp(suffix=\".docx\")\n",
    "    doc.save(temp_path)\n",
    "    return temp_path\n",
    "\n",
    "# === Streamlit App ===\n",
    "st.set_page_config(page_title=\"Resume Tailor AI\", layout=\"wide\")\n",
    "st.title(\"🧠 Resume Tailoring Assistant\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Upload your Resume (DOCX format)\", type=[\"docx\"])\n",
    "job_description = st.text_area(\"Paste the Job Description\", height=200)\n",
    "\n",
    "if uploaded_file and job_description:\n",
    "    with st.spinner(\"Reading resume...\"):\n",
    "        resume_text = extract_text_from_docx(uploaded_file)\n",
    "\n",
    "    pre_score = compute_ats_score(resume_text, job_description)\n",
    "    st.markdown(f\"### ATS Score Before Tailoring: {pre_score} ✅\")\n",
    "\n",
    "    if st.button(\"Tailor My Resume with AI\"):\n",
    "        with st.spinner(\"Tailoring resume using LLM...\"):\n",
    "            tailored_resume = tailor_with_llm(resume_text, job_description)\n",
    "            post_score = compute_ats_score(tailored_resume, job_description)\n",
    "            docx_path = save_as_docx(tailored_resume)\n",
    "\n",
    "        st.markdown(f\"### ATS Score After Tailoring: {post_score} 🚀\")\n",
    "        st.download_button(\"Download Tailored Resume (DOCX)\", open(docx_path, \"rb\"), file_name=\"tailored_resume.docx\")\n",
    "\n",
    "        st.markdown(\"### Tailored Resume Preview\")\n",
    "        st.text_area(\"\", value=tailored_resume, height=500)\n",
    "\n",
    "# Launch instruction\n",
    "st.markdown(\"\\n---\\nTo run this app locally: `streamlit run app.py`\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60da3f7e-5f59-4b63-a200-0e1d04c37ba0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
