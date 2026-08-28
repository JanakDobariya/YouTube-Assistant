from __future__ import annotations

import os
import re

import numpy as np
import streamlit as st
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi


st.set_page_config(page_title="YouTube Assistant", page_icon="🎬", layout="centered")

GROQ_MODEL = "openai/gpt-oss-120b"


def get_groq_client() -> Groq:
    try:
        streamlit_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        streamlit_api_key = None
    api_key = os.getenv("GROQ_API_KEY") or streamlit_api_key
    if not api_key:
        st.error("GROQ_API_KEY is not configured. Add it to the environment or Streamlit secrets.")
        st.stop()
    return Groq(api_key=api_key)


def groq_invoke(prompt: str) -> str:
    try:
        completion = get_groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content or ""
    except Exception:
        st.error("The language-model request failed. Check the API configuration and try again.")
        return ""


def extract_video_id(url_or_id: str) -> str | None:
    patterns = [
        r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/)|youtu\.be/)([0-9A-Za-z_-]{11})",
        r"^([0-9A-Za-z_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id.strip())
        if match:
            return match.group(1)
    return None


def fetch_transcript(video_id: str) -> str:
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    return " ".join(snippet.text for snippet in transcript).strip()


def split_text(text: str, max_words: int = 500) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def summarize_transcript(transcript: str) -> str:
    chunk_summaries = []
    for chunk in split_text(transcript):
        prompt = (
            "Summarize this YouTube transcript section in two to four factual sentences. "
            "Do not add information that is absent from the transcript.\n\n"
            f"Transcript:\n{chunk}"
        )
        summary = groq_invoke(prompt)
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return ""

    return groq_invoke(
        "Combine the following section summaries into one coherent 100-200 word summary. "
        "Preserve uncertainty and do not invent facts.\n\n" + "\n\n".join(chunk_summaries)
    )


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def relevant_context(transcript: str, question: str, count: int = 4) -> str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(transcript)
    embeddings = get_embeddings()
    chunk_vectors = np.asarray(embeddings.embed_documents(chunks), dtype=np.float32)
    question_vector = np.asarray(embeddings.embed_query(question), dtype=np.float32)

    chunk_norms = np.linalg.norm(chunk_vectors, axis=1)
    question_norm = np.linalg.norm(question_vector)
    denominator = np.maximum(chunk_norms * question_norm, 1e-12)
    scores = (chunk_vectors @ question_vector) / denominator
    best_indices = np.argsort(scores)[-count:][::-1]
    return "\n\n".join(chunks[index] for index in best_indices)


st.title("YouTube Video Assistant")
st.caption("Read an English transcript, create a summary, or ask transcript-grounded questions.")

video_url = st.text_input("YouTube video URL or ID")
option = st.radio("Task", ("Transcript", "Summarization", "Q&A"), horizontal=True)

question = ""
answer_length = "Short"
if option == "Q&A":
    question = st.text_input("Question about the video")
    answer_length = st.selectbox("Answer length", ["Short", "Medium", "Long"])

if st.button("Process", type="primary"):
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("Enter a valid YouTube URL or 11-character video ID.")
        st.stop()

    try:
        transcript = fetch_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound):
        st.error("An English transcript is not available for this video.")
        st.stop()
    except Exception:
        st.error("The transcript could not be retrieved. YouTube may be blocking this request temporarily.")
        st.stop()

    if not transcript:
        st.error("The retrieved transcript is empty.")
        st.stop()

    if option == "Transcript":
        st.header("Transcript")
        st.write(transcript)

    elif option == "Summarization":
        st.header("Summary")
        with st.spinner("Summarizing the transcript..."):
            summary = summarize_transcript(transcript)
        if summary:
            st.markdown(summary)

    else:
        if not question.strip():
            st.warning("Enter a question first.")
            st.stop()

        length_instruction = {
            "Short": "Answer in two or three concise sentences.",
            "Medium": "Answer in five to eight sentences.",
            "Long": "Give a detailed answer of at least 150 words when the evidence supports it.",
        }[answer_length]

        with st.spinner("Searching the transcript..."):
            context = relevant_context(transcript, question)
            prompt = (
                "Answer only from the transcript context below. If the context does not contain the answer, "
                "say: I don't know based on this transcript.\n\n"
                f"{length_instruction}\n\nContext:\n{context}\n\nQuestion:\n{question}"
            )
            answer = groq_invoke(prompt)

        if answer:
            st.header("Answer")
            st.markdown(answer)
