import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
import re

# --- Environment & Groq Setup ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not set. Please add it to your .env file or Streamlit secrets.")
    st.stop()
groq_client = Groq(api_key=groq_api_key)

def groq_invoke(messages, model="llama3-70b-8192"):
    formatted_messages = []
    for i, m in enumerate(messages):
        role = "user" if i % 2 == 0 else "assistant"
        formatted_messages.append({"role": role, "content": m})
    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=formatted_messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"GROQ API Error: {e}")
        return "An error occurred with the LLM API."

def extract_video_id(url_or_id):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None

# --- Helper for Chunked Summarization ---
def split_text(text, max_words=500):
    """Split transcript into chunks with max_words each."""
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def summarize_transcript(transcript):
    """Summarize transcript in chunks, then merge summaries."""
    chunks = split_text(transcript, max_words=500)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_prompt = (
            f"You are a helpful assistant. Summarize this YouTube video transcript chunk in 2-4 sentences.\n\nTranscript:\n{chunk}\n\nSummary:"
        )
        summary = groq_invoke([chunk_prompt])
        chunk_summaries.append(summary)
    # Summarize the chunk summaries
    final_prompt = (
        "You are a helpful assistant. Summarize the following chunk summaries into a single concise summary (100-200 words):\n\n"
        + '\n\n'.join(chunk_summaries)
        + "\n\nFinal Summary:"
    )
    final_summary = groq_invoke([final_prompt])
    return final_summary

# --- Streamlit UI ---
st.set_page_config(page_title="YouTube Assistant", layout="centered")
st.title("🎬 YouTube Video Assistant (Transcript | Summarization | Q&A)")

video_url = st.text_input("🔗 Enter YouTube Video URL or ID")
option = st.radio("Choose what you want to do:", ("Transcript", "Summarization", "Q&A"))

question = ""
answer_length = ""
if option == "Q&A":
    question = st.text_input("❓ Ask a question about the video")
    answer_length = st.selectbox("📏 Select Answer Length", ["Short", "Medium", "Long"])

if st.button("▶️ Process"):
    transcript = None
    video_id = None

    if video_url:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL or ID.")
            st.stop()
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            transcript = ' '.join(chunk['text'] for chunk in transcript_list)
        except (TranscriptsDisabled, NoTranscriptFound):
            st.error("❌ Transcript is disabled or not found for this video.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error while fetching transcript: {str(e)}")
            st.stop()
    else:
        st.error("❌ Please enter a YouTube URL or ID.")
        st.stop()

    if option == "Transcript":
        st.header("📜 Transcript")
        st.write(transcript)

    elif option == "Summarization":
        st.header("📝 Summarization")
        with st.spinner("Summarizing..."):
            summary = summarize_transcript(transcript)
            st.success("✅ Summary:")
            st.markdown(summary)

    elif option == "Q&A":
        st.header("🤔 Ask a Question")
        if not question:
            st.warning("❗ Please enter your question.")
            st.stop()
        with st.spinner("Fetching answer..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.create_documents([transcript])

            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            vector_store = FAISS.from_documents(docs, embeddings)
            retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})

            if answer_length == "Short":
                length_instruction = "Answer in 2-3 short sentences."
            elif answer_length == "Medium":
                length_instruction = "Answer in a few detailed paragraphs (5-8 sentences)."
            else:
                length_instruction = "Answer in a fully detailed explanation with examples and technical depth. Use at least 150+ words."

            # Retrieve relevant context
            def format_docs(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)

            relevant_docs = retriever.get_relevant_documents(question)
            context = format_docs(relevant_docs)

            qa_prompt = (
                "You are a highly knowledgeable assistant helping users understand YouTube video transcripts.\n\n"
                "Please answer ONLY based on the transcript context provided below.\n\n"
                f"{length_instruction}\n\n"
                'If the context is insufficient, reply with: "I don\'t know."\n\n'
                f"Context:\n{context}\n\nQuestion:\n{question}"
            )
            answer = groq_invoke([qa_prompt])
            st.success("✅ Answer:")
            st.markdown(answer)
