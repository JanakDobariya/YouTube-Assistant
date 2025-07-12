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
from dotenv import load_dotenv

# --- Environment & Groq Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not set. Please add it to your .env file.")
    st.stop()
groq_client = Groq(api_key=groq_api_key)

def groq_invoke(messages, model="llama3-70b-8192"):
    formatted_messages = []
    for i, m in enumerate(messages):
        role = "user" if i % 2 == 0 else "assistant"
        formatted_messages.append({"role": role, "content": m})
    completion = groq_client.chat.completions.create(
        model=model,
        messages=formatted_messages
    )
    return completion.choices[0].message.content

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
            summary_prompt = (
                "You are a helpful assistant. Summarize the following YouTube video transcript into a concise summary (100-200 words).\n\n"
                f"Transcript:\n{transcript}\n\nSummary:"
            )
            # Use groq_invoke
            summary = groq_invoke([summary_prompt])
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
