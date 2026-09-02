# YouTube Video Assistant

A Streamlit app that retrieves an English YouTube transcript and supports three tasks: reading the transcript, producing a concise summary, and asking questions against transcript-grounded semantic search.

## Live demo

[Open the YouTube Video Assistant](https://youtube-assistant-summarizer.streamlit.app/)

The hosted app may take a few seconds to wake up after a period of inactivity.

## Run locally

```bash
git clone https://github.com/JanakDobariya/YouTube-Assistant.git
cd YouTube-Assistant
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export GROQ_API_KEY=your-key
streamlit run app.py
```

For Streamlit Community Cloud, add `GROQ_API_KEY` in the app's secret settings. Do not commit it.

Open `http://localhost:8501` if Streamlit does not open the browser automatically. On Windows, activate the environment with `.venv\Scripts\activate`.

## Offline use

The interface can be launched locally, but the complete application cannot work offline. It needs internet access to retrieve YouTube transcripts and call Groq for summaries and answers. A valid `GROQ_API_KEY` is required.

## How it works

- `youtube-transcript-api` retrieves the transcript without a YouTube API key.
- Long transcripts are summarized in sections before a final synthesis.
- Q&A splits the transcript, embeds the sections locally, and ranks them by cosine similarity.
- Groq generates the final summary or answer. Responses are instructed to stay within the transcript evidence.

YouTube transcript retrieval uses an undocumented web interface and can occasionally be blocked. Videos without an English transcript cannot be processed.
