# YouTube Video Assistant

A Streamlit app that retrieves an English YouTube transcript and supports three tasks: reading the transcript, producing a concise summary, and asking questions against transcript-grounded semantic search.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export GROQ_API_KEY=your-key
streamlit run app.py
```

For Streamlit Community Cloud, add `GROQ_API_KEY` in the app's secret settings. Do not commit it.

## How it works

- `youtube-transcript-api` retrieves the transcript without a YouTube API key.
- Long transcripts are summarized in sections before a final synthesis.
- Q&A splits the transcript, embeds the sections locally, and ranks them by cosine similarity.
- Groq generates the final summary or answer. Responses are instructed to stay within the transcript evidence.

YouTube transcript retrieval uses an undocumented web interface and can occasionally be blocked. Videos without an English transcript cannot be processed.
