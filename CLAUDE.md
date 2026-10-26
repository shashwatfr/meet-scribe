# MeetScribe

Upload a meeting recording (audio or MP4). Get a timestamped transcript,
summary, decisions and action items, stored in SQLite with per-user history
and an admin page. One-click Word export and Q&A with timestamp citations.
Take-home for an Automation + AI role. Goal: a working, accurate, fast tool.

## Stack
Python 3.12 (uv), Streamlit, SQLite, bcrypt, Pydantic, python-docx, ffmpeg.
APIs: Groq (Whisper + LLM), OpenAI (embeddings). Keys only from .env.

## Architecture rules
- Streamlit files only call functions in services/. No business logic in the UI.
- Plain Python orchestration. No LangChain, LangGraph, CrewAI, or Chroma.
- Three LLM calls: Summary, Extraction, Q&A. Each returns JSON validated
  by Pydantic. Retry once on validation failure.
- Long transcripts: summarize in chunks, then merge.
- Wrap Groq and OpenAI calls in thin functions.
- Prefer the standard library and the simplest solution. No abstractions I
  did not ask for.

## Data and user rules
- SQLite only, WAL mode on, file in data/. No encryption.
- The code that opens the database creates data/ if it does not exist.
- Users: no registration screen. seed_users.py creates demo users
  (two normal, one admin) with bcrypt-hashed passwords.
- db.py holds only the connection and schema. All queries live in store.py,
  and every meeting-related store.py function takes user_id and filters by it.
- Admin-only queries live in services/admin.py; every function there checks
  the admin role from the database first.
- Admin page: only the admin user can open it. It shows all tables with a
  user column, row counts and database file size.
- Embeddings are stored in the chunks table as BLOBs, with the embedding
  model name saved on each row.
- Delete the temporary audio file after transcription succeeds.
- Never commit .env or the database. Never print or log API keys or
  plain passwords.
- .env.example has empty values for GROQ_API_KEY and OPENAI_API_KEY.
- Hosted demo (Streamlit Community Cloud) is allowed only when: the hosted
  demo passwords (DEMO_PASSWORD_ADMIN / _ALICE / _BOB in the app's secrets)
  are long random secrets different from the local demo ones; only the
  reviewer gets the link; and hosted storage is treated as temporary.
  Otherwise the app is not for public deployment. Keys and hosted passwords
  live only in secrets, never in the repo. Run locally with
  `streamlit run app.py --server.address localhost`.

## Features
- Login with demo users, per-user history that survives refresh.
- Transcript, summary, decisions, action items from a recording.
- Timing for each pipeline step.
- Word export (python-docx).
- Q&A with timestamp citations.
- Admin page as described above.

## Working rules
- One phase at a time. Plan first, wait for my approval, then build.
- Test from the command line before building any Streamlit page.
- After each phase, summarize what changed and how I can verify it.
- Keep code simple and commented enough that I can explain it.
- Never commit. I will commit myself at the end.
- Keep FLOW.md up to date as we work. Update it after every meaningful
  change, not only at the end of a phase. It is written for me to understand
  the project, so use plain language and explain why, not just what.

## README must include
- Setup steps, how to get Groq and OpenAI keys, and how to run
  seed_users.py. List the demo logins as local-demo-only throwaway accounts.
- Architecture overview.
- One note: audio and transcripts are processed by third-party APIs.
- A short evaluation section with test recordings and results.
