# Quick Start Guide

## 1. Get an API Key
Get a free Google AI Studio API key from: https://aistudio.google.com/app/apikey

## 2. Configure Environment
```bash
cp .env.example .env
```
Open `.env` and paste your key:
```
GOOGLE_API_KEY=AIza...
```

## 3. Start the System
```bash
docker compose up -d --build
```
*The first build may take 5-10 minutes.*

## 4. Verify
Run the verification script to ensure everything is correct:
```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

## 5. Use the App
1.  Go to `http://localhost:3000`.
2.  Upload a video of physical activity.
3.  Check the logs `docker compose logs -f backend` to see the processing status.
4.  Once processed, ask questions like:
    *   "Check the safety of this movement."
    *   "What is the knee angle at the bottom?"
