# AI-Based Multimedia Steganography Detection

Detection-only system for identifying hidden data in images, video, and audio using ML classifiers, statistical analysis, and StegExpose (images).

## Project Status

**Stage 1 — Foundation** (current): monorepo scaffold, FastAPI skeleton, SQLite schema, React shell.

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Project foundation | Done |
| 2 | Image pipeline core | Planned |
| 3 | Backend API & history | Planned |
| 4 | Video detection | Planned |
| 5 | Audio detection | Planned |
| 6 | Frontend dashboard & reports | Planned |
| 7 | Quality, tests & documentation | Planned |

## Architecture

```
Upload (React) → FastAPI → Orchestrator → Detectors (image / video / audio)
                                              ↓
                                         SQLite + file store
```

See the architecture plan for full details on ML approach, datasets, and API design.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Java Runtime 8+ (for StegExpose, later stages)

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

## Repository Layout

```
backend/          FastAPI app, detectors, ML inference
frontend/         React + Vite UI
scripts/          Dataset prep and training scripts
third_party/      StegExpose JAR (download separately)
data/             Datasets and uploads (gitignored)
models/           Trained weights (gitignored)
reports/          Generated reports (gitignored)
```

## Configuration

Environment variables (optional, via `.env` at repo root):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///data/stegdetect.db` | SQLite connection string |
| `STEGEXPOSE_JAR_PATH` | — | Path to StegExpose JAR |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max upload size |

## API (planned)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health |
| POST | `/api/v1/analyze` | Upload and analyze media |
| GET | `/api/v1/analyses` | Paginated history |
| GET | `/api/v1/analyses/{id}` | Analysis detail |
| GET | `/api/v1/analyses/{id}/report` | Download report |

## Training (later stages)

```bash
# After preparing datasets
python scripts/training/train_image_model.py --data-dir data/processed/images
python scripts/training/train_audio_model.py --data-dir data/processed/audio
```

## Limitations

- Detection only — no message recovery
- Strongest on LSB-style embedding; weak against advanced adaptive steganography
- StegExpose requires Java and is image-only
- Academic/demo tool — not court-grade digital forensics

## License

See repository license file (TBD).
