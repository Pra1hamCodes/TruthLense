# DeepShield

A full-stack deepfake detection platform. Users sign up, share short videos to a community feed, and run AI-powered analysis that flags each post as **REAL** or **DEEPFAKE** with a per-frame confidence breakdown.

[Demo video](https://github.com/user-attachments/assets/07f6b195-a1d8-4446-b3c9-f92e3d8131a7)

---

## Table of contents

1. [What it does](#what-it-does)
2. [Architecture](#architecture)
3. [Tech stack](#tech-stack)
4. [The two detection models](#the-two-detection-models)
5. [Detection pipeline (request → verdict)](#detection-pipeline-request--verdict)
6. [Evaluation dashboard](#evaluation-dashboard)
7. [Repository layout](#repository-layout)
8. [Local development](#local-development)
9. [Deployment](#deployment)
10. [Security notes](#security-notes)

---

## What it does

- **Authenticated social feed** — sign up / sign in (JWT + bcrypt), post videos with title and description, like, comment.
- **On-demand deepfake detection** — every video post has a "Start Detection" button. The backend extracts frames, scores each one with a Vision Transformer, and returns a verdict plus per-frame confidences.
- **Two-model architecture** — the active production path uses a fine-tuned ViT (PyTorch). A second TensorFlow CNN+LSTM service is bundled for comparison and as a fallback.
- **Per-frame visualization** — the post detail page links to a frame-by-frame viewer that shows each sampled frame with its individual fake/real verdict and confidence.
- **Evaluation dashboard** — a separate page presents training curves, ROC, confusion matrices and model comparisons (see the [important note](#evaluation-dashboard) about the source of these numbers).

---

## Architecture

Four independent services. The frontend talks only to Express; Express fans requests out to the Flask sidecars.

```
┌─────────────────┐      HTTPS       ┌──────────────────────┐
│  Vite + React   │ ───────────────▶ │  Express API         │
│  (Vercel)       │   /api/*         │  (Render :3000)      │
└─────────────────┘                  │                      │
                                     │  Auth · Posts · CRUD │
                                     │  MongoDB (Atlas)     │
                                     │  + JSON file fallback│
                                     └─────────┬────────────┘
                                               │ axios POST /analyze
                                ┌──────────────┴──────────────┐
                                ▼                             ▼
                  ┌─────────────────────┐       ┌─────────────────────┐
                  │ Flask :5001 (ViT)   │       │ Flask :5000         │
                  │ PyTorch · prod path │       │ TensorFlow CNN+LSTM │
                  │ as_model_0.837.pt   │       │ deepfake_detection_ │
                  │   (~214 MB, LFS)    │       │ model.h5 (~95 MB)   │
                  └─────────────────────┘       └─────────────────────┘
```

Why two ML services? They're written in different frameworks (TF vs PyTorch) with different runtime requirements, so isolating each in its own process keeps dependency conflicts impossible and lets us scale or swap them independently.

---

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, framer-motion, lucide-react, recharts, react-router-dom |
| Backend API | Node 24, Express 4, Mongoose, JWT (`jsonwebtoken`), bcryptjs, express-fileupload, axios, zod |
| Database | MongoDB Atlas (with on-disk JSON fallback when Mongo is unreachable) |
| ML — model A | TensorFlow 2.15 (Keras `.h5`), OpenCV, NumPy |
| ML — model B | PyTorch 2.2, `vit-pytorch`, torchvision, OpenCV, Pillow |
| Auth | JWT bearer tokens, bcrypt password hashing, one-shot legacy plaintext → bcrypt migration |
| Deploy | Render (3 services declared in `render.yaml`) + Vercel (frontend) |

---

## The two detection models

### A. ViT — Vision Transformer (production path)

Lives in `backend/flask_server_2/`. Uses [`vit-pytorch`](https://github.com/lucidrains/vit-pytorch).

| Setting | Value |
|---|---|
| Architecture | ViT (vanilla) |
| Input size | 224 × 224 RGB |
| Patch size | 32 |
| Embedding dim | 1024 |
| Depth | 6 transformer blocks |
| Heads | 16 |
| MLP dim | 2048 |
| Dropout | 0.1 (both attention and embedding) |
| Output | Single logit → sigmoid → confidence ∈ [0, 1] |
| Threshold | `confidence < 0.5` ⇒ fake |
| Weights | `as_model_0.837.pt` (~214 MB, tracked via Git LFS) |
| Preprocessing | Resize 256 → CenterCrop 224 → ToTensor → ImageNet normalize (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) |
| Inference | CPU (`map_location='cpu'`); model is loaded once at startup and reused |

The shipped checkpoint name (`0.837`) reflects its validation score during training.

### B. CNN + LSTM (comparison / fallback)

Lives in `backend/flask_server/`. Trained outside this repo and shipped as a Keras `.h5`.

| Setting | Value |
|---|---|
| Input size | 224 × 224 RGB, scaled to `[0, 1]` |
| Output | Single sigmoid (Keras `model.predict`) |
| Threshold | `confidence < 0.5` ⇒ fake |
| Weights | `deepfake_detection_model.h5` (~95 MB, committed directly) |

Either service exposes the same HTTP contract (`POST /analyze` with `multipart/form-data` containing a `video` field) so the Express layer can route to whichever is configured via `FLASK_VIT_URL` / `FLASK_CNN_URL`.

---

## Detection pipeline (request → verdict)

1. **Upload** — `POST /api/posts` with `multipart/form-data`. Express validates the JWT, persists the file to `backend/uploads/`, and writes a `Post` document with `analysis_status: "none"`.
2. **Trigger** — the user clicks **Start Detection** on the post detail page → `POST /api/posts/analyze/:id`.
3. **Forward** — Express opens a stream of the saved video, builds a `FormData`, and `POST`s it to `${FLASK_VIT_URL}/analyze` with a long timeout (default 300 s, overridable via `FLASK_VIT_TIMEOUT_MS`). The post is marked `processing`.
4. **Frame extraction** — Flask uses OpenCV's `VideoCapture` to walk the video, saving every 20th frame as a JPEG inside a per-request directory (`frames_<uuid>_<basename>/`). UUID stems prevent collisions between concurrent uploads.
5. **Per-frame inference** — each saved frame is read, preprocessed (see model table above), and run through the ViT. The Flask handler records `{frame, frame_path, confidence, is_fake}` for every sampled frame.
6. **Aggregation** — the response includes:
   - `total_frames` — count of frames actually scored
   - `fake_frames_count`, `real_frames_count`
   - `confidence` — mean per-frame confidence
   - `is_fake` — **majority vote** with ties going to fake (`fake_frames >= real_frames`); the codebase comment notes this is a deliberate "safer to over-flag than under-flag" choice for a detection system
7. **Persist + render** — Express writes the response into `Post.deepfake_analysis` and a derived `summary` (`status: "FAKE" | "REAL"`, `confidence_percentage`, frame counts). The frontend polls every 5 s while a post is `processing` and re-renders the badge once the verdict lands.
8. **Frame viewer** — the frontend `/posts/:id/analysis` route renders each frame with its `is_fake` flag and confidence, served by the Flask sidecar's `/uploads/frames/<dir>/<file>` static endpoint (path-traversal hardened with `secure_filename` + `realpath` containment checks).
9. **Cleanup** — the uploaded video is removed from the Flask server's `uploads/` directory after analysis. Extracted frames are kept so the UI can link to them.

---

## Evaluation dashboard

The `/evaluation` page renders training curves, ROC, confusion matrices, model comparison tables, and class-wise metrics from `GET /api/model-evaluation`.

**Be honest about what this is.** That endpoint returns synthetically generated values — designed to look like a plausible training run, not measured from real inference. The response is tagged `demo: true` with the explicit note: *"These metrics are synthetically generated demo data for the evaluation dashboard and are NOT measured from real inference runs."* See [`backend/routes/evaluation.js`](backend/routes/evaluation.js).

The dashboard shows what an evaluation page *would* look like once real metrics are wired in. Replacing the demo route with output from a real test-set run is a one-file change — keep the response schema and swap the body for real numbers.

The dashboard surfaces:

- Training / validation accuracy and loss curves over 50 epochs
- ROC curves comparing ViT, CNN-LSTM, EfficientNet-B4, ResNet-101
- Confusion matrices at thresholds 0.50 / 0.60 / 0.70
- Per-class precision / recall / F1
- Model architecture comparison (parameters, inference time, AUC)
- Per-source-of-deepfake breakdown (face-swap, lip-sync, expression-puppet, full-body, authentic)
- Overfitting analysis (train-vs-val gap over time)

---

## Repository layout

```
IPD-main/
├── backend/
│   ├── index.js              # Express app: auth, CORS, route mounting
│   ├── db.js                 # Mongoose connect with file-store fallback
│   ├── models/               # Mongoose schemas: User, Post, OldModelAnalysis
│   ├── routes/
│   │   ├── post.js           # Posts CRUD + comments + likes + /analyze
│   │   └── evaluation.js     # Demo metrics endpoint
│   ├── middleware/auth.js    # JWT verification middleware
│   ├── services/
│   │   └── deepfakeDetector.js  # Thin wrapper around the CNN+LSTM Flask URL
│   ├── flask_server/         # Flask + TensorFlow (CNN+LSTM)
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   └── models/deepfake_detection_model.h5  (95 MB)
│   ├── flask_server_2/       # Flask + PyTorch (ViT) — production path
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   └── models/as_model_0.837.pt            (214 MB, Git LFS)
│   ├── package.json
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── pages/            # Home, LoginPage, RegisterPage, Dashboard,
│   │   │                     # PostAnalysis, FrameAnalysis, ModelEvaluation,
│   │   │                     # OurApproach, OldModelAnalysis
│   │   ├── components/       # Community, CreatePost, PostDetails, NavBar,
│   │   │                     # VideoAnalysisDashboard, ui/* (shadcn primitives)
│   │   ├── contexts/         # AuthContext + useAuth hook
│   │   ├── lib/              # api.ts (fetch wrapper), config.ts (env URLs)
│   │   └── types/
│   ├── package.json
│   ├── vite.config.ts
│   ├── vercel.json
│   └── .env.example
├── render.yaml               # 3-service blueprint for Render
├── DEPLOY.md                 # End-to-end deployment guide
├── .gitattributes            # *.pt → Git LFS
└── .gitignore
```

---

## Local development

### Prerequisites

- **Node.js 24.x** (the backend pins this in `package.json` → `engines.node`)
- **Python 3.11**
- **Git LFS** (`git lfs install`) — needed before clone so the `.pt` weights download
- A **MongoDB Atlas** cluster (free M0 is fine), or skip Mongo entirely and let the file-store fallback take over

### Clone

```bash
git lfs install
git clone https://github.com/Pra1hamCodes/DeepShield.git
cd DeepShield
```

### Backend (Express)

```bash
cd backend
cp .env.example .env
# edit .env: set MONGO_URI and JWT_SECRET (generate one with: openssl rand -hex 32)
npm install
npm start                     # http://localhost:3000
```

If `MONGO_URI` is unreachable, the backend logs `MongoDB disconnected — falling back to file store.` and serves from `backend/data/*.json`. This is fine for local dev. The fallback is intentionally **excluded from git** because it has historically held plaintext passwords; never commit it.

### Flask CNN+LSTM (port 5000)

```bash
cd backend/flask_server
python -m venv .venv
. .venv/Scripts/activate           # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py                      # http://localhost:5000
```

### Flask ViT (port 5001) — the active production model

```bash
cd backend/flask_server_2
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
python app.py                      # http://localhost:5001
```

First boot is slow on both — TensorFlow / PyTorch import is heavy and the ViT loads a 214 MB checkpoint into RAM.

### Frontend (Vite)

```bash
cd frontend
cp .env.example .env
npm install
npm run dev                        # http://localhost:5173
```

The default `.env.example` already points the frontend at `localhost:3000` (Express) and `localhost:5000`/`5001` (Flask), so a clean four-terminal local run works out of the box.

---

## Deployment

See [`DEPLOY.md`](DEPLOY.md) for the end-to-end Render + Vercel walkthrough. In short:

- **Render** reads `render.yaml` and provisions three services: `deepshield-backend` (Node), `deepshield-flask-cnn` (Python), `deepshield-flask-vit` (Python). All three respect `$PORT`. The `.pt` model is pulled via Git LFS during the build.
- **Vercel** hosts the frontend. Set `VITE_API_BASE_URL` and the two `VITE_FLASK_*_URL` vars to your Render URLs.
- After both sides are up, paste the Vercel URL into Render's `CORS_ORIGINS` and `FLASK_CORS_ORIGINS` so cross-origin requests aren't blocked.

Render's free tier spins services down when idle, so the first request after a cold start can take ~60 s.

---

## Security notes

- **JWT secret** — `JWT_SECRET` is required at startup; the server fails fast if it's missing. Tokens default to a 7-day expiry (`JWT_EXPIRES_IN`).
- **Password hashing** — new accounts are stored with bcrypt (cost factor 10). A one-shot migration upgrades any legacy plaintext records on next successful sign-in using `crypto.timingSafeEqual` for the constant-time compare.
- **CORS allowlist** — Express only accepts origins from `CORS_ORIGINS`; both Flask services have their own `FLASK_CORS_ORIGINS`.
- **Path-traversal guards** — Flask's frame-serving endpoints sanitize subdirectory names (rejecting Windows-reserved names like `CON`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9`) and confirm the resolved path is contained within the frames root using `os.path.realpath` + `startswith`.
- **Upload limits** — both Express and Flask cap uploads at 100 MB.
- **Flask debug mode** — disabled by default. Set `FLASK_DEBUG=1` only for local development; the Werkzeug debugger exposes an RCE-via-PIN console.
- **File-store fallback** — `backend/data/` is `.gitignore`d. If you reuse this stack, never commit those files; they have historically contained plaintext passwords from early sign-ups before the bcrypt migration ran.

---

## License

ISC (per `backend/package.json`).
