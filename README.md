# 🚨 Accident Detection & Alert System — Streamlit App

A simple, single-file Streamlit app that uses a custom-trained **YOLO** model
to detect accidents in uploaded **images** and **videos**, with an optional
nearby-hospital email alert.

This is a trimmed-down version of the original full-stack (FastAPI + MongoDB
+ multi-page) project — everything except the core detection feature has
been removed, so it deploys as a single Streamlit app.

## 📁 Project Structure

```
.
├── app.py                        # the entire Streamlit app (single file)
├── requirements.txt              # Python dependencies (CPU-only torch)
├── packages.txt                  # apt packages: libgl1 for OpenCV
├── ml_model/
│   ├── best.pt                   # trained YOLO weights (used by app.py)
│   └── training_notebook.ipynb   # notebook used to train best.pt (reference only)
├── .streamlit/
│   └── config.toml               # theme + upload limit
└── README.md
```

`packages.txt` is not optional. `ultralytics` depends on `opencv-python` (the
non-headless build), which needs `libGL.so.1` at import time. Without it a
hosted deploy dies with `ImportError: libGL.so.1: cannot open shared object
file`, even though `opencv-python-headless` is also installed.

Two traps in that file, both learned the hard way:

1. **No comments.** Streamlit Cloud feeds every line straight to
   `apt-get install`, so a `#` comment becomes a package name and the install
   fails with `E: Unable to locate package #`. Bare package names only, one per
   line.
2. **Do not add `libglib2.0-0`.** The build image is Debian trixie but also has
   a bullseye repo configured, so that name resolves to the bullseye version,
   which depends on `libffi7`/`libpcre3` — absent in trixie. apt then reports
   `held broken packages` and aborts everything. If glib ever is genuinely
   needed, the trixie name is `libglib2.0-0t64`.

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## ☁️ Deploy on Streamlit Community Cloud

1. Push this folder to a **GitHub repo** (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick your repo/branch and set **Main file path** to `app.py`.
4. Click **Deploy**. First build takes a few minutes (installs `ultralytics`/`torch`).

### Pushing to GitHub

```bash
git init
git add .
git status          # confirm "Accident Detection System/" is NOT listed
git commit -m "Accident detection streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

That commits 9 files, about 50 MB.

> **Do not commit `Accident Detection System/`.** It is the original FastAPI
> reference project (~398 MB) and `.gitignore` already excludes it. Two of its
> checkpoints, `epoch10.pt` and `epoch80.pt`, are 148 MB each — over GitHub's
> 100 MB hard limit, so the push would be rejected. It also contains a `.env`
> with a live MongoDB connection string and Gmail password.

> **Note on `best.pt` (≈50 MB):** under the 100 MB limit, so a normal push
> works. GitHub prints a size warning above 50 MB, which is harmless.

## ✉️ Email alerts to hospitals

No credentials live in the source. `app.py` is committed to a public repo, so
anything written there is world-readable. Supply them at runtime:

```toml
# Streamlit Cloud: App settings -> Secrets
# Locally:         .streamlit/secrets.toml   (git-ignored)
EMAIL_SENDER = "your-gmail-address@gmail.com"
EMAIL_PASSWORD = "your-16-char-app-password"
```

Use a Gmail **App Password** (Google Account → Security → 2-Step Verification →
App passwords), not your account password. Spaces in the password are fine, the
app strips them.

With nothing configured the app still runs detection end to end — it reports
"Alert channel: not configured" instead of sending mail.

> ⚠️ Never commit a real password to this repo. If one is ever pushed, revoking
> it is the only real fix: deleting the line afterwards does not help, because
> the value stays in git history and public repos are scraped within minutes.

## 📍 Location

The app requests the **device's GPS** through the browser, mirroring
`navigator.geolocation.getCurrentPosition`. If permission is denied it falls
back to an approximate IP lookup, and the coordinate fields stay editable
either way. Addresses are resolved with OpenStreetMap reverse geocoding.

GPS requires HTTPS or `localhost`, so it will not prompt over a plain-HTTP
LAN address.

## ⚙️ Configuration

Set as Streamlit secrets or environment variables, or edit the defaults in
`app.py`:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `ml_model/best.pt` | path to YOLO weights |
| `CONFIDENCE_THRESHOLD` | `0.60` | minimum detection confidence |
| `ALERT_RADIUS_KM` | `50` | hospital search radius |
| `MAX_HOSPITALS_TO_ALERT` | `3` | how many nearest hospitals to alert |
| `EMAIL_SENDER` | in-file default | Gmail address alerts are sent from |
| `EMAIL_PASSWORD` | in-file default | Gmail App Password |

## 🧠 Model

`ml_model/best.pt` is a YOLOv8 model fine-tuned on a road-accident dataset.
`ml_model/training_notebook.ipynb` contains the training code used to
produce it (kept for reference — not needed to run the app).
