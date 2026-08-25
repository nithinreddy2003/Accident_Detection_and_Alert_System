"""
Accident Detection & Alert System — Streamlit App
--------------------------------------------------
Single-file app: image/video accident detection with a custom-trained YOLO
model (ml_model/best.pt) plus emergency email alerts to nearby hospitals.

Everything (config, hospital list, email pipeline, UI) lives in this file.

Run locally:
    streamlit run app.py
"""

import os
import json
import math
import time
import smtplib
import tempfile
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

st.set_page_config(
    page_title="AccidentWatch AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
# Credentials are deliberately NOT stored in this file. It gets committed to a
# public repo, so anything written here is world-readable and would let anyone
# send mail as that account. Supply them at runtime instead:
#
#   Streamlit Cloud -> App settings -> Secrets:
#       EMAIL_SENDER = "you@gmail.com"
#       EMAIL_PASSWORD = "your 16 character app password"
#
#   Locally -> .streamlit/secrets.toml (git-ignored) or environment variables.
#
# With nothing configured the app still runs detection end to end; it just
# reports that alerts are not configured rather than sending them.
DEFAULT_EMAIL_SENDER = ""
DEFAULT_EMAIL_PASSWORD = ""


def _setting(key, default=""):
    """Streamlit secrets first, then environment, then the in-file default."""
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return str(os.getenv(key, default)).strip()


MODEL_PATH = _setting("MODEL_PATH", os.path.join("ml_model", "best.pt"))
CONFIDENCE_THRESHOLD = float(_setting("CONFIDENCE_THRESHOLD", "0.60"))
ALERT_RADIUS_KM = float(_setting("ALERT_RADIUS_KM", "50"))
MAX_HOSPITALS = int(_setting("MAX_HOSPITALS_TO_ALERT", "3"))
SMTP_HOST = _setting("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(_setting("SMTP_PORT", "465"))

EMAIL_SENDER = _setting("EMAIL_SENDER", DEFAULT_EMAIL_SENDER)
# Gmail displays app passwords as "abcd efgh ijkl mnop" — strip spaces.
EMAIL_PASSWORD = _setting("EMAIL_PASSWORD", DEFAULT_EMAIL_PASSWORD).replace(" ", "")
EMAIL_CONFIGURED = bool(EMAIL_SENDER and EMAIL_PASSWORD)

FRAME_SKIP = 4
MAX_EVIDENCE_FRAMES = 9

HOSPITAL_DATABASE = [
    {"name": "Guntur Government General Hospital", "email": "nithin9231@gmail.com", "phone": "0863-2222222",
     "lat": 16.3067, "lon": 80.4365, "address": "Guntur, Andhra Pradesh"},
    {"name": "Ramesh Hospitals Guntur", "email": "nithin9231@gmail.com", "phone": "0863-2344555",
     "lat": 16.3145, "lon": 80.4332, "address": "Guntur, Andhra Pradesh"},
    {"name": "KIMS SIKHARA Hospitals", "email": "nithin9231@gmail.com", "phone": "0863-2399999",
     "lat": 16.3122, "lon": 80.4285, "address": "Guntur, Andhra Pradesh"},
    {"name": "Sanjivi Hospitals", "email": "nithin9231@gmail.com", "phone": "0863-2355555",
     "lat": 16.3060, "lon": 80.4390, "address": "Guntur, Andhra Pradesh"},
    {"name": "Sree Prathima Super Speciality Hospital", "email": "nithin9231@gmail.com", "phone": "0863-2377777",
     "lat": 16.3048, "lon": 80.4372, "address": "Guntur, Andhra Pradesh"},
]

# ─────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────
# NOTE: this MUST go through st.html, not st.markdown.
# st.markdown(unsafe_allow_html=True) still runs the string through the
# frontend Markdown parser first, which treats the `*` in selectors like
# [class*="st-key-..."] and in /* comments */ as emphasis markers. That
# corrupts the <style> block, so the browser closes it early and dumps the
# rest of the CSS onto the page as visible text. st.html has no Markdown step.
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
:root{
  --bg:#0a0e1a; --bg2:#0f1425;
  --surface:rgba(255,255,255,.045); --surface2:rgba(255,255,255,.07);
  --border:rgba(255,255,255,.09); --border-lit:rgba(255,255,255,.16);
  --text:#e8ecf5; --muted:#8d9ab5; --dim:#5f6b85;
  --indigo:#6366f1; --violet:#8b5cf6; --cyan:#22d3ee;
  --danger:#ef4444; --orange:#f97316; --green:#10b981; --amber:#f59e0b;
}

/* ---------- shell ---------- */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 520px at 12% -8%, rgba(99,102,241,.16), transparent 60%),
    radial-gradient(900px 480px at 88% 0%, rgba(34,211,238,.10), transparent 55%),
    radial-gradient(700px 600px at 50% 110%, rgba(139,92,246,.12), transparent 60%),
    linear-gradient(180deg,var(--bg) 0%, var(--bg2) 100%);
  background-attachment:fixed;
}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stDecoration"]{
  background:linear-gradient(90deg,var(--indigo),var(--violet),var(--cyan));height:3px;
}
#MainMenu, footer{visibility:hidden;}
.block-container{padding-top:1.6rem; padding-bottom:4rem; max-width:1220px;}
html,body,[class*="st-"],button,input,textarea{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
h1,h2,h3,h4{font-family:'Space Grotesk','Inter',sans-serif !important;}

/* ---------- hero ---------- */
.hero{
  position:relative; overflow:hidden; border-radius:22px; padding:2.1rem 2.3rem 1.9rem;
  margin-bottom:1.3rem; border:1px solid var(--border);
  background:
    radial-gradient(700px 240px at 0% 0%, rgba(139,92,246,.22), transparent 62%),
    radial-gradient(600px 240px at 100% 100%, rgba(34,211,238,.14), transparent 60%),
    linear-gradient(135deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  box-shadow:0 22px 60px -28px rgba(0,0,0,.9), inset 0 1px 0 rgba(255,255,255,.08);
}
.hero:before{
  content:""; position:absolute; inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.05),transparent);
  transform:translateX(-100%); animation:sheen 7s ease-in-out infinite;
}
@keyframes sheen{0%,68%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.hero-badge{
  display:inline-flex; align-items:center; gap:.5rem; font-size:.68rem; font-weight:700;
  letter-spacing:.16em; text-transform:uppercase; color:#c7d0ea;
  background:rgba(139,92,246,.16); border:1px solid rgba(139,92,246,.34);
  padding:.34rem .8rem; border-radius:999px; margin-bottom:.9rem;
}
.hero-badge i{width:6px;height:6px;border-radius:50%;background:var(--cyan);
  box-shadow:0 0 10px var(--cyan); animation:blip 1.9s ease-in-out infinite;}
@keyframes blip{0%,100%{opacity:1}50%{opacity:.25}}
.hero h1{
  font-size:2.6rem !important; font-weight:700 !important; line-height:1.06;
  margin:0 0 .5rem; letter-spacing:-.03em;
  background:linear-gradient(100deg,#fff 8%,#c9b6ff 46%,#7dd3fc 92%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p{color:var(--muted); font-size:.97rem; margin:0; max-width:62ch; line-height:1.6;}
.pills{display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.3rem;}
.pill{
  display:inline-flex; align-items:center; gap:.45rem; font-size:.76rem; font-weight:600;
  padding:.42rem .85rem; border-radius:999px; border:1px solid var(--border-lit);
  background:var(--surface); color:#cfd8ee; backdrop-filter:blur(8px);
}
.pill b{font-weight:700;}
.pill .dot{width:7px;height:7px;border-radius:50%;}
.dot-ok{background:var(--green); box-shadow:0 0 9px var(--green);}
.dot-warn{background:var(--amber); box-shadow:0 0 9px var(--amber);}
.dot-off{background:var(--dim);}

/* ---------- cards ----------
   Real st.container(border=True, key="awacard-*") wrappers, so the panel
   actually encloses its widgets. A raw markdown div cannot: Streamlit puts
   every element in its own container, so the browser closes the tag
   immediately and the "card" renders as an empty box. */
[class*="st-key-awacard-"]{
  background:var(--surface) !important;
  border:1px solid var(--border) !important;
  border-radius:18px !important;
  padding:1.3rem 1.45rem !important;
  margin-bottom:1.05rem;
  backdrop-filter:blur(10px);
  box-shadow:0 14px 40px -26px rgba(0,0,0,.85), inset 0 1px 0 rgba(255,255,255,.05);
  transition:border-color .25s ease;
}
[class*="st-key-awacard-"]:hover{border-color:var(--border-lit) !important;}
.card-head{
  display:flex; align-items:center; gap:.6rem; font-family:'Space Grotesk',sans-serif;
  font-size:.73rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase;
  color:#a9b6d4; margin-bottom:1.05rem; padding-bottom:.7rem;
  border-bottom:1px solid var(--border);
}
.card-head .ico{
  width:26px;height:26px;border-radius:8px;display:grid;place-items:center;font-size:.82rem;
  background:linear-gradient(135deg,rgba(99,102,241,.3),rgba(139,92,246,.18));
  border:1px solid rgba(139,92,246,.3);
}

/* ---------- verdict ---------- */
.verdict{
  position:relative; overflow:hidden; border-radius:18px; padding:1.7rem 1.5rem;
  text-align:center; border:1px solid; margin-bottom:1.15rem;
}
.verdict.hit{
  border-color:rgba(239,68,68,.5);
  background:radial-gradient(520px 200px at 50% 0%, rgba(239,68,68,.26), transparent 68%),
             linear-gradient(180deg, rgba(239,68,68,.1), rgba(249,115,22,.05));
  animation:alarm 2.4s ease-in-out infinite;
}
@keyframes alarm{
  0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.32), inset 0 1px 0 rgba(255,255,255,.06);}
  50%{box-shadow:0 0 34px 5px rgba(239,68,68,.16), inset 0 1px 0 rgba(255,255,255,.06);}
}
.verdict.clear{
  border-color:rgba(16,185,129,.45);
  background:radial-gradient(520px 200px at 50% 0%, rgba(16,185,129,.2), transparent 68%),
             linear-gradient(180deg, rgba(16,185,129,.08), rgba(34,211,238,.04));
}
.verdict .glyph{font-size:2.5rem; line-height:1; margin-bottom:.5rem;}
.verdict h2{
  font-size:1.5rem !important; font-weight:700 !important; margin:0; letter-spacing:.01em;
}
.verdict.hit h2{color:#ff8080;}
.verdict.clear h2{color:#4ade9f;}
.verdict p{margin:.5rem 0 0; font-size:.87rem; color:var(--muted);}

/* ---------- confidence ---------- */
.conf-wrap{margin:.3rem 0 1.1rem;}
.conf-top{display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.5rem;}
.conf-label{font-size:.74rem; font-weight:600; letter-spacing:.13em; text-transform:uppercase; color:var(--muted);}
.conf-val{font-family:'JetBrains Mono',monospace; font-size:1.45rem; font-weight:500; color:#fff;}
.conf-track{height:9px; border-radius:99px; background:rgba(255,255,255,.07); overflow:hidden;}
.conf-fill{height:100%; border-radius:99px; transition:width .8s cubic-bezier(.22,1,.36,1);}
.fill-hit{background:linear-gradient(90deg,#f59e0b,#ef4444);box-shadow:0 0 16px rgba(239,68,68,.55);}
.fill-clear{background:linear-gradient(90deg,#22d3ee,#10b981);box-shadow:0 0 16px rgba(16,185,129,.5);}

/* ---------- object rows ---------- */
.obj{
  display:flex; align-items:center; gap:.85rem; padding:.7rem .9rem; margin-bottom:.5rem;
  border-radius:12px; background:var(--surface2); border:1px solid var(--border);
  border-left:3px solid var(--violet);
}
.obj-name{font-weight:600; font-size:.88rem; text-transform:capitalize; min-width:92px;}
.obj-bar{flex:1; height:6px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden;}
.obj-bar i{display:block; height:100%; border-radius:99px;
  background:linear-gradient(90deg,var(--violet),var(--cyan));}
.obj-pct{font-family:'JetBrains Mono',monospace; font-size:.8rem; color:#c9d4ee; min-width:52px; text-align:right;}

/* ---------- stats ---------- */
.stats{display:grid; grid-template-columns:repeat(auto-fit,minmax(112px,1fr)); gap:.65rem; margin-bottom:1.1rem;}
.stat{
  padding:.9rem .8rem; border-radius:14px; text-align:center;
  background:var(--surface2); border:1px solid var(--border);
}
.stat .n{font-family:'JetBrains Mono',monospace; font-size:1.32rem; font-weight:500; color:#fff; line-height:1.1;}
.stat .k{font-size:.66rem; letter-spacing:.11em; text-transform:uppercase; color:var(--dim); margin-top:.3rem;}
.stat.hot .n{color:#ff8f8f;}

/* ---------- alert timeline ---------- */
.alert-row{
  display:flex; align-items:flex-start; gap:.7rem; padding:.62rem .9rem; margin-bottom:.45rem;
  border-radius:11px; background:var(--surface2); border:1px solid var(--border); font-size:.85rem;
}
.alert-row .mk{
  width:8px;height:8px;border-radius:50%;margin-top:.42rem;flex:0 0 8px;
}
.mk-ok{background:var(--green);box-shadow:0 0 9px var(--green);}
.mk-err{background:var(--danger);box-shadow:0 0 9px var(--danger);}
.mk-warn{background:var(--amber);box-shadow:0 0 9px var(--amber);}
.mk-info{background:var(--cyan);box-shadow:0 0 9px var(--cyan);}
.alert-row.err{border-color:rgba(239,68,68,.34); background:rgba(239,68,68,.08);}
.alert-row.ok{border-color:rgba(16,185,129,.28); background:rgba(16,185,129,.07);}

/* ---------- guidance ---------- */
.guide{
  margin-top:1.1rem; padding:1.1rem 1.25rem; border-radius:14px; font-size:.87rem; line-height:1.65;
}
.guide.sos{background:linear-gradient(135deg,rgba(239,68,68,.13),rgba(249,115,22,.06));
  border:1px solid rgba(239,68,68,.34); color:#ffd9d9;}
.guide.safe{background:linear-gradient(135deg,rgba(16,185,129,.11),rgba(34,211,238,.05));
  border:1px solid rgba(16,185,129,.3); color:#c8f5e2;}
.guide b{color:#fff;}

/* ---------- empty state ---------- */
.empty{text-align:center; padding:3.1rem 1rem; color:var(--dim);}
.empty .g{font-size:3.1rem; opacity:.5; margin-bottom:.85rem;
  filter:drop-shadow(0 0 22px rgba(139,92,246,.34));}
.empty p{font-size:.88rem; margin:0;}

/* ---------- streamlit widgets ---------- */
.stTabs [data-baseweb="tab-list"]{
  gap:.4rem; background:var(--surface); padding:.4rem; border-radius:14px;
  border:1px solid var(--border); margin-bottom:1.2rem;
}
.stTabs [data-baseweb="tab"]{
  height:auto; padding:.62rem 1.35rem; border-radius:10px; font-size:.88rem; font-weight:600;
  color:var(--muted); background:transparent; border:none;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,var(--indigo),var(--violet)) !important;
  color:#fff !important; box-shadow:0 6px 18px -8px rgba(139,92,246,.85);
}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none;}

[data-testid="stFileUploaderDropzone"]{
  background:rgba(255,255,255,.03); border:1.5px dashed var(--border-lit); border-radius:14px;
  transition:border-color .22s ease, background .22s ease;
}
[data-testid="stFileUploaderDropzone"]:hover{
  border-color:var(--violet); background:rgba(139,92,246,.07);
}

.stButton>button,[data-testid="stFormSubmitButton"]>button,[data-testid="stBaseButton-primaryFormSubmit"]{
  border-radius:12px; font-weight:600; font-size:.9rem; padding:.62rem 1.1rem;
  border:1px solid var(--border-lit); transition:transform .18s ease, box-shadow .22s ease;
}
[data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"],
[data-testid="stBaseButton-primaryFormSubmit"]{
  background:linear-gradient(135deg,var(--indigo),var(--violet)) !important;
  border:none !important; color:#fff !important;
  box-shadow:0 10px 26px -12px rgba(139,92,246,.9);
}
[data-testid="stFormSubmitButton"]>button:hover,.stButton>button:hover{
  transform:translateY(-1px); box-shadow:0 14px 30px -12px rgba(139,92,246,.75);
}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{
  background:rgba(255,255,255,.045) !important; border-radius:10px;
  font-family:'JetBrains Mono',monospace; font-size:.85rem;
}
[data-testid="stNumberInput"] input:focus,[data-testid="stTextInput"] input:focus{
  border-color:var(--violet) !important; box-shadow:0 0 0 3px rgba(139,92,246,.18) !important;
}
[data-testid="stWidgetLabel"] label{
  font-size:.73rem !important; font-weight:600 !important; letter-spacing:.09em;
  text-transform:uppercase; color:var(--muted) !important;
}
[data-testid="stImage"] img{border-radius:14px; border:1px solid var(--border);}
[data-testid="stCaptionContainer"],.stCaption{color:var(--dim) !important; font-size:.78rem !important;}
[data-testid="stSidebar"]{background:rgba(10,14,26,.96); border-right:1px solid var(--border);}
[data-testid="stSidebar"] h3{font-size:.95rem !important;}
hr{border-color:var(--border) !important;}
.foot{text-align:center; color:var(--dim); font-size:.76rem; margin-top:2.6rem;
  padding-top:1.4rem; border-top:1px solid var(--border);}
.foot b{color:#a9b6d4;}
</style>
""")


# ─────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading detection model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────
# LOCATION
# ─────────────────────────────────────────────────────────────────────────
# Device GPS widget. Mirrors the working frontend's
# navigator.geolocation.getCurrentPosition: it auto-tries on load AND offers an
# explicit button, exactly like the "Auto-detect Location" button there.
#
# Two hard constraints shape this code:
#   1. A component runs in a sandboxed iframe, so reading window.parent.location
#      can throw a SecurityError. Every parent access is therefore wrapped in
#      try/catch — an earlier version did not, and the exception aborted the
#      script before getCurrentPosition ever ran, which is why nothing happened.
#   2. The browser cannot hand a value back to Python, so a successful fix is
#      passed via the parent URL's query string. If that navigation is blocked,
#      the coordinates are shown so they can be copied into the fields instead.
_GEO_JS = """
<style>
  body{margin:0;font-family:Inter,-apple-system,'Segoe UI',sans-serif;background:transparent;}
  .geo{display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;}
  button{
    font:600 .82rem Inter,sans-serif;color:#fff;cursor:pointer;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);border:none;
    border-radius:10px;padding:.55rem 1rem;
  }
  button:hover{filter:brightness(1.08);}
  #msg{font-size:.78rem;color:#8d9ab5;}
  #msg.ok{color:#34d399;} #msg.err{color:#fca5a5;}
  code{font-family:'JetBrains Mono',monospace;color:#e8ecf5;background:rgba(255,255,255,.08);
    padding:.15rem .4rem;border-radius:6px;}
</style>
<div class="geo">
  <button id="go">Use my current location</button>
  <span id="msg">Checking location access...</span>
</div>
<script>
(function () {
  var msg = document.getElementById("msg");
  var btn = document.getElementById("go");

  function say(text, cls) { msg.innerHTML = text; msg.className = cls || ""; }

  // Any of these may throw inside a sandboxed iframe.
  function parentUrl() {
    try { return new URL(window.parent.location.href); } catch (e) { return null; }
  }
  function navigate(url) {
    try { window.parent.location.replace(url); return true; } catch (e) {}
    try { window.top.location.replace(url); return true; } catch (e) {}
    return false;
  }

  function alreadyAnswered() {
    var u = parentUrl();
    if (!u) return false;                       // cannot tell -> allow an attempt
    return u.searchParams.has("lat") || u.searchParams.get("geo") === "off";
  }

  function handoff(lat, lon) {
    var u = parentUrl();
    if (!u) return false;
    u.searchParams.set("lat", lat);
    u.searchParams.set("lon", lon);
    u.searchParams.set("geo", "on");
    return navigate(u.toString());
  }

  function markUnavailable(reason) {
    var u = parentUrl();
    if (u) {
      u.searchParams.set("geo", "off");
      if (navigate(u.toString())) return;
    }
    say(reason, "err");
  }

  function request() {
    // Geolocation only works on https:// or localhost.
    if (!window.isSecureContext) {
      say("Blocked: this page is not a secure context. Open the app on " +
          "<code>localhost</code> or over <code>https</code>.", "err");
      return;
    }
    if (!navigator.geolocation) {
      markUnavailable("This browser does not support geolocation.");
      return;
    }
    say("Requesting location...");
    navigator.geolocation.getCurrentPosition(
      function (pos) {
        var lat = pos.coords.latitude.toFixed(6);
        var lon = pos.coords.longitude.toFixed(6);
        if (!handoff(lat, lon)) {
          say("Location found - type it into the fields below: " +
              "<code>" + lat + "</code>, <code>" + lon + "</code>", "ok");
        }
      },
      function (err) {
        var why = err.code === 1 ? "Permission denied."
                : err.code === 2 ? "Position unavailable."
                : err.code === 3 ? "Request timed out."
                : "Location request failed.";
        markUnavailable(why + " Enter the coordinates manually.");
      },
      { timeout: 10000, enableHighAccuracy: true, maximumAge: 0 }
    );
  }

  btn.addEventListener("click", request);
  if (alreadyAnswered()) { say("Using the location from this page."); } else { request(); }
})();
</script>
"""


@st.cache_data(ttl=3600, show_spinner=False)
def reverse_geocode(lat, lon):
    """Street-level address for GPS coordinates (OpenStreetMap Nominatim)."""
    try:
        url = "https://nominatim.openstreetmap.org/reverse?" + urllib.parse.urlencode(
            {"format": "json", "lat": f"{lat:.6f}", "lon": f"{lon:.6f}", "zoom": "16"}
        )
        req = urllib.request.Request(url, headers={"User-Agent": "accident-detection-app/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.load(resp).get("display_name") or ""
    except Exception:
        return ""


def gps_fix():
    """The device's real GPS position, or None.

    There is deliberately no fallback: a guessed location on an emergency alert
    is worse than none, so if the browser will not give a fix the coordinates
    stay empty and must be typed in.
    """
    params = st.query_params
    if "lat" in params and "lon" in params:
        try:
            lat, lon = float(params["lat"]), float(params["lon"])
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return {"lat": lat, "lon": lon,
                        "address": reverse_geocode(lat, lon),
                        "source": "device GPS"}
        except (TypeError, ValueError):
            pass
    return None


def calc_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_hospitals(lat, lon, n=MAX_HOSPITALS):
    """Nearest hospitals within ALERT_RADIUS_KM, falling back to the nearest
    few when nothing is in range so an alert still goes out."""
    if lat is None or lon is None:
        return HOSPITAL_DATABASE[:n]
    scored = sorted(
        ((h, calc_distance(lat, lon, h["lat"], h["lon"])) for h in HOSPITAL_DATABASE),
        key=lambda pair: pair[1],
    )
    in_range = [h for h, d in scored if d <= ALERT_RADIUS_KM]
    return (in_range or [h for h, _ in scored])[:n]


# ─────────────────────────────────────────────────────────────────────────
# EMAIL
# ─────────────────────────────────────────────────────────────────────────
def build_email_html(incident_type, timestamp, address, location_block, details_html):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<style>
  body{{margin:0;padding:0;background:#f4f4f4;font-family:'Segoe UI',sans-serif}}
  .wrap{{max-width:620px;margin:0 auto;background:#fff;border-radius:12px;overflow:hidden}}
  .hdr{{background:linear-gradient(135deg,#7b0000,#dc2626);color:#fff;padding:28px 24px;text-align:center}}
  .hdr h1{{margin:0;font-size:22px;letter-spacing:1px}}
  .hdr p{{margin:6px 0 0;font-size:12px;opacity:.8}}
  .banner{{background:#b71c1c;color:#fff;text-align:center;padding:14px;font-size:18px;font-weight:700}}
  .body{{padding:28px 24px}}
  .info-box{{background:#fdf5f5;border-left:5px solid #dc2626;border-radius:6px;padding:16px;margin:18px 0}}
  .info-box h3{{margin:0 0 12px;color:#7b0000;font-size:14px;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #f0c0c0;padding-bottom:8px}}
  .row{{margin-bottom:8px;font-size:13px}}
  .lbl{{font-weight:700;color:#333;display:inline-block;min-width:120px}}
  .val{{color:#555}}
  .btn{{display:inline-block;padding:12px 28px;background:#dc2626;color:#fff;text-decoration:none;border-radius:6px;font-weight:700;font-size:13px;margin-top:12px}}
  .attach{{background:#fff8e1;color:#7c5200;padding:10px 14px;text-align:center;border:1px solid #ffe082;border-radius:6px;margin-top:16px;font-size:12px}}
  .footer{{background:#1a0505;color:#888;text-align:center;padding:16px;font-size:11px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="hdr">
    <h1>🚨 ACCIDENT DETECTION SYSTEM</h1>
    <p>AI-Powered Emergency Response · YOLO Detection</p>
  </div>
  <div class="banner">{incident_type}</div>
  <div class="body">
    <p style="font-size:15px;color:#222;">Dear Emergency Response Team,</p>
    <p style="font-size:14px;color:#444;line-height:1.6;">Our AI system has detected a potential road accident.
    Immediate medical attention may be required at the location below.</p>

    <div class="info-box">
      <h3>📍 Incident Location</h3>
      <div class="row"><span class="lbl">Time:</span><span class="val">{timestamp}</span></div>
      <div class="row"><span class="lbl">Address:</span><span class="val">{address}</span></div>
      {location_block}
    </div>

    <div class="info-box" style="border-left-color:#ff4444;">
      <h3>📋 AI Detection Report</h3>
      {details_html}
    </div>

    <div class="attach">
      <strong>📎 Evidence Attached:</strong> See the attached image for visual evidence of the detected accident.
    </div>
  </div>
  <div class="footer">
    Automated Alert · Accident Detection &amp; Alert System Using YOLO · Do not reply to this email.
  </div>
</div>
</body>
</html>"""


def send_alert_email(subject, html_body, hospitals, image_bytes, filename="evidence.jpg"):
    """Email every hospital that has an address on file, reusing one SMTP
    connection. Returns (sent, failed, skipped, alerted_names, error)."""
    if not EMAIL_CONFIGURED:
        return 0, 0, 0, [], ""

    recipients = [h for h in hospitals if h.get("email")]
    skipped = len(hospitals) - len(recipients)
    if not recipients:
        return 0, 0, skipped, [], ""

    try:
        server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30)
    except Exception as exc:
        return 0, len(recipients), skipped, [], f"Could not reach {SMTP_HOST}:{SMTP_PORT} — {exc}"

    try:
        try:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        except smtplib.SMTPAuthenticationError:
            return (0, len(recipients), skipped, [],
                    "Gmail rejected the login. Use a 16-character App Password "
                    "(Google Account → Security → App passwords), not the account password.")
        except Exception as exc:
            return 0, len(recipients), skipped, [], f"SMTP login failed — {exc}"

        sent, failed, alerted, error = 0, 0, [], ""
        for hospital in recipients:
            try:
                # "mixed" (not "alternative") so the HTML body AND the image
                # attachment both survive delivery.
                msg = MIMEMultipart("mixed")
                msg["From"] = EMAIL_SENDER
                msg["To"] = hospital["email"]
                msg["Subject"] = subject
                msg.attach(MIMEText(html_body, "html", "utf-8"))

                if image_bytes:
                    img = MIMEImage(image_bytes, _subtype="jpeg")
                    img.add_header("Content-Disposition", "attachment", filename=filename)
                    msg.attach(img)

                server.send_message(msg)
                sent += 1
                alerted.append(hospital["name"])
                time.sleep(0.2)
            except Exception as exc:
                failed += 1
                error = str(exc)
        return sent, failed, skipped, alerted, error
    finally:
        try:
            server.quit()
        except Exception:
            pass


def alert_status_lines(hospitals, sent, failed, skipped, alerted, error):
    """Returns (kind, text) pairs so the UI can colour each line."""
    if not EMAIL_CONFIGURED:
        return [("warn", "Email not configured — set EMAIL_SENDER / EMAIL_PASSWORD in app.py.")]
    if not hospitals:
        return [("warn", "No hospitals available to alert.")]
    if sent == 0 and skipped == len(hospitals) and skipped > 0:
        return [("warn", "No nearby hospital has an email address on file.")]

    lines = [("ok", f"Alert delivered to <b>{name}</b>") for name in alerted]
    if failed:
        lines.append(("err", f"{failed} alert(s) failed{(' — ' + error) if error else ''}"))
    if skipped:
        lines.append(("warn", f"{skipped} hospital(s) skipped — no email on file"))
    return lines or [("warn", "No alerts were sent.")]


def coordinate_note(lat, lon):
    """Exactly what location the alert carried — no silent substitutions."""
    if lat is not None and lon is not None:
        return ("info", f"Location sent: <b>{lat:.6f}, {lon:.6f}</b>")
    if lat is not None or lon is not None:
        missing = "Longitude" if lon is None else "Latitude"
        return ("warn", f"{missing} is missing — the alert was sent "
                        f"<b>without a location</b>")
    return ("warn", "No coordinates available — the alert was sent "
                    "<b>without a location</b>")


def dispatch_alerts(kind, details_html, hospitals, image_bytes, filename, address, lat, lon):
    """Compose + send the alert and return (kind, text) status lines."""
    timestamp = datetime.now().strftime("%d %b %Y, %I:%M %p")
    has_coords = lat is not None and lon is not None
    address_str = address.strip() if address and address.strip() else (
        "Not provided" if has_coords else "Not available"
    )

    # No coordinates means no coordinates: never substitute a placeholder
    # location, because a wrong position on an emergency alert is worse than an
    # explicit "unknown".
    if has_coords:
        maps_link = f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"
        location_block = (
            f'<div class="row"><span class="lbl">Coordinates:</span>'
            f'<span class="val">{lat:.6f}, {lon:.6f}</span></div>'
            f'<div style="text-align:center;margin-top:14px;">'
            f'<a href="{maps_link}" class="btn" target="_blank">📍 Open in Google Maps</a></div>'
        )
        subject_where = address_str if address_str != "Not provided" else f"{lat:.5f}, {lon:.5f}"
    else:
        location_block = (
            '<div class="row"><span class="lbl">Coordinates:</span>'
            '<span class="val" style="color:#b71c1c;font-weight:700;">NOT AVAILABLE</span></div>'
            '<p style="margin:12px 0 0;font-size:12px;color:#7b0000;">'
            'GPS could not be obtained for this report, so no location is attached. '
            'Please confirm the location with the reporting party before dispatching.</p>'
        )
        subject_where = "location unavailable"

    html_body = build_email_html(
        f"🚨 CRITICAL ALERT: ACCIDENT DETECTED IN {kind.upper()}",
        timestamp, address_str, location_block, details_html,
    )
    subject = f"🚨 EMERGENCY: Accident Detected — {kind.title()} Evidence | {subject_where}"

    sent, failed, skipped, alerted, error = send_alert_email(
        subject, html_body, hospitals, image_bytes, filename
    )
    return ([coordinate_note(lat, lon)]
            + alert_status_lines(hospitals, sent, failed, skipped, alerted, error))


# ─────────────────────────────────────────────────────────────────────────
# EVIDENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────
def build_collage(frames_bgr, cols=3, thumb_w=320, thumb_h=180):
    frames = frames_bgr[:MAX_EVIDENCE_FRAMES]
    if not frames:
        return None
    rows = math.ceil(len(frames) / cols)
    thumbs = []
    for i, f in enumerate(frames):
        t = cv2.resize(f, (thumb_w, thumb_h))
        cv2.rectangle(t, (0, 0), (100, 26), (0, 0, 0), -1)
        cv2.putText(t, f"Frame {i+1}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
    blank = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    while len(thumbs) % cols != 0:
        thumbs.append(blank)
    collage = np.vstack([np.hstack(thumbs[r * cols:(r + 1) * cols]) for r in range(rows)])
    return cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)


def jpeg_bytes(rgb_image):
    if rgb_image is None:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else None


# ─────────────────────────────────────────────────────────────────────────
# UI PARTS
# ─────────────────────────────────────────────────────────────────────────
@contextmanager
def card(icon, title, key):
    """Glass panel that genuinely wraps its contents (see the CSS note)."""
    with st.container(border=True, key=f"awacard-{key}"):
        st.html(f'<div class="card-head"><span class="ico">{icon}</span>{title}</div>')
        yield


def location_form(prefix, submit_label, disabled):
    """Location fields + submit button inside a form.

    The form matters: with a plain st.button, clicking straight after typing
    can register the click before the number_input commits, which sent alerts
    with no coordinates at all.
    """
    auto = gps_fix()
    with card("📍", "Incident Location", f"{prefix}-loc"):
        # Outside the form: a form defers every interaction until submit, which
        # would stop this button from ever firing.
        components.html(_GEO_JS, height=46)
        if auto:
            st.caption(f"✓ Device GPS acquired — {auto['lat']:.6f}, {auto['lon']:.6f}")
        else:
            st.caption("No GPS fix yet. Alerts are sent **without a location** "
                       "unless coordinates are filled in below.")

    with st.form(f"{prefix}_form", border=False):
        with card("🧭", "Coordinates", f"{prefix}-coords"):
            c1, c2 = st.columns(2)
            lat = c1.number_input(
                "Latitude", value=auto["lat"] if auto else None,
                min_value=-90.0, max_value=90.0, step=1e-6, format="%.6f", key=f"{prefix}_lat",
            )
            lon = c2.number_input(
                "Longitude", value=auto["lon"] if auto else None,
                min_value=-180.0, max_value=180.0, step=1e-6, format="%.6f", key=f"{prefix}_lon",
            )
            address = st.text_input(
                "Location description", value=auto["address"] if auto else "",
                placeholder="e.g. NH-65, Vijayawada highway", key=f"{prefix}_addr",
            )
            if auto:
                st.caption("Filled from device GPS — edit if the scene is elsewhere.")
            else:
                st.caption("Type the coordinates if GPS is unavailable.")
        submitted = st.form_submit_button(submit_label, type="primary", use_container_width=True,
                                          disabled=disabled)
    return lat, lon, address, submitted


def show_verdict(detected, confidence, extra=""):
    cls = "hit" if detected else "clear"
    glyph = "🚨" if detected else "✅"
    title = "ACCIDENT DETECTED" if detected else "NO ACCIDENT"
    sub = (f"Identified with {confidence*100:.1f}% peak confidence. {extra}".strip()
           if detected else "The model found no accident indicators in this media.")
    st.html(f'<div class="verdict {cls}"><div class="glyph">{glyph}</div>'
            f'<h2>{title}</h2><p>{sub}</p></div>')

    st.html(
        f'<div class="conf-wrap"><div class="conf-top">'
        f'<span class="conf-label">Confidence</span>'
        f'<span class="conf-val">{confidence*100:.1f}%</span></div>'
        f'<div class="conf-track"><div class="conf-fill {"fill-hit" if detected else "fill-clear"}" '
        f'style="width:{min(max(confidence,0.0),1.0)*100:.1f}%"></div></div></div>'
    )


def show_objects(objects):
    if not objects:
        return
    st.html('<div class="card-head" style="margin-top:.4rem;">'
            '<span class="ico">🎯</span>Detected Objects</div>')
    for o in objects:
        pct = o["confidence"] * 100
        st.html(
            f'<div class="obj"><span class="obj-name">{o["type"]}</span>'
            f'<span class="obj-bar"><i style="width:{pct:.1f}%"></i></span>'
            f'<span class="obj-pct">{pct:.1f}%</span></div>'
        )


def show_stats(items):
    cells = "".join(
        f'<div class="stat{" hot" if hot else ""}"><div class="n">{val}</div>'
        f'<div class="k">{key}</div></div>'
        for key, val, hot in items
    )
    st.html(f'<div class="stats">{cells}</div>')


def show_alerts(status_lines):
    if not status_lines:
        return
    st.html('<div class="card-head" style="margin-top:1rem;">'
            '<span class="ico">📡</span>Emergency Dispatch</div>')
    for kind, text in status_lines:
        row_cls = kind if kind in ("ok", "err") else ""
        st.html(f'<div class="alert-row {row_cls}"><span class="mk mk-{kind}"></span>'
                f'<span>{text}</span></div>')


def show_guidance(detected):
    if detected:
        st.html(
            '<div class="guide sos"><b>🚨 IMMEDIATE ACTION REQUIRED</b><br>'
            'If you are at the scene: ensure personal safety first, call <b>108</b> for an '
            'ambulance, and do not move injured persons unless they are in immediate danger.</div>')
    else:
        st.html(
            '<div class="guide safe"><b>✅ All clear</b><br>'
            'No accident indicators were found, so no emergency alerts were dispatched.</div>')


def empty_state(text):
    st.html(f'<div class="empty"><div class="g">🛰️</div><p>{text}</p></div>')


# ─────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────
model = load_model()
_loc = gps_fix()

model_ok = model is not None
pills = [
    ("ok" if model_ok else "off", "Detection model", "YOLO ready" if model_ok else "not found"),
    ("ok" if EMAIL_CONFIGURED else "off", "Alert channel",
     "email active" if EMAIL_CONFIGURED else "not configured"),
    ("ok" if _loc else "warn", "Location",
     "device GPS" if _loc else "not acquired"),
    ("ok", "Threshold", f"{CONFIDENCE_THRESHOLD*100:.0f}% · {ALERT_RADIUS_KM:.0f} km radius"),
]
pill_html = "".join(
    f'<span class="pill"><span class="dot dot-{d}"></span>{label} <b>{val}</b></span>'
    for d, label, val in pills
)

st.html(f"""
<div class="hero">
  <div class="hero-badge"><i></i>AI Emergency Response</div>
  <h1>AccidentWatch AI</h1>
  <p>Real-time road accident detection powered by a custom-trained YOLO model.
  Upload footage to analyse the scene and dispatch geo-tagged alerts with photo
  evidence to the nearest hospitals automatically.</p>
  <div class="pills">{pill_html}</div>
</div>
""")

if not model_ok:
    st.error(f"Detection model not found at `{MODEL_PATH}` — detection is disabled.")

# ─────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────
tab_image, tab_video = st.tabs(["📷  Image Analysis", "🎬  Video Analysis"])

# ── IMAGE ───────────────────────────────────────────────────────────────
with tab_image:
    left, right = st.columns([1, 1], gap="large")

    with left:
        with card("📷", "Upload Image", "img-up"):
            # Uploader sits outside the form so the preview appears immediately.
            img_file = st.file_uploader("Image", type=["jpg", "jpeg", "png"],
                                        label_visibility="collapsed", key="img_up")
            if img_file:
                st.image(img_file, use_container_width=True)
            else:
                st.caption("JPG, JPEG or PNG · analysed on device")

        img_lat, img_lon, img_address, run_image = location_form(
            "img", "🚨  Detect Accident", disabled=(not model_ok or img_file is None)
        )

    with right:
        if run_image and img_file is not None:
            with st.spinner("Running detection..."):
                results = model.predict(Image.open(img_file).convert("RGB"),
                                        conf=CONFIDENCE_THRESHOLD, verbose=False)
                boxes = results[0].boxes
                detected = len(boxes) > 0

                objects, confidence, annotated_rgb = [], 0.0, None
                if detected:
                    for box in boxes:
                        objects.append({
                            "type": model.names[int(box.cls)],
                            "confidence": round(float(box.conf), 4),
                        })
                    confidence = max(float(b.conf) for b in boxes)
                    annotated_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

                status_lines = []
                if detected:
                    details_html = f"""
<div class="row"><span class="lbl">Detection:</span><span class="val">Image Analysis</span></div>
<div class="row"><span class="lbl">Objects Found:</span><span class="val">{len(objects)} object(s)</span></div>
<div class="row"><span class="lbl">Peak Confidence:</span><span class="val">{confidence*100:.1f}%</span></div>
<ul style="margin:10px 0 0;padding-left:18px;font-size:13px;color:#444;">
{''.join(f"<li><strong>{o['type']}</strong> — {o['confidence']*100:.1f}% confidence</li>" for o in objects)}
</ul>"""
                    status_lines = dispatch_alerts(
                        "image", details_html, nearest_hospitals(img_lat, img_lon),
                        jpeg_bytes(annotated_rgb), "accident_evidence.jpg",
                        img_address, img_lat, img_lon,
                    )

            with card("🔍", "Detection Report", "img-out"):
                show_verdict(detected, confidence)
                show_stats([
                    ("Objects", len(objects), False),
                    ("Confidence", f"{confidence*100:.0f}%", detected),
                    ("Alerts", sum(1 for k, _ in status_lines if k == "ok"), False),
                ])
                show_objects(objects)
                if annotated_rgb is not None:
                    st.image(annotated_rgb, use_container_width=True, caption="Annotated detection")
                show_alerts(status_lines)
                show_guidance(detected)
        else:
            with card("🔍", "Detection Report", "img-idle"):
                empty_state("Upload an image and run detection to see the analysis here.")

# ── VIDEO ───────────────────────────────────────────────────────────────
with tab_video:
    left, right = st.columns([1, 1], gap="large")

    with left:
        with card("🎬", "Upload Video", "vid-up"):
            vid_file = st.file_uploader("Video", type=["mp4", "avi", "mov"],
                                       label_visibility="collapsed", key="vid_up")
            if vid_file:
                st.video(vid_file)
            else:
                st.caption(f"MP4, AVI or MOV · every {FRAME_SKIP}th frame analysed")

        vid_lat, vid_lon, vid_address, run_video = location_form(
            "vid", "🎬  Analyze Video", disabled=(not model_ok or vid_file is None)
        )

    with right:
        if run_video and vid_file is not None:
            progress_slot = st.empty()
            # getvalue() is pointer-independent, so st.video() above cannot leave
            # us writing an empty temp file.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(vid_file.getvalue())
                video_path = tmp.name

            cap = None
            try:
                cap = cv2.VideoCapture(video_path)
                frame_idx = total_frames = analyzed_frames = accident_frames = 0
                confidence, sample_rgb, evidence_frames = 0.0, None, []
                frame_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                bar = progress_slot.progress(0.0, text="Analysing frames...")

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    total_frames += 1
                    if frame_idx % FRAME_SKIP == 0:
                        analyzed_frames += 1
                        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                        if len(results[0].boxes) > 0:
                            accident_frames += 1
                            conf_val = max(float(b.conf) for b in results[0].boxes)
                            ann = results[0].plot()
                            if len(evidence_frames) < MAX_EVIDENCE_FRAMES:
                                evidence_frames.append(ann.copy())
                            if conf_val > confidence:
                                confidence = conf_val
                                sample_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                        if frame_hint:
                            bar.progress(min(total_frames / frame_hint, 1.0),
                                         text=f"Analysing frame {total_frames} of {frame_hint}")
                    frame_idx += 1
            finally:
                if cap is not None:
                    cap.release()
                progress_slot.empty()
                try:
                    os.unlink(video_path)
                except OSError:
                    pass

            detected = accident_frames > 0
            collage_rgb = build_collage(evidence_frames)

            status_lines = []
            if detected:
                details_html = f"""
<div class="row"><span class="lbl">Detection:</span><span class="val">Video Analysis</span></div>
<div class="row"><span class="lbl">Total Frames:</span><span class="val">{total_frames}</span></div>
<div class="row"><span class="lbl">Frames Analyzed:</span><span class="val">{analyzed_frames} (every {FRAME_SKIP}th)</span></div>
<div class="row"><span class="lbl">Accident Frames:</span><span class="val"><strong style="color:#dc2626;">{accident_frames}</strong></span></div>
<div class="row"><span class="lbl">Peak Confidence:</span><span class="val">{confidence*100:.1f}%</span></div>
<div class="row"><span class="lbl">Evidence Collage:</span><span class="val">{len(evidence_frames)} frame(s)</span></div>"""
                status_lines = dispatch_alerts(
                    "video", details_html, nearest_hospitals(vid_lat, vid_lon),
                    jpeg_bytes(collage_rgb) or jpeg_bytes(sample_rgb),
                    "accident_evidence_collage.jpg", vid_address, vid_lat, vid_lon,
                )

            with card("🔍", "Detection Report", "vid-out"):
                show_verdict(detected, confidence,
                             f"{accident_frames} of {analyzed_frames} analysed frames flagged.")
                show_stats([
                    ("Total frames", total_frames, False),
                    ("Analysed", analyzed_frames, False),
                    ("Flagged", accident_frames, detected),
                    ("Evidence", len(evidence_frames), False),
                ])
                if sample_rgb is not None:
                    st.image(sample_rgb, use_container_width=True, caption="Highest-confidence frame")
                if collage_rgb is not None:
                    st.image(collage_rgb, use_container_width=True,
                             caption="Multi-frame evidence collage")
                show_alerts(status_lines)
                show_guidance(detected)
        else:
            with card("🔍", "Detection Report", "vid-idle"):
                empty_state("Upload a video and run analysis to see the frame-by-frame report here.")

st.html('<div class="foot">Accident Detection &amp; Alert System · '
        'powered by <b>YOLO</b> object detection</div>')
