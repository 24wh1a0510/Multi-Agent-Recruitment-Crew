"""
frontend/streamlit_ui.py
-------------------------
Premium Streamlit UI for the Multi-Agent Recruitment Crew.

Talks to the FastAPI backend (app.py) over HTTP so the LangGraph
pipeline runs server-side. Run the backend first:

    uvicorn app:app --reload

Then run this UI:

    streamlit run frontend/streamlit_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# Allow `from config import settings` etc. when run from the frontend/ dir.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

BACKEND_URL = settings.backend_url

st.set_page_config(
    page_title="Multi-Agent Recruitment Crew",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME / GLASSMORPHISM CSS
# ---------------------------------------------------------------------------

if "theme" not in st.session_state:
    st.session_state.theme = "dark"


def inject_css(theme: str) -> None:
    if theme == "dark":
        bg = "linear-gradient(135deg,#0f0c29,#302b63,#24243e)"
        card_bg = "rgba(255,255,255,0.08)"
        sidebar_bg = "rgba(20,16,60,0.92)"
        border = "rgba(255,255,255,0.18)"
        text = "#FFFFFF"
        subtext = "#D0D0E8"
        sidebar_text = "#FFFFFF"
        sidebar_subtext = "#C8C8E0"
        nav_text = "#FFFFFF"
        nav_active_text = "#a78bfa"
        nav_hover_bg = "rgba(127,90,240,0.25)"
        nav_active_bg = "linear-gradient(90deg, rgba(127,90,240,0.5), rgba(44,182,125,0.4))"
        nav_active_border = "#a78bfa"
        label_color = "#C8C8E0"
        btn_default_bg = "transparent"
    else:
        bg = "linear-gradient(135deg,#e0eafc,#cfdef3,#f5f7fa)"
        card_bg = "rgba(255,255,255,0.80)"
        sidebar_bg = "rgba(230,236,252,0.97)"
        border = "rgba(180,180,220,0.6)"
        text = "#0f0c29"
        subtext = "#2a2a4a"
        sidebar_text = "#1a1040"
        sidebar_subtext = "#4a4a7a"
        nav_text = "#1a1040"
        nav_active_text = "#5b21b6"
        nav_hover_bg = "rgba(127,90,240,0.12)"
        nav_active_bg = "linear-gradient(90deg, rgba(127,90,240,0.18), rgba(44,182,125,0.15))"
        nav_active_border = "#7f5af0"
        label_color = "#2a2a4a"
        btn_default_bg = "transparent"

    st.markdown(
        f"""
        <style>
        /* ── Base app ──────────────────────────────────────────────── */
        .stApp {{
            background: {bg};
            background-attachment: fixed;
        }}

        /* Force ALL text elements to the right color */
        .stApp, .stApp p, .stApp span, .stApp div,
        .stApp li, .stApp label, .stApp h1, .stApp h2,
        .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: {text} !important;
            font-size: 16px;
        }}

        /* Markdown paragraphs */
        .stMarkdown p, .stMarkdown li, .stMarkdown span {{
            color: {text} !important;
            font-size: 16px !important;
            line-height: 1.7 !important;
        }}

        /* Headings */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: {text} !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }}

        /* ── Sidebar ───────────────────────────────────────────────── */
        [data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            backdrop-filter: blur(18px);
            border-right: 1px solid {border};
            min-width: 240px !important;
            width: 240px !important;
        }}
        /* Prevent sidebar content from being clipped */
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarUserContent"] {{
            width: 100% !important;
            min-width: 0 !important;
            overflow-x: hidden !important;
            padding: 1rem 0.6rem !important;
            box-sizing: border-box !important;
        }}
        /* Base sidebar text — use sidebar_text (theme-aware) */
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stMarkdown {{
            color: {sidebar_text} !important;
            font-size: 15px !important;
            font-weight: 500 !important;
        }}
        /* ── Sidebar radio nav — theme-aware ── */
        [data-testid="stSidebar"] .stRadio label p,
        [data-testid="stSidebar"] .stRadio label span,
        [data-testid="stSidebar"] .stRadio label div,
        [data-testid="stSidebar"] [data-testid="stRadioLabel"],
        [data-testid="stSidebar"] [data-testid="stRadioLabel"] p,
        [data-testid="stSidebar"] [class*="radioLabel"],
        [data-testid="stSidebar"] [class*="radio"] label,
        [data-testid="stSidebar"] [class*="radio"] p {{
            color: {nav_text} !important;
            font-size: 16px !important;
            font-weight: 600 !important;
        }}
        /* Navigation section heading */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: {sidebar_text} !important;
            font-size: 15px !important;
            font-weight: 700 !important;
        }}
        /* Sidebar text catch-all — theme-aware, NOT hardcoded white */
        [data-testid="stSidebar"] * {{
            color: {sidebar_text} !important;
        }}
        /* Override catch-all for caption/small */
        [data-testid="stSidebar"] .stCaption *,
        [data-testid="stSidebar"] small {{
            color: {sidebar_subtext} !important;
            font-size: 13px !important;
        }}
        /* Sidebar section headings */
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {{
            color: {sidebar_text} !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
        }}
        /* Sidebar caption/footer text */
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] small {{
            color: {sidebar_subtext} !important;
            font-size: 13px !important;
        }}
        /* Toggle label */
        [data-testid="stSidebar"] .stCheckbox label,
        [data-testid="stSidebar"] .stToggle label {{
            color: {sidebar_text} !important;
            font-size: 15px !important;
        }}

        /* ── Metric widgets ────────────────────────────────────────── */
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {{
            color: {label_color} !important;
            font-size: 14px !important;
            font-weight: 600 !important;
        }}
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div {{
            color: {text} !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
        }}

        /* ── Info / warning / success boxes ───────────────────────── */
        .stAlert p {{ color: #1a1a2e !important; font-size: 15px !important; }}

        /* ── File uploader ─────────────────────────────────────────── */
        /* Outer label above the dropzone — must match page text color */
        [data-testid="stFileUploader"] > label,
        [data-testid="stFileUploader"] > label p,
        [data-testid="stFileUploader"] > label span {{
            color: {text} !important;
            font-size: 16px !important;
            font-weight: 600 !important;
        }}
        /* Everything INSIDE the dropzone box — dark text on white bg */
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] div,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] + div span,
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] + div p {{
            color: #1a1a2e !important;
            font-size: 15px !important;
            font-weight: 500 !important;
        }}
        /* File uploader dropzone background */
        [data-testid="stFileUploaderDropzone"] {{
            background: rgba(255,255,255,0.92) !important;
            border: 2px dashed #7f5af0 !important;
            border-radius: 12px !important;
        }}

        /* ── Selectbox ─────────────────────────────────────────────── */
        .stSelectbox label, .stSelectbox span,
        [data-testid="stSelectbox"] label {{
            color: {text} !important;
            font-size: 15px !important;
            font-weight: 600 !important;
        }}
        [data-testid="stSelectbox"] div[data-baseweb="select"] span {{
            color: #1a1a2e !important;
            font-size: 15px !important;
        }}

        /* ── Success / info / warning / error message text ─────────── */
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span {{
            color: #1a1a2e !important;
            font-size: 15px !important;
        }}

        /* ── Text area / input ─────────────────────────────────────── */
        .stTextArea label, .stTextInput label {{
            color: {text} !important;
            font-size: 15px !important;
            font-weight: 600 !important;
        }}
        .stTextArea textarea {{
            color: #1a1a2e !important;
            background: rgba(255,255,255,0.92) !important;
            font-size: 15px !important;
            border: 1px solid {border} !important;
        }}
        /* Input text box */
        .stTextInput input {{
            color: #1a1a2e !important;
            background: rgba(255,255,255,0.92) !important;
            font-size: 15px !important;
        }}

        /* ── Expander ──────────────────────────────────────────────── */
        .streamlit-expanderHeader p,
        .streamlit-expanderHeader span {{
            color: {text} !important;
            font-size: 15px !important;
            font-weight: 600 !important;
        }}

        /* ── Caption / small text ──────────────────────────────────── */
        .stCaption, .stCaption p {{
            color: {label_color} !important;
            font-size: 13px !important;
        }}

        /* ── Glass cards ───────────────────────────────────────────── */
        .glass-card {{
            background: {card_bg};
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid {border};
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.22);
        }}

        /* ── Hero ──────────────────────────────────────────────────── */
        .hero-title {{
            font-size: 2.6rem;
            font-weight: 800;
            background: linear-gradient(90deg,#a78bfa,#34d399,#fbbf24);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }}
        .hero-subtitle {{
            color: {subtext} !important;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
            font-weight: 500;
        }}

        /* ── Section titles ────────────────────────────────────────── */
        .section-title {{
            font-size: 1.2rem;
            font-weight: 700;
            color: {text} !important;
            margin-bottom: 0.6rem;
        }}

        /* ── Badges ────────────────────────────────────────────────── */
        .badge {{
            display: inline-block;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            margin-right: 0.4rem;
        }}
        .badge-hire {{ background: linear-gradient(90deg,#2cb67d,#16a34a); color: white !important; }}
        .badge-interview {{ background: linear-gradient(90deg,#3b82f6,#6366f1); color: white !important; }}
        .badge-hold {{ background: linear-gradient(90deg,#f59e0b,#f97316); color: white !important; }}
        .badge-reject {{ background: linear-gradient(90deg,#ef4444,#dc2626); color: white !important; }}
        .badge-agent {{ background: rgba(167,139,250,0.25); color: {text} !important; border: 1px solid rgba(167,139,250,0.5);}}

        /* ── Timeline ──────────────────────────────────────────────── */
        .timeline-step {{
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.6rem 1rem;
            border-radius: 12px;
            background: {card_bg};
            border: 1px solid {border};
            margin-bottom: 0.5rem;
        }}
        .timeline-step b, .timeline-step div {{
            color: {text} !important;
            font-size: 15px !important;
        }}
        .timeline-dot {{
            width: 12px; height: 12px; border-radius: 50%;
            background: linear-gradient(90deg,#a78bfa,#34d399);
            flex-shrink: 0;
        }}

        /* ── Buttons ───────────────────────────────────────────────── */
        div.stButton > button {{
            background: linear-gradient(90deg,#7f5af0,#2cb67d);
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.65rem 1.5rem;
            font-weight: 700;
            font-size: 15px !important;
            transition: transform 0.15s ease;
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px);
            filter: brightness(1.1);
        }}
        div.stButton > button:disabled {{
            opacity: 0.45;
        }}

        /* ── Score pill ────────────────────────────────────────────── */
        .score-pill {{
            font-size: 1.8rem;
            font-weight: 800;
            color: {text} !important;
        }}

        /* ── Metric label ──────────────────────────────────────────── */
        .metric-label {{
            color: {label_color} !important;
            font-size: 13px !important;
            font-weight: 500;
        }}

        /* ── Code blocks ───────────────────────────────────────────── */
        .stCode code, code {{
            color: #e2e8f0 !important;
            font-size: 13px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css(st.session_state.theme)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

for key, default in {
    "resume_bytes": None,
    "resume_filename": None,
    "resume_queue": [],
    "job_description": "",
    "result_state": None,
    "run_history": [],
    "running": False,
    "page": "Home",
    "interview_questions": [],
    "schedule_confirmed": False,
    "voice_transcript": "",
    "voice_evaluation": "",
    "audio_frames": [],
    "webrtc_audio_frames": [],      # persistent buffer for WebRTC frames across reruns
    "webrtc_was_playing": False,    # tracks play→stop transition for transcription trigger
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

DECISION_BADGE = {
    "Hire": "badge-hire",
    "Interview": "badge-interview",
    "Hold": "badge-hold",
    "Reject": "badge-reject",
}

# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION  (styled vertical nav — matches dropdown sidebar design)
# ---------------------------------------------------------------------------

NAV_PAGES = [
    ("🏠", "Home"),
    ("📄", "Upload Resume"),
    ("📝", "Job Description"),
    ("🌐", "Post JD to Portals"),
    ("🚀", "Run Crew"),
    ("🧾", "Execution Logs"),
    ("📋", "Final Report"),
    ("🎯", "Interview Questions"),
    ("🎙️", "Voice Interview"),
    ("📅", "Schedule Interview"),
]

# Inject custom sidebar nav styles — theme-aware
_nav_theme = st.session_state.theme
_nav_brand_color = "#FFFFFF" if _nav_theme == "dark" else "#1a1040"
_nav_item_label_color = "#FFFFFF" if _nav_theme == "dark" else "#1a1040"
_nav_active_label_color = "#a78bfa" if _nav_theme == "dark" else "#5b21b6"
_nav_brand_border = "rgba(255,255,255,0.15)" if _nav_theme == "dark" else "rgba(127,90,240,0.25)"
_nav_status_border = "rgba(255,255,255,0.15)" if _nav_theme == "dark" else "rgba(127,90,240,0.2)"
_nav_section_heading_color = "rgba(200,200,220,0.8)" if _nav_theme == "dark" else "rgba(90,60,160,0.7)"
_nav_hover_bg = "rgba(127,90,240,0.22)" if _nav_theme == "dark" else "rgba(127,90,240,0.10)"
_nav_active_bg = "linear-gradient(90deg, rgba(127,90,240,0.45), rgba(44,182,125,0.35))" if _nav_theme == "dark" else "linear-gradient(90deg, rgba(127,90,240,0.15), rgba(44,182,125,0.12))"
_nav_active_border_color = "#a78bfa" if _nav_theme == "dark" else "#7f5af0"

st.markdown(
    f"""
    <style>
    /* Hide default radio widget in sidebar nav section */
    [data-testid="stSidebar"] .nav-radio {{ display: none !important; }}

    /* Sidebar brand header */
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0 4px 14px 4px;
        border-bottom: 1px solid {_nav_brand_border};
        margin-bottom: 10px;
    }}
    .sidebar-brand-icon {{
        font-size: 1.5rem;
    }}
    .sidebar-brand-text {{
        font-size: 1rem;
        font-weight: 800;
        color: {_nav_brand_color} !important;
        letter-spacing: 0.01em;
    }}

    /* Nav item base */
    .nav-item {{
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 9px 14px;
        border-radius: 10px;
        cursor: pointer;
        transition: background 0.18s ease, transform 0.12s ease;
        margin-bottom: 2px;
        text-decoration: none !important;
    }}
    .nav-item:hover {{
        background: {_nav_hover_bg};
        transform: translateX(3px);
    }}
    .nav-item-active {{
        background: {_nav_active_bg} !important;
        border-left: 3px solid {_nav_active_border_color};
    }}
    .nav-item-icon {{
        font-size: 1.1rem;
        min-width: 22px;
        text-align: center;
    }}
    .nav-item-label {{
        font-size: 14.5px !important;
        font-weight: 600 !important;
        color: {_nav_item_label_color} !important;
    }}
    .nav-item-active .nav-item-label {{
        color: {_nav_active_label_color} !important;
    }}

    /* Pipeline status section */
    .nav-status-section {{
        border-top: 1px solid {_nav_status_border};
        margin-top: 10px;
        padding-top: 12px;
    }}
    .nav-section-heading {{
        font-size: 11px !important;
        font-weight: 700 !important;
        color: {_nav_section_heading_color} !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0 14px;
        margin-bottom: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    # Brand header
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-brand-icon">🧭</div>'
        '<div class="sidebar-brand-text">Recruitment Crew</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Theme toggle
    theme_choice = st.toggle("🌙 Dark mode", value=(st.session_state.theme == "dark"))
    new_theme = "dark" if theme_choice else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown('<div style="margin: 8px 0 4px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-section-heading">Navigation</div>', unsafe_allow_html=True)

    # Render nav buttons using st.button with custom HTML labels
    for icon, label in NAV_PAGES:
        is_active = (st.session_state.page == label)
        active_cls = "nav-item-active" if is_active else ""
        # Render a clickable styled button via st.button with markdown hack
        btn_html = (
            f'<div class="nav-item {active_cls}">'
            f'<span class="nav-item-icon">{icon}</span>'
            f'<span class="nav-item-label">{label}</span>'
            f'</div>'
        )
        if st.button(
            f"{icon}  {label}",
            key=f"nav_{label}",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.page = label
            st.rerun()

    # Override button styling to look like custom nav items — theme-aware
    st.markdown(
        f"""
        <style>
        /* Nav button overrides — make all sidebar buttons look like nav items */
        [data-testid="stSidebar"] div.stButton {{
            width: 100% !important;
        }}
        [data-testid="stSidebar"] div.stButton > button {{
            background: transparent !important;
            border: none !important;
            border-radius: 10px !important;
            color: {_nav_item_label_color} !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 8px 10px !important;
            transition: background 0.18s ease, transform 0.12s ease;
            margin-bottom: 1px !important;
            box-shadow: none !important;
            justify-content: flex-start !important;
            white-space: nowrap !important;
            overflow: visible !important;
            width: 100% !important;
            min-width: 0 !important;
        }}
        /* Ensure inner p/span inside button don't wrap or clip */
        [data-testid="stSidebar"] div.stButton > button p,
        [data-testid="stSidebar"] div.stButton > button span {{
            color: {_nav_item_label_color} !important;
            white-space: nowrap !important;
            overflow: visible !important;
            font-size: 14px !important;
        }}
        [data-testid="stSidebar"] div.stButton > button:hover {{
            background: {_nav_hover_bg} !important;
            transform: translateX(2px) !important;
            color: {_nav_item_label_color} !important;
        }}
        /* Active page button */
        [data-testid="stSidebar"] div.stButton[data-nav-active="true"] > button,
        [data-testid="stSidebar"] div[data-testid="stButtonGroup"] button[kind="secondary"][aria-pressed="true"] {{
            background: {_nav_active_bg} !important;
            border-left: 3px solid {_nav_active_border_color} !important;
            color: {_nav_active_label_color} !important;
        }}
        /* Active nav button — JS workaround */
        [data-testid="stSidebar"] div.stButton > button[data-active="true"] {{
            background: {_nav_active_bg} !important;
            border-left: 3px solid {_nav_active_border_color} !important;
            color: {_nav_active_label_color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Highlight active page via JS injection
    active_page = st.session_state.page
    nav_labels_js = [label for _, label in NAV_PAGES]
    active_idx = nav_labels_js.index(active_page) if active_page in nav_labels_js else 0

    st.markdown(
        f"""
        <script>
        (function() {{
            function highlightActive() {{
                const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
                if (!sidebar) return;
                const buttons = sidebar.querySelectorAll('div.stButton > button');
                const activeIdx = {active_idx};
                const defaultColor = '{_nav_item_label_color}';
                const activeColor = '{_nav_active_label_color}';
                const activeBg = '{_nav_active_bg}';
                const activeBorder = '3px solid {_nav_active_border_color}';
                buttons.forEach((btn, i) => {{
                    btn.style.background = '';
                    btn.style.borderLeft = '';
                    btn.style.color = defaultColor;
                }});
                if (buttons[activeIdx]) {{
                    buttons[activeIdx].style.background = activeBg;
                    buttons[activeIdx].style.borderLeft = activeBorder;
                    buttons[activeIdx].style.color = activeColor;
                }}
            }}
            setTimeout(highlightActive, 300);
            setTimeout(highlightActive, 800);
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Pipeline status
    st.markdown('<div class="nav-status-section">', unsafe_allow_html=True)
    st.markdown('<div class="nav-section-heading">Pipeline Status</div>', unsafe_allow_html=True)
    if st.session_state.run_history:
        st.success(f"{len(st.session_state.run_history)} candidate(s) evaluated")
        if st.session_state.resume_queue:
            st.info(f"{len(st.session_state.resume_queue)} resume(s) in queue")
    else:
        st.info("No run yet")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"Backend: `{BACKEND_URL}`")

# Resolve page from session state (set by nav buttons above)
page = st.session_state.page


def backend_healthy() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# PAGE: HOME
# ---------------------------------------------------------------------------

def page_home() -> None:
    st.markdown('<div class="hero-title">Multi-Agent Recruitment Crew</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">LangGraph-orchestrated agents that read, score, verify, '
        'and decide -- collaboratively, transparently, and with human escalation built in.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    agent_info = [
        ("📄", "Resume Analyst", "Extracts structured candidate data from the uploaded PDF."),
        ("📊", "Scoring Agent", "Rubric-scores the candidate against the job description."),
        ("🛡️", "Verification Agent", "Independently double-checks borderline scores for bias & injection."),
        ("✅", "Decision Agent", "Produces the final Hire / Interview / Hold / Reject call."),
    ]
    for col, (icon, name, desc) in zip(cols, agent_info):
        with col:
            st.markdown(
                f'<div class="glass-card"><div style="font-size:1.8rem">{icon}</div>'
                f'<div class="section-title">{name}</div>'
                f'<div class="metric-label">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
1. **Upload Resume** — provide a candidate's resume as a PDF.
2. **Job Description** — paste or load the sample job description.
3. **Run Crew** — the LangGraph pipeline executes Supervisor → Resume Analyst → Scoring Agent →
   (conditionally) Verification Agent → Decision Agent.
4. **Execution Logs** — watch the live agent timeline, retries, and token/timing metrics.
5. **Final Report** — see the hiring decision, reasoning, and a full shared-state viewer.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if not backend_healthy():
        st.warning(
            f"⚠️ Backend not reachable at `{BACKEND_URL}`. Start it with `uvicorn app:app --reload`."
        )


# ---------------------------------------------------------------------------
# PAGE: UPLOAD RESUME
# ---------------------------------------------------------------------------

def page_upload_resume() -> None:
    st.markdown('<div class="section-title">📄 Upload Resumes</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload one or more candidate resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        existing_names = {r["filename"] for r in st.session_state.resume_queue}
        added = 0
        for f in uploaded_files:
            if f.name not in existing_names:
                st.session_state.resume_queue.append({
                    "filename": f.name,
                    "bytes": f.read(),
                })
                added += 1
        if added:
            st.success(f"Added {added} resume(s) to queue.")

    # Sample resume button
    if st.button("Add bundled sample resume"):
        sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_resume.pdf"
        if sample_path.exists():
            existing_names = {r["filename"] for r in st.session_state.resume_queue}
            if "sample_resume.pdf" not in existing_names:
                st.session_state.resume_queue.append({
                    "filename": "sample_resume.pdf",
                    "bytes": sample_path.read_bytes(),
                })
                st.success("Sample resume added to queue.")
            else:
                st.info("Sample resume already in queue.")
        else:
            st.error("Sample resume not found in data/.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Show queue
    if st.session_state.resume_queue:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">Resume Queue ({len(st.session_state.resume_queue)} file(s))</div>', unsafe_allow_html=True)
        for i, r in enumerate(st.session_state.resume_queue):
            col_name, col_size, col_remove = st.columns([4, 2, 1])
            with col_name:
                st.write(f"**{i+1}. {r['filename']}**")
            with col_size:
                st.write(f"{len(r['bytes'])/1024:.1f} KB")
            with col_remove:
                if st.button("✕", key=f"remove_{i}"):
                    st.session_state.resume_queue.pop(i)
                    st.rerun()

        if st.button("Clear all resumes"):
            st.session_state.resume_queue = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No resumes in queue yet. Upload PDFs above.")


# ---------------------------------------------------------------------------
# PAGE: JOB DESCRIPTION
# ---------------------------------------------------------------------------

def page_job_description() -> None:
    st.markdown('<div class="section-title">📝 Enter Job Description</div>', unsafe_allow_html=True)

    # ── JD from job portal URL ────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🌐 Import from Job Portal</div>', unsafe_allow_html=True)
    st.caption("Paste a job URL from Naukri, LinkedIn, Internshala, Unstop, Indeed, Glassdoor, Wellfound, Workday, etc.")

    portal_url = st.text_input(
        "Job posting URL",
        placeholder="https://www.naukri.com/job-listings-... or https://unstop.com/jobs/...",
    )

    if st.button("📥 Fetch JD from URL"):
        if not portal_url.strip():
            st.warning("Please enter a URL first.")
        else:
            with st.spinner("Fetching job description from URL..."):
                try:
                    import urllib.request
                    import html
                    import re as _re

                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                    req = urllib.request.Request(portal_url.strip(), headers=headers)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        raw_html = resp.read().decode("utf-8", errors="ignore")

                    # Strip HTML tags
                    text = _re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=_re.DOTALL | _re.IGNORECASE)
                    text = _re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=_re.DOTALL | _re.IGNORECASE)
                    text = _re.sub(r"<[^>]+>", " ", text)
                    text = html.unescape(text)
                    text = _re.sub(r"\s+", " ", text).strip()

                    # Try to find the relevant JD section (heuristic: look for "responsibilities" or "requirements")
                    keywords = ["responsibilities", "requirements", "qualifications",
                                "about the role", "job description", "what you'll do",
                                "about the job", "role summary"]
                    start_idx = len(text)
                    for kw in keywords:
                        idx = text.lower().find(kw)
                        if idx != -1 and idx < start_idx:
                            start_idx = max(0, idx - 100)

                    extracted = text[start_idx:start_idx + 4000] if start_idx < len(text) else text[:4000]

                    if len(extracted.strip()) < 100:
                        st.warning("Could not extract enough text. The site may block scrapers. Try pasting the JD manually below.")
                    else:
                        st.session_state.job_description = extracted.strip()
                        st.success(f"Fetched {len(extracted)} characters from URL.")

                except Exception as exc:
                    st.error(f"Could not fetch URL: {exc}. Try pasting the JD manually.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Manual JD entry ───────────────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✏️ Job Description Text</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Load sample JD"):
            try:
                r = requests.get(f"{BACKEND_URL}/sample-jd", timeout=5)
                if r.status_code == 200:
                    st.session_state.job_description = r.json().get("job_description", "")
                    st.success("Sample JD loaded.")
            except requests.RequestException as exc:
                st.error(f"Could not reach backend: {exc}")
        if st.button("Clear JD"):
            st.session_state.job_description = ""
            st.rerun()

    with col1:
        st.session_state.job_description = st.text_area(
            "Paste or edit job description here",
            value=st.session_state.job_description,
            height=340,
            placeholder="Paste the job description here, or use the URL importer above...",
        )

    if st.session_state.job_description:
        st.caption(f"{len(st.session_state.job_description)} characters | {len(st.session_state.job_description.split())} words")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: RUN CREW
# ---------------------------------------------------------------------------

def page_run_crew() -> None:
    st.markdown('<div class="section-title">🚀 Run Multi-Agent Crew</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    has_resumes = bool(st.session_state.resume_queue)
    has_jd = bool(st.session_state.job_description.strip())
    ready = has_resumes and has_jd

    if not has_resumes:
        st.warning("No resumes in queue. Go to **Upload Resume** and add at least one PDF.")
    if not has_jd:
        st.warning("No job description. Go to **Job Description** and enter one.")

    if ready:
        st.info(f"Ready to evaluate **{len(st.session_state.resume_queue)} candidate(s)** against the job description.")

    if st.button("▶️ Run Recruitment Crew", disabled=not ready):
        st.session_state.running = True
        progress = st.progress(0)
        total = len(st.session_state.resume_queue)

        for idx, resume in enumerate(st.session_state.resume_queue):
            with st.spinner(f"Processing {resume['filename']} ({idx+1}/{total})..."):
                try:
                    files = {
                        "resume": (
                            resume["filename"],
                            resume["bytes"],
                            "application/pdf",
                        )
                    }
                    data = {"job_description": st.session_state.job_description}
                    resp = requests.post(f"{BACKEND_URL}/run", files=files, data=data, timeout=180)
                    if resp.status_code == 200:
                        state = resp.json()["state"]
                        # Store in history — append, don't overwrite
                        st.session_state.run_history.append({
                            "filename": resume["filename"],
                            "state": state,
                        })
                        # Also set result_state to latest for Execution Logs / Final Report
                        st.session_state.result_state = state
                        st.success(f"✅ {resume['filename']} — done.")
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"❌ {resume['filename']}: {detail}")
                except requests.RequestException as exc:
                    st.error(f"❌ {resume['filename']}: Could not reach backend — {exc}")

            progress.progress((idx + 1) / total)

        st.session_state.running = False
        st.session_state.resume_queue = []   # clear queue after run
        st.success(f"All {total} candidate(s) processed. View results in **Final Report**.")
        st.session_state.page = "Final Report"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.job_description:
        with st.expander("Preview job description (first 500 chars)"):
            st.code(st.session_state.job_description[:500])


# ---------------------------------------------------------------------------
# PAGE: EXECUTION LOGS
# ---------------------------------------------------------------------------

def page_execution_logs() -> None:
    st.markdown('<div class="section-title">🧾 Execution Logs & Agent Timeline</div>', unsafe_allow_html=True)

    state = st.session_state.result_state
    if not state:
        st.info("Run the crew first on the **Run Crew** page.")
        return

    path = state.get("execution_path", [])
    revision_count = state.get("revision_count", 0)
    escalated = state.get("escalated", False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Agent steps", len(path))
    m2.metric("Retries used", revision_count)
    m3.metric("Escalated", "Yes" if escalated else "No")
    total_tokens = sum((t.get("tokens_used") or 0) for t in state.get("logs", []) if isinstance(t, dict))
    m4.metric("Est. tokens used", total_tokens)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Execution Path (Animated Timeline)</div>', unsafe_allow_html=True)
    for i, agent in enumerate(path, start=1):
        st.markdown(
            f'<div class="timeline-step"><div class="timeline-dot"></div>'
            f'<div><b>{i}. {agent}</b></div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    timings = state.get("timings_ms", [])
    if timings:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Execution Time per Agent</div>', unsafe_allow_html=True)
        st.bar_chart({t["agent"]: t["duration_ms"] for t in timings if isinstance(t, dict)})
        st.markdown("</div>", unsafe_allow_html=True)

    errors = state.get("errors", [])
    if errors:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚠️ Errors</div>', unsafe_allow_html=True)
        for e in errors:
            st.error(f"[{e.get('agent')}] {e.get('message')}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Full Log Stream</div>', unsafe_allow_html=True)
    for log in state.get("logs", []):
        level = log.get("level", "INFO")
        prefix = "🔴" if level == "ERROR" else ("🟡" if level == "WARNING" else "🟢")
        st.markdown(
            f"{prefix} `{log.get('timestamp','')[11:19]}` **[{log.get('agent')}]** {log.get('message')}"
            + (f"  _(⏱ {log['duration_ms']}ms)_" if log.get("duration_ms") else "")
            + (f"  _(🔤 {log['tokens_used']} tok)_" if log.get("tokens_used") else "")
        )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔍 Shared State Viewer (raw JSON)"):
        st.json(state)


# ---------------------------------------------------------------------------
# PAGE: FINAL REPORT
# ---------------------------------------------------------------------------

def page_final_report() -> None:
    st.markdown('<div class="section-title">📋 Final Hiring Dashboard</div>', unsafe_allow_html=True)

    # Show all runs from history
    history = st.session_state.run_history
    if not history:
        st.info("Run the crew first on the **Run Crew** page.")
        return

    # Candidate selector when multiple runs exist
    if len(history) > 1:
        candidate_names = [
            f"{i+1}. {r['filename']} — {(r['state'].get('decision') or {}).get('decision', 'N/A')}"
            for i, r in enumerate(history)
        ]
        selected_idx = st.selectbox(
            "Select candidate to view:",
            range(len(history)),
            format_func=lambda i: candidate_names[i],
        )
        state = history[selected_idx]["state"]

        # Comparison summary table
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">All Candidates Comparison</div>', unsafe_allow_html=True)
        import pandas as pd
        rows = []
        for r in history:
            sc = r["state"].get("scorecard") or {}
            dec = r["state"].get("decision") or {}
            rows.append({
                "Candidate": r["filename"],
                "Overall Score": sc.get("overall_score", "N/A"),
                "Decision": dec.get("decision", "N/A"),
                "Confidence": f"{dec.get('confidence', 0)*100:.0f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        state = history[0]["state"]

    # Clear history button
    if st.button("🗑 Clear all results"):
        st.session_state.run_history = []
        st.session_state.result_state = None
        st.rerun()

    profile = state.get("parsed_profile") or {}
    scorecard = state.get("scorecard") or {}
    verification = state.get("verification_result") or {}
    decision = state.get("decision") or {}

    # Candidate header
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"### {profile.get('name', 'Unknown Candidate')}")
        st.write(f"✉️ {profile.get('email') or 'N/A'}   |   📞 {profile.get('phone') or 'N/A'}")
        skills = profile.get("skills") or []
        if skills:
            st.write("**Skills:** " + ", ".join(skills[:15]))
    with c2:
        label = decision.get("decision", "Hold")
        badge_class = DECISION_BADGE.get(label, "badge-hold")
        st.markdown(f'<span class="badge {badge_class}">{label.upper()}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="score-pill">{scorecard.get("overall_score", 0)} / 5</div>', unsafe_allow_html=True)
        st.caption(f"Confidence: {decision.get('confidence', 0)*100:.0f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    # Score breakdown
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Color-Coded Score Breakdown</div>', unsafe_allow_html=True)
    dims = ["skills_score", "experience_score", "education_score", "projects_score", "communication_score"]
    cols = st.columns(len(dims))
    for col, dim in zip(cols, dims):
        val = scorecard.get(dim, 0)
        color = "#2cb67d" if val >= 4 else ("#f59e0b" if val >= 2.5 else "#ef4444")
        with col:
            st.markdown(
                f'<div style="text-align:center"><div style="font-size:1.4rem;font-weight:800;color:{color}">'
                f'{val}</div><div class="metric-label">{dim.replace("_score","").title()}</div></div>',
                unsafe_allow_html=True,
            )
    st.progress(min(1.0, max(0.0, scorecard.get("overall_score", 0) / 5)))
    st.markdown("</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">✅ Strengths</div>', unsafe_allow_html=True)
        for s in scorecard.get("strengths", []) or ["No strengths recorded."]:
            st.write(f"- {s}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚠️ Gaps</div>', unsafe_allow_html=True)
        for g in scorecard.get("gaps", []) or ["No gaps recorded."]:
            st.write(f"- {g}")
        st.markdown("</div>", unsafe_allow_html=True)

    if verification:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🛡️ Verification Summary</div>', unsafe_allow_html=True)
        st.write(f"**Status:** `{verification.get('status', 'skipped')}`")
        st.write(f"**Prompt injection detected:** {verification.get('prompt_injection_detected')}")
        st.write(f"**Bias check passed:** {verification.get('bias_check_passed')}")
        st.write(f"**Reason:** {verification.get('reason')}")
        if state.get("escalated"):
            st.error("🚨 This candidate required human escalation.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧠 Decision Reasoning</div>', unsafe_allow_html=True)
    st.write(decision.get("reasoning", "N/A"))
    recs = decision.get("recommendations") or []
    if recs:
        st.markdown("**Recommendations:**")
        for r in recs:
            st.write(f"- {r}")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: INTERVIEW QUESTIONS
# ---------------------------------------------------------------------------

def page_interview_questions() -> None:
    _txt = "#FFFFFF" if st.session_state.theme == "dark" else "#1a1040"
    _subtxt = "#D0D0E8" if st.session_state.theme == "dark" else "#3a3a6a"
    st.markdown('<div class="section-title">🎯 Dynamic Interview Questions</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{_subtxt};font-size:15px;">Questions are generated dynamically based on the '
        'candidate\'s resume and the job description. Safe for non-technical HRs — '
        'expected answers are provided for every question.</p>',
        unsafe_allow_html=True,
    )

    history = st.session_state.run_history
    has_jd = bool(st.session_state.job_description.strip())

    if not has_jd:
        st.warning("Please enter a Job Description first (Job Description page).")
        return

    # Candidate selector
    candidate_profile_str = ""
    selected_name = "Candidate"
    if history:
        names = [f"{i+1}. {r['filename']}" for i, r in enumerate(history)]
        idx = st.selectbox("Select candidate", range(len(history)), format_func=lambda i: names[i])
        profile = history[idx]["state"].get("parsed_profile") or {}
        selected_name = profile.get("name", history[idx]["filename"])
        skills = ", ".join(profile.get("skills", [])[:15])
        exp = "; ".join(profile.get("experience_summary", [])[:3])
        edu = "; ".join(profile.get("education", [])[:2])
        candidate_profile_str = f"Name: {selected_name}\nSkills: {skills}\nExperience: {exp}\nEducation: {edu}"
    else:
        st.info("No candidates evaluated yet. You can still generate generic questions from the JD.")
        candidate_profile_str = "(No resume uploaded — generate from JD only)"

    # Question type filter
    q_types = st.multiselect(
        "Question categories to include",
        ["Technical", "Behavioural", "Situational", "Culture Fit", "Role-Specific"],
        default=["Technical", "Behavioural", "Role-Specific"],
    )
    num_questions = st.slider("Number of questions", 5, 20, 10)

    if st.button("Generate Interview Questions"):
        with st.spinner("Generating personalised questions..."):
            try:
                from utils.helpers import get_llm
                llm = get_llm(temperature=0.7)
                categories = ", ".join(q_types)
                prompt = f"""You are an expert HR interviewer. Generate {num_questions} interview questions 
for the following candidate and job description.

CANDIDATE PROFILE:
{candidate_profile_str}

JOB DESCRIPTION:
{st.session_state.job_description[:3000]}

QUESTION CATEGORIES: {categories}

For each question provide:
1. The question itself
2. Category (Technical/Behavioural/Situational/Culture Fit/Role-Specific)
3. What a good answer should cover (2-3 bullet points) — written for a non-technical HR

Format as JSON array:
[
  {{
    "question": "...",
    "category": "...",
    "expected_answer_hints": ["hint 1", "hint 2", "hint 3"]
  }}
]

Return ONLY the JSON array, no markdown fences."""

                response = llm.invoke([{"role": "user", "content": prompt}])
                import json, re as _re
                raw = response.content.strip()
                raw = _re.sub(r"^```(json)?", "", raw).strip()
                raw = _re.sub(r"```$", "", raw).strip()
                # Extract array
                arr_match = _re.search(r"\[.*\]", raw, _re.DOTALL)
                if arr_match:
                    questions = json.loads(arr_match.group(0))
                    st.session_state.interview_questions = questions
                    st.success(f"Generated {len(questions)} questions for {selected_name}.")
                else:
                    st.error("Could not parse questions. Try again.")
            except Exception as exc:
                st.error(f"Error generating questions: {exc}")

    # Display questions
    if st.session_state.interview_questions:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">Questions for {selected_name}</div>', unsafe_allow_html=True)

        category_colors = {
            "Technical": "#3b82f6",
            "Behavioural": "#2cb67d",
            "Situational": "#f59e0b",
            "Culture Fit": "#a78bfa",
            "Role-Specific": "#ef4444",
        }

        for i, q in enumerate(st.session_state.interview_questions, 1):
            cat = q.get("category", "General")
            color = category_colors.get(cat, "#7f5af0")
            q_bg = "rgba(255,255,255,0.06)" if st.session_state.theme == "dark" else "rgba(255,255,255,0.55)"
            st.markdown(
                f'<div style="margin-bottom:1rem;padding:1rem;background:{q_bg};'
                f'border-left:4px solid {color};border-radius:8px;">'
                f'<div style="color:{color};font-size:12px;font-weight:700;text-transform:uppercase;'
                f'margin-bottom:4px;">{cat}</div>'
                f'<div style="color:{_txt};font-size:16px;font-weight:600;">Q{i}. {q.get("question","")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            hints = q.get("expected_answer_hints", [])
            if hints:
                with st.expander(f"Expected answer hints for Q{i}"):
                    for h in hints:
                        st.markdown(f"• {h}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Download as text
        if st.button("Download Questions as Text"):
            lines = [f"Interview Questions — {selected_name}\n{'='*50}\n"]
            for i, q in enumerate(st.session_state.interview_questions, 1):
                lines.append(f"Q{i} [{q.get('category','')}]: {q.get('question','')}")
                for h in q.get("expected_answer_hints", []):
                    lines.append(f"   • {h}")
                lines.append("")
            st.download_button(
                "⬇ Download",
                data="\n".join(lines),
                file_name=f"interview_questions_{selected_name.replace(' ','_')}.txt",
                mime="text/plain",
            )


# ---------------------------------------------------------------------------
# PAGE: SCHEDULE INTERVIEW
# ---------------------------------------------------------------------------

def page_schedule_interview() -> None:
    st.markdown('<div class="section-title">📅 Schedule Interview & Send Email</div>', unsafe_allow_html=True)

    history = st.session_state.run_history
    if not history:
        st.info("No candidates evaluated yet. Run the crew first.")
        return

    # Candidate selector
    names = [f"{i+1}. {r['filename']} — {(r['state'].get('decision') or {}).get('decision','N/A')}"
             for i, r in enumerate(history)]
    idx = st.selectbox("Select candidate", range(len(history)), format_func=lambda i: names[i])
    selected = history[idx]
    profile = selected["state"].get("parsed_profile") or {}
    decision = selected["state"].get("decision") or {}
    candidate_name = profile.get("name", selected["filename"])
    candidate_email_prefill = profile.get("email") or ""

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Scheduling for: {candidate_name}</div>', unsafe_allow_html=True)

    label = decision.get("decision", "Hold")
    badge_class = {"Hire": "badge-hire", "Interview": "badge-interview",
                   "Hold": "badge-hold", "Reject": "badge-reject"}.get(label, "badge-hold")
    st.markdown(f'<span class="badge {badge_class}">{label.upper()}</span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        interview_date = st.date_input("Interview Date")
        interview_time = st.time_input("Interview Time")
        interview_mode = st.selectbox("Mode", ["Video Call (Google Meet)", "Video Call (Zoom)",
                                                "Phone Call", "In-Person", "WhatsApp Video"])
    with col2:
        interview_round = st.selectbox("Round", ["HR Screening", "Technical Round 1",
                                                  "Technical Round 2", "Managerial", "Final HR"])
        duration = st.selectbox("Duration", ["30 minutes", "45 minutes", "60 minutes", "90 minutes"])
        interviewer_name = st.text_input("Interviewer Name", placeholder="e.g. Priya Sharma")

    meet_link = st.text_input("Meeting Link (optional)", placeholder="https://meet.google.com/...")
    st.markdown("</div>", unsafe_allow_html=True)

    # Email section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📧 Send Interview Invitation Email</div>', unsafe_allow_html=True)

    to_email = st.text_input("Candidate Email", value=candidate_email_prefill,
                              placeholder="candidate@email.com")
    from_email = st.text_input("Your Email (sender)", placeholder="hr@company.com")
    email_password = st.text_input("Email App Password", type="password",
                                   help="Use Gmail App Password (not your main password). Enable 2FA → App Passwords.")

    # Auto-generate email body
    email_subject = f"Interview Invitation — {interview_round} | {candidate_name}"
    meet_info = f"\nMeeting Link: {meet_link}" if meet_link.strip() else ""
    email_body = f"""Dear {candidate_name},

We are pleased to invite you for an interview at our company.

Interview Details:
- Round: {interview_round}
- Date: {interview_date.strftime('%A, %d %B %Y')}
- Time: {interview_time.strftime('%I:%M %p')}
- Mode: {interview_mode}
- Duration: {duration}{meet_info}
- Interviewer: {interviewer_name or 'To be confirmed'}

Please confirm your availability by replying to this email.

Best regards,
{interviewer_name or 'Recruitment Team'}
HR Department"""

    email_body = st.text_area("Email Body (editable)", value=email_body, height=280)

    if st.button("📤 Send Interview Email"):
        if not to_email.strip():
            st.error("Please enter candidate email.")
        elif not from_email.strip():
            st.error("Please enter your sender email.")
        elif not email_password.strip():
            st.error("Please enter your email app password.")
        else:
            with st.spinner("Sending email..."):
                try:
                    import smtplib
                    from email.mime.text import MIMEText
                    from email.mime.multipart import MIMEMultipart

                    msg = MIMEMultipart()
                    msg["From"] = from_email
                    msg["To"] = to_email
                    msg["Subject"] = email_subject
                    msg.attach(MIMEText(email_body, "plain"))

                    # Auto-detect SMTP from sender domain
                    domain = from_email.split("@")[-1].lower()
                    smtp_configs = {
                        "gmail.com": ("smtp.gmail.com", 587),
                        "outlook.com": ("smtp.office365.com", 587),
                        "hotmail.com": ("smtp.office365.com", 587),
                        "yahoo.com": ("smtp.mail.yahoo.com", 587),
                    }
                    smtp_host, smtp_port = smtp_configs.get(domain, ("smtp.gmail.com", 587))

                    with smtplib.SMTP(smtp_host, smtp_port) as server:
                        server.starttls()
                        server.login(from_email, email_password)
                        server.sendmail(from_email, to_email, msg.as_string())

                    st.success(f"Email sent to {to_email} successfully!")
                    st.session_state.schedule_confirmed = True

                    # Save to history
                    history[idx]["scheduled"] = {
                        "date": str(interview_date),
                        "time": str(interview_time),
                        "round": interview_round,
                        "mode": interview_mode,
                        "email_sent_to": to_email,
                    }

                except smtplib.SMTPAuthenticationError:
                    st.error("Authentication failed. Use a Gmail App Password, not your main password. "
                             "Go to Google Account → Security → 2-Step Verification → App Passwords.")
                except Exception as exc:
                    st.error(f"Failed to send email: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Scheduled interviews summary
    scheduled = [(r["filename"], r.get("scheduled")) for r in history if r.get("scheduled")]
    if scheduled:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Scheduled Interviews</div>', unsafe_allow_html=True)
        import pandas as pd
        rows = [{"Candidate": fn, "Date": s["date"], "Time": s["time"],
                 "Round": s["round"], "Mode": s["mode"], "Email": s["email_sent_to"]}
                for fn, s in scheduled]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: POST JD TO PORTALS
# ---------------------------------------------------------------------------

def page_post_jd() -> None:
    _txt = "#FFFFFF" if st.session_state.theme == "dark" else "#1a1040"
    _subtxt = "#D0D0E8" if st.session_state.theme == "dark" else "#3a3a6a"
    _desc_color = "#C0C0D8" if st.session_state.theme == "dark" else "#4a4a7a"
    _card_bg = "rgba(255,255,255,0.07)" if st.session_state.theme == "dark" else "rgba(255,255,255,0.65)"
    _card_border = "rgba(255,255,255,0.15)" if st.session_state.theme == "dark" else "rgba(127,90,240,0.2)"
    st.markdown('<div class="section-title">🌐 Post JD to Job Portals</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{_subtxt};font-size:15px;">Direct API posting requires paid recruiter accounts '
        'on each portal. Below are direct links to post on each platform, plus a formatted JD '
        'ready to copy-paste.</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.job_description.strip():
        st.warning("Please enter a Job Description first (Job Description page).")
        return

    # Portal links
    portals = [
        ("Naukri.com", "https://www.naukri.com/recruiter/home", "🟠",
         "India's largest job portal — free job postings for recruiters"),
        ("LinkedIn Jobs", "https://www.linkedin.com/talent/post-a-job", "🔵",
         "Professional network — best for experienced roles"),
        ("Internshala", "https://internshala.com/post-internship-or-job", "🟢",
         "Best for internships & freshers"),
        ("Unstop", "https://unstop.com/post-opportunity", "🟣",
         "Competitions, hackathons & campus hiring"),
        ("Indeed India", "https://employers.indeed.com/", "🔴",
         "High volume applications — good for mass hiring"),
        ("Glassdoor", "https://www.glassdoor.com/employers/", "🟩",
         "Company reviews + job postings combined"),
        ("Wellfound (AngelList)", "https://wellfound.com/recruiting", "⚫",
         "Startup ecosystem — equity & remote roles"),
        ("Shine.com", "https://www.shine.com/recruiter/", "🟡",
         "Mid-level professionals across India"),
        ("Hirist.tech", "https://www.hirist.tech/", "🔷",
         "Tech-only job portal"),
        ("Freshersworld", "https://www.freshersworld.com/employer", "🟤",
         "Entry-level and fresher candidates"),
        ("IIMJobs", "https://www.iimjobs.com/for-employers", "🔶",
         "MBA and management roles"),
        ("TimesJobs", "https://employer.timesjobs.com/", "🔴",
         "General hiring across sectors"),
    ]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick Post Links</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (name, url, icon, desc) in enumerate(portals):
        with cols[i % 3]:
            st.markdown(
                f'<div style="background:{_card_bg};border:1px solid {_card_border};'
                f'border-radius:12px;padding:12px;margin-bottom:10px;">'
                f'<div style="font-size:20px">{icon}</div>'
                f'<div style="color:{_txt};font-weight:700;font-size:15px;">{name}</div>'
                f'<div style="color:{_desc_color};font-size:12px;margin:4px 0 8px 0;">{desc}</div>'
                f'<a href="{url}" target="_blank" style="background:linear-gradient(90deg,#7f5af0,#2cb67d);'
                f'color:white;padding:6px 14px;border-radius:8px;text-decoration:none;font-size:13px;font-weight:700;">'
                f'Post Now →</a></div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # Formatted JD for copy-paste
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Formatted JD — Copy & Paste to Any Portal</div>', unsafe_allow_html=True)
    st.text_area(
        "Ready-to-post Job Description",
        value=st.session_state.job_description,
        height=300,
    )
    st.download_button(
        "⬇ Download JD as .txt",
        data=st.session_state.job_description,
        file_name="job_description.txt",
        mime="text/plain",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Email blast feature
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📧 Email JD to College Placement Cells</div>', unsafe_allow_html=True)
    st.caption("Send your JD directly to college TPOs for campus hiring")
    college_emails = st.text_area(
        "Placement cell emails (one per line)",
        placeholder="tpo@iit.ac.in\nplacements@nit.ac.in\nhrd@bits.ac.in",
        height=100,
    )
    from_email_jd = st.text_input("Your email", placeholder="hr@company.com", key="jd_from")
    from_pass_jd = st.text_input("App Password", type="password", key="jd_pass")
    company_name = st.text_input("Company Name", placeholder="TechVest Solutions")

    if st.button("Send JD to All Colleges"):
        emails = [e.strip() for e in college_emails.strip().splitlines() if e.strip()]
        if not emails:
            st.error("Enter at least one email.")
        elif not from_email_jd or not from_pass_jd:
            st.error("Enter your email and app password.")
        else:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            sent, failed = 0, 0
            progress = st.progress(0)
            for i, to_email in enumerate(emails):
                try:
                    msg = MIMEMultipart()
                    msg["From"] = from_email_jd
                    msg["To"] = to_email
                    msg["Subject"] = f"Job Opportunity at {company_name or 'Our Company'} — Campus Hiring"
                    body = f"Dear Placement Officer,\n\nWe are hiring! Please find the job description below.\n\n{st.session_state.job_description}\n\nKindly circulate among eligible students.\n\nRegards,\nHR Team\n{company_name or ''}"
                    msg.attach(MIMEText(body, "plain"))
                    domain = from_email_jd.split("@")[-1].lower()
                    smtp_map = {"gmail.com": ("smtp.gmail.com", 587), "outlook.com": ("smtp.office365.com", 587)}
                    host, port = smtp_map.get(domain, ("smtp.gmail.com", 587))
                    with smtplib.SMTP(host, port) as s:
                        s.starttls()
                        s.login(from_email_jd, from_pass_jd)
                        s.sendmail(from_email_jd, to_email, msg.as_string())
                    sent += 1
                except Exception:
                    failed += 1
                progress.progress((i + 1) / len(emails))
            st.success(f"Sent to {sent} college(s)." + (f" {failed} failed." if failed else ""))
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: VOICE INTERVIEW
# ---------------------------------------------------------------------------

def page_voice_interview() -> None:
    import os
    _txt = "#FFFFFF" if st.session_state.theme == "dark" else "#1a1040"
    _subtxt = "#D0D0E8" if st.session_state.theme == "dark" else "#3a3a6a"

    st.markdown('<div class="section-title">🎙️ Voice Interview & Answer Evaluation</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{_subtxt};font-size:15px;">'
        'Record the candidate\'s spoken answer live in the browser. '
        'Audio is sent to the local Whisper model for instant transcription — no API key required.'
        '</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.interview_questions:
        st.warning("Generate interview questions first on the **Interview Questions** page.")
        return

    # Question selector
    q_labels = [f"Q{i+1}: {q['question'][:80]}..." for i, q in enumerate(st.session_state.interview_questions)]
    selected_q_idx = st.selectbox("Select question to evaluate", range(len(q_labels)),
                                   format_func=lambda i: q_labels[i])
    selected_q = st.session_state.interview_questions[selected_q_idx]

    # Question card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#a78bfa;font-weight:700;font-size:13px;text-transform:uppercase;'
        f'letter-spacing:.06em;">Question {selected_q_idx+1}</div>'
        f'<div style="color:{_txt};font-size:18px;font-weight:700;margin:8px 0 12px 0;">'
        f'{selected_q["question"]}</div>',
        unsafe_allow_html=True,
    )
    hints = selected_q.get("expected_answer_hints", [])
    if hints:
        st.markdown('<div style="color:#94a3b8;font-size:12px;font-weight:600;'
                    'text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px;">'
                    'Expected answer hints (HR reference)</div>', unsafe_allow_html=True)
        for h in hints:
            st.markdown(f'<div style="color:#cbd5e1;font-size:14px;margin-left:10px;'
                        f'margin-bottom:3px;">• {h}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Browser MediaRecorder component ────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎤 Live Recording</div>', unsafe_allow_html=True)

    recorder_html = """
<style>
  body { margin:0; background:transparent; font-family:'Segoe UI',sans-serif; }
  #recorder-wrap {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 16px;
    padding: 20px 24px;
    color: #fff;
  }
  #status-text {
    font-size: 14px; color: #94a3b8; margin-bottom: 14px; min-height: 20px;
  }
  /* Waveform canvas */
  #waveform {
    width: 100%; height: 64px;
    border-radius: 10px;
    background: rgba(0,0,0,0.25);
    display: block;
    margin-bottom: 16px;
  }
  /* Timer */
  #timer {
    font-size: 28px; font-weight: 800; color: #a78bfa;
    text-align: center; margin-bottom: 14px; letter-spacing: .04em;
    display: none;
  }
  /* Buttons */
  .rec-btn {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 10px 24px; border-radius: 10px; border: none;
    font-size: 15px; font-weight: 700; cursor: pointer;
    transition: transform .15s, filter .15s; margin-right: 10px;
  }
  .rec-btn:hover { transform: translateY(-2px); filter: brightness(1.12); }
  #btn-start { background: linear-gradient(90deg,#7f5af0,#2cb67d); color: #fff; }
  #btn-stop  { background: linear-gradient(90deg,#ef4444,#f97316); color: #fff; display:none; }
  #btn-send  { background: linear-gradient(90deg,#2cb67d,#3b82f6); color: #fff; display:none; }
  /* Recording dot */
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  #rec-dot { display:inline-block; width:10px; height:10px; border-radius:50%;
             background:#ef4444; animation: pulse 1s infinite; }
  /* Audio preview */
  #audio-preview { width:100%; margin-top:12px; display:none; border-radius:8px; }
</style>

<div id="recorder-wrap">
  <div id="status-text">Click <b>Start Recording</b> to begin.</div>
  <canvas id="waveform"></canvas>
  <div id="timer">00:00</div>
  <button class="rec-btn" id="btn-start">🎙️ Start Recording</button>
  <button class="rec-btn" id="btn-stop"><span id="rec-dot"></span>&nbsp;Stop</button>
  <button class="rec-btn" id="btn-send">📤 Send for Transcription</button>
  <audio id="audio-preview" controls></audio>
</div>

<script>
(function() {
  let mediaRecorder, audioChunks = [], stream, animFrame;
  let timerInterval, seconds = 0;
  let audioBlob = null;

  const canvas   = document.getElementById('waveform');
  const ctx2d    = canvas.getContext('2d');
  const status   = document.getElementById('status-text');
  const timer    = document.getElementById('timer');
  const btnStart = document.getElementById('btn-start');
  const btnStop  = document.getElementById('btn-stop');
  const btnSend  = document.getElementById('btn-send');
  const preview  = document.getElementById('audio-preview');

  function drawIdle() {
    canvas.width = canvas.offsetWidth;
    ctx2d.clearRect(0,0,canvas.width,canvas.height);
    ctx2d.strokeStyle = 'rgba(167,139,250,0.3)';
    ctx2d.lineWidth = 2;
    ctx2d.beginPath();
    ctx2d.moveTo(0, canvas.height/2);
    ctx2d.lineTo(canvas.width, canvas.height/2);
    ctx2d.stroke();
  }

  function drawWave(analyser, bufLen) {
    animFrame = requestAnimationFrame(() => drawWave(analyser, bufLen));
    canvas.width = canvas.offsetWidth;
    const data = new Uint8Array(bufLen);
    analyser.getByteTimeDomainData(data);
    ctx2d.clearRect(0,0,canvas.width,canvas.height);
    const grad = ctx2d.createLinearGradient(0,0,canvas.width,0);
    grad.addColorStop(0,'#7f5af0');
    grad.addColorStop(0.5,'#2cb67d');
    grad.addColorStop(1,'#fbbf24');
    ctx2d.strokeStyle = grad;
    ctx2d.lineWidth = 2.5;
    ctx2d.beginPath();
    const sliceW = canvas.width / bufLen;
    let x = 0;
    for (let i=0; i<bufLen; i++) {
      const v = data[i]/128.0;
      const y = v * canvas.height/2;
      i===0 ? ctx2d.moveTo(x,y) : ctx2d.lineTo(x,y);
      x += sliceW;
    }
    ctx2d.lineTo(canvas.width, canvas.height/2);
    ctx2d.stroke();
  }

  function formatTime(s) {
    const m = Math.floor(s/60).toString().padStart(2,'0');
    const sec = (s%60).toString().padStart(2,'0');
    return m+':'+sec;
  }

  drawIdle();

  btnStart.onclick = async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio:true, video:false});
    } catch(e) {
      status.innerHTML = '❌ Mic access denied. Allow microphone in your browser and refresh.';
      return;
    }
    audioChunks = [];
    audioBlob = null;
    preview.style.display = 'none';
    btnSend.style.display = 'none';

    // Waveform
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);
    drawWave(analyser, analyser.frequencyBinCount);

    // Timer
    seconds = 0;
    timer.style.display = 'block';
    timer.textContent = '00:00';
    timerInterval = setInterval(() => {
      seconds++;
      timer.textContent = formatTime(seconds);
    }, 1000);

    // MediaRecorder — prefer webm/opus, fallback to any supported
    const mimeType = ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/ogg','']
      .find(m => m === '' || MediaRecorder.isTypeSupported(m));
    const opts = mimeType ? {mimeType} : {};
    mediaRecorder = new MediaRecorder(stream, opts);
    mediaRecorder.ondataavailable = e => { if(e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      cancelAnimationFrame(animFrame);
      clearInterval(timerInterval);
      drawIdle();
      stream.getTracks().forEach(t => t.stop());

      audioBlob = new Blob(audioChunks, {type: mediaRecorder.mimeType || 'audio/webm'});
      const url = URL.createObjectURL(audioBlob);
      preview.src = url;
      preview.style.display = 'block';
      btnSend.style.display = 'inline-flex';
      status.innerHTML = `✅ Recorded <b>${formatTime(seconds)}</b>. Preview below, then click <b>Send for Transcription</b>.`;
      timer.style.display = 'none';
    };
    mediaRecorder.start(250); // collect chunks every 250ms

    btnStart.style.display = 'none';
    btnStop.style.display  = 'inline-flex';
    status.innerHTML = '🔴 Recording... click <b>Stop</b> when done.';
  };

  btnStop.onclick = () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    btnStop.style.display  = 'none';
    btnStart.style.display = 'inline-flex';
  };

  btnSend.onclick = () => {
    if (!audioBlob) return;
    status.innerHTML = '⏳ Sending to Whisper for transcription...';
    btnSend.disabled = true;

    const ext = (audioBlob.type.includes('ogg')) ? 'ogg' : 'webm';
    const filename = 'recording.' + ext;
    const formData = new FormData();
    formData.append('audio', audioBlob, filename);

    fetch('{BACKEND_URL}/transcribe', {
      method: 'POST',
      body: formData,
    })
    .then(r => r.json())
    .then(data => {
      const transcript = data.transcript || '';
      const lang = data.language || '';
      const dur = data.duration_s || '';

      // Write transcript into the Streamlit textarea by finding it via DOM
      // Streamlit renders textareas inside iframes — we need to walk up to parent
      const doc = window.parent.document;
      // Find any visible textarea that has the placeholder text
      const textareas = doc.querySelectorAll('textarea');
      let targetTA = null;
      textareas.forEach(ta => {
        if (ta.placeholder && ta.placeholder.includes('transcript')) {
          targetTA = ta;
        }
      });
      if (targetTA) {
        // React-controlled input: need to trigger native input event
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
        nativeInputValueSetter.call(targetTA, transcript);
        targetTA.dispatchEvent(new Event('input', { bubbles: true }));
      }
      status.innerHTML = '✅ Transcribed' + (dur ? ' in ' + dur + 's' : '') + (lang ? ' | Language: ' + lang : '') + '. See Transcript below.';
      btnSend.disabled = false;
    })
    .catch(err => {
      status.innerHTML = '❌ Transcription failed: ' + err.message + '. Is the backend running?';
      btnSend.disabled = false;
    });
  };
})();
</script>
"""

    import streamlit.components.v1 as components
    # Inject the backend URL into the JS so it can fetch /transcribe directly
    recorder_html_rendered = recorder_html.replace("{BACKEND_URL}", BACKEND_URL)
    components.html(recorder_html_rendered, height=320, scrolling=False)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Transcript (editable fallback) ────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✍️ Transcript</div>', unsafe_allow_html=True)
    st.caption("Transcribed text appears here automatically. You can also type/paste manually.")
    st.session_state.voice_transcript = st.text_area(
        "Candidate's answer",
        value=st.session_state.voice_transcript,
        height=130,
        placeholder="Record audio above — the transcript will appear here automatically...",
        label_visibility="collapsed",
    )
    col_eval, col_clear = st.columns([3, 1])
    with col_clear:
        if st.button("🗑 Clear", key="clear_transcript"):
            st.session_state.voice_transcript = ""
            st.session_state.voice_evaluation = ""
            st.session_state["_last_audio_b64"] = ""
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── AI evaluation ─────────────────────────────────────────────────────
    if st.session_state.voice_transcript.strip():
        with col_eval:
            run_eval = st.button("🧠 Evaluate Answer with AI", use_container_width=True)
        if run_eval:
            with st.spinner("Evaluating answer..."):
                try:
                    from utils.helpers import get_llm
                    llm = get_llm(temperature=0.1)
                    hints_str = "\n".join(f"- {h}" for h in selected_q.get("expected_answer_hints", []))
                    eval_prompt = f"""You are an expert interviewer evaluating a candidate's spoken answer.

QUESTION: {selected_q['question']}

EXPECTED ANSWER SHOULD COVER:
{hints_str}

CANDIDATE'S ACTUAL ANSWER:
{st.session_state.voice_transcript}

Provide a concise evaluation for a non-technical HR:
1. Score: X/10
2. ✅ What was covered well
3. ⚠️ What was missing or weak
4. Verdict: Excellent / Good / Average / Poor
5. Suggested follow-up question"""
                    response = llm.invoke([{"role": "user", "content": eval_prompt}])
                    st.session_state.voice_evaluation = response.content
                except Exception as exc:
                    st.error(f"Evaluation failed: {exc}")

    if st.session_state.voice_evaluation:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 AI Evaluation</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{_txt};font-size:15px;line-height:1.8;">'
            f'{st.session_state.voice_evaluation.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("➡️ Next Question"):
            st.session_state.voice_transcript = ""
            st.session_state.voice_evaluation = ""
            st.session_state["_last_audio_b64"] = ""
            # Advance to next question if possible
            next_idx = selected_q_idx + 1
            if next_idx < len(st.session_state.interview_questions):
                st.session_state["_voice_q_idx"] = next_idx
            st.rerun()


# ---------------------------------------------------------------------------
# ROUTER
# ---------------------------------------------------------------------------

PAGES = {
    "Home": page_home,
    "Upload Resume": page_upload_resume,
    "Job Description": page_job_description,
    "Post JD to Portals": page_post_jd,
    "Run Crew": page_run_crew,
    "Execution Logs": page_execution_logs,
    "Final Report": page_final_report,
    "Interview Questions": page_interview_questions,
    "Voice Interview": page_voice_interview,
    "Schedule Interview": page_schedule_interview,
}

PAGES[st.session_state.page]()
