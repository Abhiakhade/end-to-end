import streamlit as st
import torch
import pickle
import requests
from src.models.matrix_factorization import MatrixFactorization

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
OMDB_API_KEY = "73b72385"   # free at omdbapi.com/apikey.aspx
OMDB_URL     = "http://www.omdbapi.com/"

st.set_page_config(
    page_title="RecoEngine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]          { background:#0a0d14; }
[data-testid="stSidebar"]                   { background:#10141f; border-right:1px solid #1e2535; }
[data-testid="stSidebar"] *                 { color:#c9d1e0 !important; }
.block-container                            { padding:1.75rem 2rem; max-width:1200px; }

[data-testid="metric-container"]            { background:#10141f; border:1px solid #1e2535; border-radius:12px; padding:.9rem 1.1rem; }
[data-testid="metric-container"] label      { color:#4b5675 !important; font-size:11px; letter-spacing:.06em; text-transform:uppercase; }
[data-testid="stMetricValue"]               { font-size:26px !important; font-weight:500; color:#e2e8f5 !important; }
[data-testid="stMetricDelta"] svg           { display:none; }

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input         { background:#10141f !important; border:1px solid #1e2535 !important;
                                              border-radius:8px !important; color:#e2e8f5 !important; font-size:14px !important; }
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus   { border-color:#3b82f6 !important; box-shadow:0 0 0 3px rgba(59,130,246,.15) !important; }
[data-testid="stSelectbox"] > div > div     { background:#10141f !important; border:1px solid #1e2535 !important;
                                              border-radius:8px !important; color:#e2e8f5 !important; }
[data-testid="stMultiSelect"] > div         { background:#10141f !important; border:1px solid #1e2535 !important; border-radius:8px !important; }
[data-testid="stMultiSelect"] span          { color:#e2e8f5 !important; }

[data-testid="stButton"] > button           { background:#2563eb !important; color:#fff !important; border:none !important;
                                              border-radius:8px !important; padding:.45rem 1.4rem !important;
                                              font-weight:500 !important; font-size:14px !important; }
[data-testid="stButton"] > button:hover     { background:#1d4ed8 !important; }

.movie-card {
    background:#10141f; border:1px solid #1e2535; border-radius:12px;
    overflow:hidden; transition:border-color .15s; margin-bottom:4px;
}
.movie-card:hover { border-color:#3b82f6; }
.poster-wrap { position:relative; }
.poster-wrap img { width:100%; display:block; }
.poster-placeholder {
    height:180px; background:#161b27; display:flex; align-items:center;
    justify-content:center; font-size:44px;
}
.rank-badge {
    position:absolute; top:6px; left:6px; background:rgba(10,13,20,.85);
    border:1px solid #2563eb; color:#60a5fa; border-radius:50%;
    width:24px; height:24px; display:flex; align-items:center;
    justify-content:center; font-size:11px; font-weight:600;
}
.score-badge {
    position:absolute; bottom:6px; right:6px; background:rgba(10,13,20,.85);
    border:1px solid #1e2535; color:#e2e8f5; border-radius:6px;
    padding:2px 7px; font-size:11px; font-weight:500;
}
.card-body   { padding:10px 12px 12px; }
.card-title  { font-size:13px; font-weight:500; color:#e2e8f5; margin-bottom:3px;
                white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.card-meta   { font-size:11px; color:#4b5675; margin-bottom:6px; }
.score-bar-bg   { height:4px; background:#1e2535; border-radius:99px; }
.score-bar-fill { height:100%; border-radius:99px; }
.imdb-rating { font-size:11px; color:#facc15; margin-top:4px; }

hr { border-color:#1e2535 !important; margin:1.25rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    with open("artifacts/encoders/user_encoder.pkl", "rb") as f:
        user_enc = pickle.load(f)
    with open("artifacts/encoders/item_encoder.pkl", "rb") as f:
        item_enc = pickle.load(f)
    n_users = len(user_enc.classes_)
    n_items = len(item_enc.classes_)
    model = MatrixFactorization(n_users, n_items)
    model.load_state_dict(torch.load("artifacts/models/model.pt", map_location="cpu"))
    model.eval()
    return model, user_enc, item_enc


# ──────────────────────────────────────────────
# OMDb helpers
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=86400)
def omdb_search(title: str) -> dict:
    """Fetch metadata + poster for a single movie title via OMDb."""
    try:
        r = requests.get(
            OMDB_URL,
            params={"apikey": OMDB_API_KEY, "t": title, "type": "movie"},
            timeout=5,
        )
        data = r.json()
        return data if data.get("Response") == "True" else {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=86400)
def omdb_search_query(query: str) -> list:
    """Search OMDb by keyword — returns list of result dicts."""
    try:
        r = requests.get(
            OMDB_URL,
            params={"apikey": OMDB_API_KEY, "s": query, "type": "movie"},
            timeout=5,
        )
        data = r.json()
        if data.get("Response") == "True":
            # Enrich each search hit with full metadata
            results = []
            for item in data.get("Search", [])[:12]:
                full = omdb_by_id(item["imdbID"])
                results.append(full if full else item)
            return results
        return []
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=86400)
def omdb_by_id(imdb_id: str) -> dict:
    try:
        r = requests.get(
            OMDB_URL,
            params={"apikey": OMDB_API_KEY, "i": imdb_id, "type": "movie"},
            timeout=5,
        )
        data = r.json()
        return data if data.get("Response") == "True" else {}
    except Exception:
        return {}


# Curated "trending" fallback list — replace with any titles you like
CURATED_TRENDING = [
    "Oppenheimer", "Dune Part Two", "Poor Things", "Killers of the Flower Moon",
    "The Zone of Interest", "Past Lives", "All of Us Strangers", "Saltburn",
    "Anatomy of a Fall", "May December",
]

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_trending_omdb() -> list:
    results = []
    for title in CURATED_TRENDING:
        data = omdb_search(title)
        if data:
            results.append(data)
    return results


# ──────────────────────────────────────────────
# Recommend
# ──────────────────────────────────────────────
def recommend(user_id, model, user_enc, item_enc, top_k=5):
    if user_id not in user_enc.classes_:
        return None, None
    user_idx = user_enc.transform([user_id])[0]
    items = torch.arange(len(item_enc.classes_))
    users = torch.tensor([user_idx] * len(items))
    with torch.no_grad():
        scores = model(users, items)
    top_scores, top_idx = torch.topk(scores, top_k)
    top_items = item_enc.inverse_transform(top_idx.numpy())
    s_min = top_scores.min().item()
    s_max = top_scores.max().item()
    norm  = [int((s - s_min) / (s_max - s_min + 1e-9) * 100) for s in top_scores.tolist()]
    return list(top_items), norm


# ──────────────────────────────────────────────
# Card renderer
# ──────────────────────────────────────────────
def render_movie_card(
    meta: dict,
    rank: int  = None,
    score: int = None,
    bar_color: str = "#2563eb",
    placeholder: str = "🎬",
):
    title   = meta.get("Title", meta.get("title", "Unknown"))
    year    = meta.get("Year",        "")
    genre   = meta.get("Genre",       "")
    rating  = meta.get("imdbRating",  "")
    poster  = meta.get("Poster",      "")

    has_poster = bool(poster and poster != "N/A")

    # ── poster image or placeholder ──────────────────
    if has_poster:
        img_html = (
            f'<img src="{poster}" alt="{title}" '
            f'style="width:100%;height:200px;object-fit:cover;display:block;" />'
        )
    else:
        img_html = (
            f'<div style="height:200px;background:#161b27;display:flex;'
            f'align-items:center;justify-content:center;font-size:48px;">'
            f'{placeholder}</div>'
        )

    # ── overlays ──────────────────────────────────────
    rank_html = (
        f'<div style="position:absolute;top:8px;left:8px;width:26px;height:26px;'
        f'border-radius:50%;background:rgba(10,13,20,.88);border:1px solid #2563eb;'
        f'color:#60a5fa;display:flex;align-items:center;justify-content:center;'
        f'font-size:11px;font-weight:600;">{rank}</div>'
    ) if rank else ""

    score_html = (
        f'<div style="position:absolute;bottom:8px;right:8px;background:rgba(10,13,20,.88);'
        f'border:1px solid #1e2535;color:#e2e8f5;border-radius:6px;'
        f'padding:2px 8px;font-size:11px;font-weight:500;">{score}%</div>'
    ) if score is not None else ""

    # ── score bar ─────────────────────────────────────
    bar_html = (
        f'<div style="height:4px;background:#1e2535;border-radius:99px;overflow:hidden;margin-bottom:6px;">'
        f'<div style="width:{score}%;height:100%;background:{bar_color};border-radius:99px;"></div>'
        f'</div>'
    ) if score is not None else ""

    # ── IMDb rating (stars inline — no block element) ─
    rating_html = (
        f'<span style="font-size:11px;color:#facc15;display:inline-block;margin-top:4px;">'
        f'&#9733; {rating} IMDb</span>'
    ) if rating and rating != "N/A" else ""

    # ── genre / meta line ────────────────────────────
    genre_str  = genre[:28] if genre and genre != "N/A" else ""
    meta_parts = " · ".join(filter(None, [year, genre_str]))
    meta_line  = (
        f'<div style="font-size:11px;color:#4b5675;margin-bottom:5px;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
        f'{meta_parts}</div>'
    ) if meta_parts else ""

    # ── watchlist button state ────────────────────────
    wl     = st.session_state.get("watchlist", [])
    in_wl  = title in wl
    btn_bg = "#162032" if in_wl else "#10141f"
    btn_cl = "#4ade80" if in_wl else "#60a5fa"
    btn_bd = "#166534" if in_wl else "#1e2535"
    btn_tx = "✓ Saved" if in_wl else "+ Watchlist"

    # ── full card HTML ────────────────────────────────
    st.markdown(f"""
    <div style="background:#10141f;border:1px solid #1e2535;border-radius:12px;
                overflow:hidden;margin-bottom:4px;transition:border-color .15s;">
      <div style="position:relative;">
        {img_html}
        {rank_html}
        {score_html}
      </div>
      <div style="padding:10px 12px 4px;">
        <div style="font-size:13px;font-weight:500;color:#e2e8f5;margin-bottom:3px;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{title}</div>
        {meta_line}
        {bar_html}
        {rating_html}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── watchlist toggle (native Streamlit button) ────
    if st.button(btn_tx, key=f"wl_{title}_{rank}_{score}", use_container_width=True):
        if in_wl:
            st.session_state["watchlist"].remove(title)
        else:
            st.session_state.setdefault("watchlist", []).append(title)
        st.rerun()

    # ── per-button color override via scoped CSS ──────
    st.markdown(f"""
    <style>
    div[data-testid="stButton"]:has(button[kind="secondary"][data-testid*="wl_{title[:10]}"]) button {{
        background: {btn_bg} !important;
        color: {btn_cl} !important;
        border: 1px solid {btn_bd} !important;
        border-radius: 0 0 10px 10px !important;
        margin-top: -4px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
st.session_state.setdefault("watchlist", [])
st.session_state.setdefault("history",   [])

# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
with st.spinner("Loading model…"):
    model, user_enc, item_enc = load_model()

n_users = len(user_enc.classes_)
n_items = len(item_enc.classes_)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 RecoEngine")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Search", "Trending", "Watchlist", "User Lookup", "Model Info"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Quick filters**")
    genre_filter = st.multiselect(
        "Genres", ["Action","Drama","Sci-Fi","Comedy","Thriller","Horror","Romance"],
        label_visibility="collapsed",
    )
    year_range = st.slider("Release year", 1980, 2024, (2000, 2024))
    st.markdown("---")
    st.markdown(
        '<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#4b5675">'
        '<div style="width:7px;height:7px;border-radius:50%;background:#4ade80"></div>'
        'Model active · MF v2.1</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("### Movie Recommendation System")
    st.markdown(
        '<p style="color:#4b5675;font-size:13px;margin-bottom:1.25rem">'
        'Matrix factorization · personalized top-K predictions via OMDb</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total users",  f"{n_users:,}")
    c2.metric("Catalog size", f"{n_items:,}")
    c3.metric("Model",        "MF")
    c4.metric("Watchlist",    len(st.session_state["watchlist"]))
    st.markdown("---")

    col_uid, col_k, col_btn = st.columns([3, 2, 1.5])
    user_id = col_uid.number_input("User ID", min_value=1, step=1)
    top_k   = col_k.selectbox("Top K", [5, 10, 15, 20])
    col_btn.write("")
    run = col_btn.button("Recommend →", use_container_width=True)

    if run:
        st.session_state["history"].append(int(user_id))
        items, scores = recommend(user_id, model, user_enc, item_enc, top_k)

        if items is None:
            st.error(f"User ID **{user_id}** not found in training set.", icon="⚠️")
        else:
            st.markdown(f"**Results for user `#{user_id}`** — {top_k} recommendations")
            cols = st.columns(min(5, top_k))
            for i, (item, score) in enumerate(zip(items, scores)):
                with cols[i % min(5, top_k)]:
                    meta = omdb_search(str(item))
                    render_movie_card(meta, rank=i + 1, score=score)

    st.markdown("---")
    st.markdown("### Trending picks")
    trending = fetch_trending_omdb()
    if trending:
        tcols = st.columns(5)
        for i, movie in enumerate(trending[:5]):
            with tcols[i]:
                render_movie_card(movie, bar_color="#7c3aed")
    else:
        st.info("Add your OMDb API key to load movie data.")


# ══════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════
elif page == "Search":
    st.markdown("### Search movies")
    query = st.text_input("", placeholder="Type a movie title…", label_visibility="collapsed")

    if query and len(query) >= 2:
        with st.spinner("Searching…"):
            results = omdb_search_query(query)

        if not results:
            st.warning("No results found.")
        else:
            st.markdown(f"**{len(results)} results** for *{query}*")
            cols = st.columns(4)
            for i, movie in enumerate(results):
                with cols[i % 4]:
                    render_movie_card(movie, bar_color="#7c3aed")
    else:
        st.markdown(
            '<p style="color:#4b5675;font-size:13px">Start typing to search the OMDb catalog…</p>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# TRENDING
# ══════════════════════════════════════════════
elif page == "Trending":
    st.markdown("### Trending picks")
    st.caption("Curated selection — update `CURATED_TRENDING` list in code to customize.")
    trending = fetch_trending_omdb()
    if not trending:
        st.info("Add your OMDb API key to load movie data.")
    else:
        cols = st.columns(5)
        for i, movie in enumerate(trending):
            with cols[i % 5]:
                render_movie_card(movie, rank=i + 1, bar_color="#7c3aed")


# ══════════════════════════════════════════════
# WATCHLIST
# ══════════════════════════════════════════════
elif page == "Watchlist":
    st.markdown("### My watchlist")
    wl = st.session_state["watchlist"]
    if not wl:
        st.info("Nothing saved yet. Hit **+ Watchlist** on any movie card.")
    else:
        st.markdown(f"{len(wl)} saved movies")
        cols = st.columns(4)
        for i, title in enumerate(wl):
            with cols[i % 4]:
                meta = omdb_search(title)
                render_movie_card(meta, bar_color="#059669")
        st.markdown("---")
        if st.button("Clear watchlist"):
            st.session_state["watchlist"] = []
            st.rerun()


# ══════════════════════════════════════════════
# USER LOOKUP
# ══════════════════════════════════════════════
elif page == "User Lookup":
    st.markdown("### User lookup")
    uid = st.number_input("Enter user ID", min_value=1, step=1)
    if st.button("Check"):
        if uid in user_enc.classes_:
            idx = user_enc.transform([uid])[0]
            st.success(f"User **{uid}** found — internal index `{idx}`")
        else:
            st.warning(f"User **{uid}** not in training set.")

    if st.session_state["history"]:
        st.markdown("**Recent lookups**")
        st.write(", ".join(f"`{u}`" for u in st.session_state["history"][-10:]))


# ══════════════════════════════════════════════
# MODEL INFO
# ══════════════════════════════════════════════
elif page == "Model Info":
    st.markdown("### Model information")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Architecture**")
        st.code(f"MatrixFactorization\n  users : {n_users:,}\n  items : {n_items:,}", language="yaml")
    with c2:
        st.markdown("**Artifacts**")
        st.code("model.pt\nuser_encoder.pkl\nitem_encoder.pkl", language="text")
    total = sum(p.numel() for p in model.parameters())
    st.metric("Total parameters", f"{total:,}")
    st.markdown("**Lookup history**")
    if st.session_state["history"]:
        st.write([f"#{u}" for u in st.session_state["history"]])
    else:
        st.caption("No queries yet.")