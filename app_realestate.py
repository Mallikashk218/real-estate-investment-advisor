"""
PropSight - Real Estate Investment Advisor (Streamlit, data-analyst version, no machine learning)

Install once:  pip install --upgrade streamlit pandas numpy matplotlib seaborn pyarrow
Run:           streamlit run app_realestate.py
Needs:         cleaned_housing_data.csv  (from the cleaning script) in the same folder
Tip:           put .streamlit/config.toml (provided) next to this file for the full dark theme.
"""

import base64
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

BRAND = "PropSight"          # change the app name here

st.set_page_config(page_title=f"{BRAND} - Real Estate Investment Advisor", page_icon="🏙️",
                   layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------
# COLOURS
# ------------------------------------------------------------------
BG, PANEL, TEXT, MUTED = "#070b0a", "#0d1412", "#e8e6e1", "#8a948f"
GREEN, BRIGHT, AMBER = "#12a370", "#2ee6a6", "#e0a458"

# ------------------------------------------------------------------
# CSS (the look of the app)
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#070b0a;--panel:#0d1412;--line:#1a2622;--text:#e8e6e1;--muted:#8a948f;--accent:#12a370;--accent2:#2ee6a6;--warn:#e0a458;}
html, body, .stApp, [data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text);font-family:'Inter',sans-serif;}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer,
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"]{display:none!important;}
.block-container{max-width:1280px;padding-top:1.4rem;padding-bottom:3rem;}
h1,h2,h3,h4{font-family:'Syne',sans-serif;color:var(--text);letter-spacing:-0.01em;}
p, li, label, span, div{font-family:'Inter',sans-serif;}

/* ---------- top navigation ---------- */
.nav{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;padding:8px 0 20px;border-bottom:1px solid var(--line);margin-bottom:38px;}
.brand{display:flex;align-items:center;gap:12px;font-family:'Syne',sans-serif;font-weight:800;font-size:1.75rem;color:var(--text);}
.links{display:flex;gap:34px;flex-wrap:wrap;}
.links a{font-size:.7rem;letter-spacing:.28em;text-transform:uppercase;color:var(--muted)!important;text-decoration:none!important;font-weight:700;padding:6px 0;border-bottom:2px solid transparent;}
.links a:hover{color:var(--text)!important;}
.links a.active{color:var(--accent2)!important;border-bottom-color:var(--accent);}

/* ---------- home hero ---------- */
.eyebrow{font-size:.7rem;letter-spacing:.3em;text-transform:uppercase;color:var(--muted);font-weight:700;}
.hero{display:grid;grid-template-columns:1.1fr 1fr;gap:52px;align-items:center;}
.hero h1{font-size:clamp(2.8rem,5.6vw,5.2rem);line-height:1;font-weight:800;margin:18px 0 22px;padding:0;color:#e4e2dc;}
.hl{background:linear-gradient(transparent 58%,#0b7a58 58%);padding:0 6px;}
.rule{width:80px;height:1px;background:#33423d;margin:22px 0;}
.lead{font-size:1.25rem;font-weight:700;line-height:1.5;color:var(--text);}
.sub{color:#b4b9b4;line-height:1.75;max-width:540px;margin-top:14px;font-size:1.02rem;}
.pillars{display:grid;grid-template-columns:1fr 1fr;gap:14px 40px;margin:28px 0;max-width:540px;}
.pillars span{font-size:.7rem;letter-spacing:.24em;text-transform:uppercase;color:var(--muted);font-weight:700;}
.pillars b{color:#566660;margin-right:10px;font-weight:700;}
.cta{display:inline-block;background:#0f9d6e;color:#ffffff!important;text-decoration:none!important;font-size:.72rem;letter-spacing:.3em;text-transform:uppercase;font-weight:700;padding:22px 32px;border-radius:5px;}
.cta:hover{background:#13b880;}
.tagline{border-top:1px solid var(--line);margin-top:34px;padding-top:18px;color:#a3aca7;font-size:.9rem;max-width:560px;}
.art{position:relative;border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#000;}
.art img{width:100%;display:block;}
.cap{position:absolute;bottom:18px;left:24px;right:24px;display:flex;justify-content:space-between;font-size:.62rem;letter-spacing:.26em;text-transform:uppercase;color:#9aa5a0;font-weight:700;}
.cap .g{color:var(--accent2);}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:54px 0 18px;}
.kpi{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:20px 22px;}
.kpi .v{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:var(--text);}
.kpi .l{font-size:.64rem;letter-spacing:.22em;text-transform:uppercase;color:var(--muted);margin-top:6px;font-weight:700;}
.feats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:22px;}
.feat{display:block;border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:22px;text-decoration:none!important;}
.feat:hover{border-color:var(--accent);}
.feat .n{font-size:.68rem;letter-spacing:.26em;color:var(--accent2);font-weight:700;}
.feat .t{font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;color:var(--text)!important;margin:8px 0;}
.feat .d{color:#a3aca7!important;font-size:.92rem;line-height:1.6;}

/* ---------- inner pages ---------- */
.page-head{margin:4px 0 26px;}
.page-head h2{font-size:2.5rem;font-weight:800;margin:10px 0 8px;padding:0;}
.page-head p{color:#aab3ae;max-width:740px;line-height:1.7;}
.tag{font-size:.66rem;letter-spacing:.26em;text-transform:uppercase;color:var(--muted);font-weight:700;}
.verdict{border:1px solid var(--line);background:var(--panel);border-radius:14px;padding:26px;}
.verdict .big{font-family:'Syne',sans-serif;font-size:2.1rem;font-weight:800;margin:8px 0 14px;}
.good{color:var(--accent2);} .bad{color:var(--warn);}
.bar{height:8px;border-radius:6px;background:#16201d;overflow:hidden;margin:10px 0 16px;}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#0e7a5a,#2ee6a6);}
.rulerow{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:11px 0;border-bottom:1px solid var(--line);font-size:.92rem;}
.rulerow small{color:var(--muted);}
.pill{padding:3px 11px;border-radius:99px;font-size:.64rem;letter-spacing:.14em;text-transform:uppercase;font-weight:700;white-space:nowrap;}
.pill.met{background:#0b3a2c;color:var(--accent2);} .pill.no{background:#3a2a12;color:var(--warn);}
.scen{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0 18px;}
.scen div{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:16px;}
.scen .v{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;}
.scen .base{border-color:var(--accent);}
.prop{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:18px 20px;margin-bottom:12px;}
.prop .t{font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;margin:6px 0;}
.prop .why{color:#aab3ae;font-size:.92rem;line-height:1.6;margin-top:6px;}
.note{border-left:3px solid var(--accent);background:var(--panel);padding:14px 18px;border-radius:0 10px 10px 0;color:#c4cac6;line-height:1.7;margin:14px 0;}
.card{border:1px solid var(--line);background:var(--panel);border-radius:12px;padding:20px 22px;margin-bottom:14px;color:#c4cac6;line-height:1.7;}
.card h4{margin:0 0 8px;font-size:1.1rem;}
.foot{border-top:1px solid var(--line);margin-top:48px;padding-top:18px;color:var(--muted);font-size:.78rem;letter-spacing:.12em;}

/* ---------- Streamlit widgets ---------- */
[data-testid="stFormSubmitButton"] button, .stButton button, .stDownloadButton button{background:#0f9d6e;color:#fff;border:0;border-radius:5px;letter-spacing:.2em;text-transform:uppercase;font-weight:700;font-size:.72rem;padding:.8rem 1.6rem;}
[data-testid="stFormSubmitButton"] button:hover, .stButton button:hover, .stDownloadButton button:hover{background:#13b880;color:#fff;}
[data-testid="stForm"]{border:1px solid var(--line);background:var(--panel);border-radius:14px;padding:22px;}
[data-testid="stWidgetLabel"] p{font-size:.7rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);font-weight:700;}

@media (max-width: 900px){
.hero{grid-template-columns:1fr;} .kpis{grid-template-columns:repeat(2,1fr);} .feats{grid-template-columns:1fr;} .scen{grid-template-columns:1fr;}
}
</style>
"""


def H(s):
    """Remove line breaks and indentation so Markdown treats the HTML as one block."""
    return " ".join(line.strip() for line in s.strip().splitlines())


st.markdown(H(CSS), unsafe_allow_html=True)


# ------------------------------------------------------------------
# IMAGES (drawn as SVG, so no image files are needed)
# ------------------------------------------------------------------
def to_img_src(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def logo_src():
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path d="M24 3 45 24 24 45 3 24Z" '
           f'fill="none" stroke="{BRIGHT}" stroke-width="2.5"/><path d="M24 12 36 24 24 36 12 24Z" '
           f'fill="{GREEN}" opacity=".35"/><circle cx="24" cy="24" r="3.5" fill="{BRIGHT}"/></svg>')
    return to_img_src(svg)


def skyline_src():
    rnd = random.Random(11)
    W, H_, base = 600, 640, 560
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H_}">',
         '<defs>'
         '<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#040706"/>'
         '<stop offset="1" stop-color="#0b1613"/></linearGradient>'
         '<linearGradient id="bld" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#0a1311"/>'
         '<stop offset=".55" stop-color="#10201b"/><stop offset="1" stop-color="#060b09"/></linearGradient>'
         '<radialGradient id="glow" cx=".5" cy="1" r=".6"><stop offset="0" stop-color="#12a370" stop-opacity=".35"/>'
         '<stop offset="1" stop-color="#12a370" stop-opacity="0"/></radialGradient>'
         '<pattern id="win" width="14" height="18" patternUnits="userSpaceOnUse">'
         '<rect x="3" y="4" width="7" height="9" fill="#0b1815"/></pattern></defs>',
         f'<rect width="{W}" height="{H_}" fill="url(#sky)"/>',
         f'<ellipse cx="300" cy="{base + 10}" rx="320" ry="120" fill="url(#glow)"/>']

    def tower(x, w, top, opacity, slant=False):
        if slant:
            shape = f'{x},{base} {x},{top + 40} {x + w},{top} {x + w},{base}'
            p.append(f'<polygon points="{shape}" fill="url(#bld)" stroke="{BRIGHT}" stroke-opacity=".55" '
                     f'stroke-width="1.2" opacity="{opacity}"/>')
            p.append(f'<polygon points="{shape}" fill="url(#win)" opacity="{opacity}"/>')
        else:
            p.append(f'<rect x="{x}" y="{top}" width="{w}" height="{base - top}" fill="url(#bld)" '
                     f'stroke="{BRIGHT}" stroke-opacity=".35" stroke-width="1" opacity="{opacity}"/>')
            p.append(f'<rect x="{x}" y="{top}" width="{w}" height="{base - top}" fill="url(#win)" opacity="{opacity}"/>')
        cols, rows = int((w - 8) / 14), int((base - top - 30) / 18)
        for c in range(cols):
            for r in range(rows):
                if rnd.random() < 0.17:
                    color = rnd.choice(["#c9f5e3", BRIGHT, "#f2d9a0"])
                    p.append(f'<rect x="{x + 7 + c * 14}" y="{top + 48 + r * 18}" width="7" height="9" '
                             f'fill="{color}" opacity="{round(rnd.uniform(.45, .95) * opacity, 2)}"/>')

    for x, w, top in [(40, 84, 300), (130, 72, 360), (432, 92, 330), (520, 64, 390)]:      # back row
        tower(x, w, top, 0.55)
    for x, w, top in [(195, 96, 250), (352, 84, 290)]:                                     # middle row
        tower(x, w, top, 0.85)
    tower(268, 112, 90, 1.0, slant=True)                                                  # tallest tower
    p.append(f'<rect x="0" y="{base}" width="{W}" height="{H_ - base}" fill="#050807"/>')
    p.append(f'<line x1="0" y1="{base}" x2="{W}" y2="{base}" stroke="{GREEN}" stroke-opacity=".5"/>')
    p.append('</svg>')
    return to_img_src("".join(p))


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    d = pd.read_csv("cleaned_housing_data.csv.gz")
    d["Rule_Price"] = d["Price_vs_City_Median"] <= 1.0
    d["Rule_Transport"] = d["Transport_Score"] >= 2
    d["Rule_Schools_Hospitals"] = (d["Nearby_Schools"] + d["Nearby_Hospitals"]) >= 11
    d["Rule_Amenities_Security"] = (d["Amenity_Count"] >= 3) | (d["Security_Code"] == 1)
    return d


df = load_data()
city_median_pps = df.groupby("City")["Price_per_SqFt"].median()
cities = sorted(df["City"].unique())
types = sorted(df["Property_Type"].unique())
bhks = sorted(df["BHK"].unique())
AMENITIES = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]


# ------------------------------------------------------------------
# SMALL HELPERS
# ------------------------------------------------------------------
def gap_pct(series):
    return (series.max() - series.min()) / series.min() * 100


def strength(corr):
    a = abs(corr)
    if a < 0.1:
        return "no meaningful relationship"
    if a < 0.3:
        return "a weak relationship"
    if a < 0.6:
        return "a moderate relationship"
    return "a strong relationship"


def style_ax(fig, ax):
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#1f2d29")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(color="#1a2622", linewidth=0.6)
    ax.set_axisbelow(True)


def chart(w=8, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h))
    style_ax(fig, ax)
    return fig, ax


def style_legend(ax):
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor(PANEL)
        leg.get_frame().set_edgecolor("#1f2d29")
        for t in leg.get_texts():
            t.set_color(TEXT)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def page_head(label, title, text):
    st.markdown(H(f'<div class="page-head"><div class="eyebrow">{label}</div><h2>{title}</h2><p>{text}</p></div>'),
                unsafe_allow_html=True)


def note(text):
    st.markdown(H(f'<div class="note">{text}</div>'), unsafe_allow_html=True)


# ------------------------------------------------------------------
# NAVIGATION (links change ?page=...)
# ------------------------------------------------------------------
PAGES = [("home", "Home"), ("analyse", "Analyse"), ("atlas", "Market Atlas"), ("insights", "Insights"),
         ("advisor", "Advisor"), ("performance", "Performance"), ("about", "About")]

page = st.query_params.get("page", "home")
if page not in [k for k, _ in PAGES]:
    page = "home"

links = "".join(f'<a class="{"active" if k == page else ""}" href="?page={k}" target="_self">{label}</a>'
                for k, label in PAGES)
st.markdown(H(f'<div class="nav"><div class="brand"><img src="{logo_src()}" height="38"><span>{BRAND}</span></div>'
              f'<div class="links">{links}</div></div>'), unsafe_allow_html=True)

# ==================================================================
# HOME
# ==================================================================
if page == "home":
    good_pct = df["Good_Investment"].mean() * 100
    st.markdown(H(f"""
    <div class="hero">
      <div>
        <div class="eyebrow">Property intelligence / India / 2026</div>
        <h1>Invest with <span class="hl">evidence</span>, not instinct.</h1>
        <div class="rule"></div>
        <div class="lead">Every listing tells a story.<br>Every rule shows a reason.</div>
        <div class="sub">An analysis of {len(df):,} Indian property listings. Price, location and neighbourhood
        data become a clear investment verdict and a five-year price scenario, with the reasoning behind both.</div>
        <div class="pillars"><span><b>01</b>/ Price</span><span><b>02</b>/ Location</span>
        <span><b>03</b>/ Infrastructure</span><span><b>04</b>/ Growth</span></div>
        <a class="cta" href="?page=analyse" target="_self">Analyse a property &nbsp;&#8599;</a>
        <div class="tagline">Real data. Clear rules. Honest assumptions.</div>
      </div>
      <div class="art"><img src="{skyline_src()}"><div class="cap"><span>Architectural intelligence</span>
      <span class="g">India &middot; 2026</span></div></div>
    </div>
    <div class="kpis">
      <div class="kpi"><div class="v">{len(df):,}</div><div class="l">Listings analysed</div></div>
      <div class="kpi"><div class="v">{df['City'].nunique()}</div><div class="l">Cities covered</div></div>
      <div class="kpi"><div class="v">{df['Price_in_Lakhs'].mean():.0f} L</div><div class="l">Average price</div></div>
      <div class="kpi"><div class="v">{good_pct:.0f}%</div><div class="l">Rated good investments</div></div>
    </div>
    <div class="feats">
      <a class="feat" href="?page=analyse" target="_self"><div class="n">01</div><div class="t">Analyse</div>
      <div class="d">Enter a property and get a verdict, the rules behind it and a five-year price scenario.</div></a>
      <a class="feat" href="?page=atlas" target="_self"><div class="n">02</div><div class="t">Market Atlas</div>
      <div class="d">Compare price, price per sqft and investment quality across 42 cities.</div></a>
      <a class="feat" href="?page=advisor" target="_self"><div class="n">03</div><div class="t">Advisor</div>
      <div class="d">Set a budget and preferences and get the best matching properties, with reasons.</div></a>
    </div>
    """), unsafe_allow_html=True)

# ==================================================================
# ANALYSE
# ==================================================================
elif page == "analyse":
    page_head("02 / Analyse", "Analyse a property",
              "Enter the details below. You get a verdict, the rules behind it and a five-year price scenario.")

    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            city = st.selectbox("City", cities)
            ptype = st.selectbox("Property type", types)
            bhk = st.selectbox("BHK", bhks)
        with c2:
            size = st.number_input("Size (sqft)", min_value=200, max_value=10000, value=1500, step=50)
            price = st.number_input("Price (lakhs)", min_value=1.0, max_value=5000.0, value=250.0, step=5.0)
            growth = st.slider("Expected yearly price growth (%)", 1.0, 15.0, 6.0, 0.5)
        with c3:
            schools = st.slider("Nearby schools", int(df["Nearby_Schools"].min()), int(df["Nearby_Schools"].max()), 5)
            hospitals = st.slider("Nearby hospitals", int(df["Nearby_Hospitals"].min()),
                                  int(df["Nearby_Hospitals"].max()), 5)
            transport = st.selectbox("Public transport access", ["Low", "Medium", "High"], index=1)
        amenities = st.multiselect("Amenities", AMENITIES, default=["Gym", "Pool"])
        security = st.radio("Security", ["Yes", "No"], horizontal=True)
        submitted = st.form_submit_button("Check investment")

    if submitted:
        pps = price * 100000 / size
        city_med = city_median_pps[city]
        t_score = {"Low": 1, "Medium": 2, "High": 3}[transport]
        rules = pd.DataFrame({
            "Rule": ["Price per sqft at or below the city median", "Medium or high public transport",
                     "Schools plus hospitals of at least 11", "3+ amenities or security"],
            "Detail": [f"Rs {pps:,.0f} per sqft vs city median Rs {city_med:,.0f}", transport,
                       f"{schools + hospitals} nearby",
                       f"{len(amenities)} amenities, security: {security}"],
            "Met": [pps <= city_med, t_score >= 2, (schools + hospitals) >= 11,
                    (len(amenities) >= 3) or security == "Yes"]})
        peers = df[(df["City"] == city) & (df["Property_Type"] == ptype) & (df["BHK"] == bhk)]
        if len(peers) < 30:
            peers = df[(df["City"] == city) & (df["Property_Type"] == ptype)]
        st.session_state["result"] = {"city": city, "ptype": ptype, "price": price, "growth": growth,
                                      "score": int(rules["Met"].sum()), "rules": rules,
                                      "peer_rate": peers["Good_Investment"].mean() * 100, "peer_n": len(peers)}

    if "result" in st.session_state:
        r = st.session_state["result"]
        good = r["score"] >= 3
        strength_txt = {4: "Strong: all 4 rules met", 3: "Borderline: 3 of 4 rules met",
                        2: "Borderline: 2 of 4 rules met", 1: "Weak: 1 of 4 rules met",
                        0: "Weak: no rules met"}[r["score"]]
        rows_html = "".join(
            f'<div class="rulerow"><span>{x.Rule}<br><small>{x.Detail}</small></span>'
            f'<span class="pill {"met" if x.Met else "no"}">{"Met" if x.Met else "Not met"}</span></div>'
            for x in r["rules"].itertuples())

        left, right = st.columns(2)
        with left:
            st.markdown(H(f"""
            <div class="verdict">
              <div class="tag">Verdict</div>
              <div class="big {'good' if good else 'bad'}">{'Good investment' if good else 'Not a good investment'}</div>
              <div class="tag">Rules met: {r['score']} of 4</div>
              <div class="bar"><i style="width:{r['score'] * 25}%"></i></div>
              <div class="tag">Confidence</div><div>{strength_txt}</div>
              <div class="tag" style="margin-top:14px">Similar listings</div>
              <div>{r['peer_rate']:.1f}% of {r['peer_n']:,} similar listings in {r['city']} are good investments</div>
              <div style="margin-top:12px">{rows_html}</div>
            </div>"""), unsafe_allow_html=True)

        with right:
            years = list(range(6))
            scen = {"Low": max(r["growth"] - 2, 0), "Base": r["growth"], "High": r["growth"] + 2}
            proj = pd.DataFrame({k: [r["price"] * (1 + g / 100) ** y for y in years] for k, g in scen.items()},
                                index=years)
            cards = "".join(
                f'<div class="{"base" if k == "Base" else ""}"><div class="tag">{k} &middot; {g:.1f}% a year</div>'
                f'<div class="v">{proj[k].iloc[-1]:,.0f} L</div></div>' for k, g in scen.items())
            st.markdown(H(f'<div class="tag">Estimated price after 5 years</div>'
                          f'<div class="scen">{cards}</div>'), unsafe_allow_html=True)
            fig, ax = chart(6.5, 4)
            ax.plot(proj.index, proj["Low"], marker="o", color=MUTED, label="Low")
            ax.plot(proj.index, proj["Base"], marker="o", color=BRIGHT, linewidth=2.4, label="Base")
            ax.plot(proj.index, proj["High"], marker="o", color=AMBER, label="High")
            ax.set_title("Projected price over 5 years")
            ax.set_xlabel("Years from now")
            ax.set_ylabel("Price (lakhs)")
            ax.legend()
            style_legend(ax)
            show(fig)
            note("The growth rate is your assumption. The dataset has only today's prices, so it cannot "
                 "measure real appreciation. Treat the three lines as a range of scenarios.")

# ==================================================================
# MARKET ATLAS
# ==================================================================
elif page == "atlas":
    page_head("03 / Market Atlas", "Compare markets",
              "See how price, price per sqft and investment quality vary across cities, states and property types.")

    @st.cache_data
    def heat_table(_d, metric, by):
        d = _d.assign(Good_Investment_Pct=_d["Good_Investment"] * 100)
        return d.pivot_table(values=metric, index="City", columns=by, aggfunc="mean")

    a1, a2 = st.columns(2)
    with a1:
        measure = st.selectbox("Measure", ["Average price (lakhs)", "Average price per sqft (Rs)",
                                           "Good investment %"])
    with a2:
        by = st.selectbox("Compare across", ["Property_Type", "BHK"])
    metric = {"Average price (lakhs)": "Price_in_Lakhs", "Average price per sqft (Rs)": "Price_per_SqFt",
              "Good investment %": "Good_Investment_Pct"}[measure]

    table = heat_table(df, metric, by)
    cmap = LinearSegmentedColormap.from_list("estate", ["#0d1412", "#0e7a5a", BRIGHT])
    fig, ax = plt.subplots(figsize=(8, 12))
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    sns.heatmap(table, annot=True, fmt=".0f", cmap=cmap, ax=ax, linewidths=0.5, linecolor=PANEL,
                annot_kws={"color": TEXT, "fontsize": 8})
    ax.set_title(f"{measure} by city and {by}", color=TEXT)
    ax.set_xlabel(by, color=MUTED)
    ax.set_ylabel("City", color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.collections[0].colorbar.ax.tick_params(colors=MUTED)
    show(fig)

    city_avg = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
    note(f"Average price runs from {city_avg.iloc[-1]:.1f} lakhs ({city_avg.index[-1]}) to {city_avg.iloc[0]:.1f} "
         f"lakhs ({city_avg.index[0]}), a gap of only {gap_pct(city_avg):.1f}% across {len(city_avg)} cities.")

    st.markdown('<div class="tag" style="margin-top:22px">State view</div>', unsafe_allow_html=True)
    state = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False)
    fig, ax = chart(12, 4.5)
    state.plot(kind="bar", color=GREEN, ax=ax)
    ax.set_title("Average price per sqft by state")
    ax.set_xlabel("State")
    ax.set_ylabel("Price per sqft (Rs)")
    plt.xticks(rotation=60, ha="right")
    show(fig)
    note(f"Highest: {state.index[0]} (Rs {state.iloc[0]:,.0f} per sqft). Lowest: {state.index[-1]} "
         f"(Rs {state.iloc[-1]:,.0f}). The gap across {len(state)} states is {gap_pct(state):.1f}%.")

# ==================================================================
# INSIGHTS
# ==================================================================
elif page == "insights":
    page_head("04 / Insights", "What drives price?",
              "Pick a question and see what the data says. Where a factor has little effect, the text says so.")

    choice = st.selectbox("Question", [
        "How are prices distributed?", "Does size affect price?", "Price per sqft by property type",
        "Does furnishing affect price?", "Do amenities affect price per sqft?",
        "Do schools and hospitals affect price per sqft?", "Transport and investment potential",
        "Does property age affect price?"])

    if choice == "How are prices distributed?":
        p = df["Price_in_Lakhs"]
        fig, ax = chart()
        ax.hist(p, bins=30, color=GREEN, edgecolor=PANEL)
        ax.set_title("Distribution of property prices")
        ax.set_xlabel("Price (lakhs)")
        ax.set_ylabel("Number of properties")
        show(fig)
        note(f"Prices range from {p.min():.0f} to {p.max():.0f} lakhs (mean {p.mean():.1f}, median {p.median():.1f}). "
             f"The spread is balanced, with skewness of {p.skew():.2f}.")

    elif choice == "Does size affect price?":
        c = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
        s = df.sample(5000, random_state=1)
        fig, ax = chart()
        ax.scatter(s["Size_in_SqFt"], s["Price_in_Lakhs"], s=8, alpha=0.4, color=BRIGHT)
        ax.set_title("Size vs price (sample of 5,000)")
        ax.set_xlabel("Size (sqft)")
        ax.set_ylabel("Price (lakhs)")
        show(fig)
        note(f"The correlation is {c:.3f}: {strength(c)}. Bigger properties are not priced higher in this data.")

    elif choice == "Price per sqft by property type":
        t = df.groupby("Property_Type")["Price_per_SqFt"].mean()
        fig, ax = chart(6.5, 4)
        t.plot(kind="bar", color=GREEN, ax=ax)
        ax.set_title("Average price per sqft by property type")
        ax.set_xlabel("Property type")
        ax.set_ylabel("Price per sqft (Rs)")
        plt.xticks(rotation=0)
        show(fig)
        note(f"The gap between the highest and lowest type is {gap_pct(t):.1f}%, so property type has little effect.")

    elif choice == "Does furnishing affect price?":
        f = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean()
        fig, ax = chart(6.5, 4)
        f.plot(kind="bar", color=BRIGHT, ax=ax)
        ax.set_title("Average price by furnished status")
        ax.set_xlabel("Furnished status")
        ax.set_ylabel("Average price (lakhs)")
        plt.xticks(rotation=0)
        show(fig)
        note(f"Average prices differ by only {gap_pct(f):.1f}% across furnished statuses.")

    elif choice == "Do amenities affect price per sqft?":
        rows = []
        for a in AMENITIES:
            has = df["Amenities"].str.contains(a)
            rows.append({"Amenity": a, "With": df.loc[has, "Price_per_SqFt"].mean(),
                         "Without": df.loc[~has, "Price_per_SqFt"].mean()})
        am = pd.DataFrame(rows).set_index("Amenity")
        fig, ax = chart(8, 4.2)
        am.plot(kind="bar", ax=ax, color=[BRIGHT, MUTED])
        ax.set_title("Price per sqft with and without each amenity")
        ax.set_xlabel("Amenity")
        ax.set_ylabel("Price per sqft (Rs)")
        plt.xticks(rotation=0)
        style_legend(ax)
        show(fig)
        diff = (am["With"] - am["Without"]).abs().max()
        note(f"Each amenity changes price per sqft by at most Rs {diff:,.0f}, so amenities add little to price here.")

    elif choice == "Do schools and hospitals affect price per sqft?":
        cs = df["Nearby_Schools"].corr(df["Price_per_SqFt"])
        ch = df["Nearby_Hospitals"].corr(df["Price_per_SqFt"])
        fig, ax = chart(8, 4.2)
        df.groupby("Nearby_Schools")["Price_per_SqFt"].mean().plot(marker="o", color=BRIGHT, ax=ax, label="Schools")
        df.groupby("Nearby_Hospitals")["Price_per_SqFt"].mean().plot(marker="o", color=AMBER, ax=ax, label="Hospitals")
        ax.set_title("Price per sqft by number of nearby schools and hospitals")
        ax.set_xlabel("Number nearby")
        ax.set_ylabel("Price per sqft (Rs)")
        ax.legend()
        style_legend(ax)
        show(fig)
        note(f"Schools: correlation {cs:.3f} ({strength(cs)}). Hospitals: correlation {ch:.3f} ({strength(ch)}).")

    elif choice == "Transport and investment potential":
        t = df.groupby("Public_Transport_Accessibility").agg(
            pps=("Price_per_SqFt", "mean"), gi=("Good_Investment", lambda x: x.mean() * 100)).loc[
            ["Low", "Medium", "High"]]
        k1, k2 = st.columns(2)
        with k1:
            fig, ax = chart(5.5, 3.8)
            t["pps"].plot(kind="bar", color=GREEN, ax=ax)
            ax.set_title("Price per sqft by transport access")
            ax.set_xlabel("Transport access")
            ax.set_ylabel("Price per sqft (Rs)")
            plt.xticks(rotation=0)
            show(fig)
        with k2:
            fig, ax = chart(5.5, 3.8)
            t["gi"].plot(kind="bar", color=BRIGHT, ax=ax)
            ax.set_title("Good investment % by transport access")
            ax.set_xlabel("Transport access")
            ax.set_ylabel("Good investment (%)")
            plt.xticks(rotation=0)
            show(fig)
        note(f"Price per sqft differs by only {gap_pct(t['pps']):.1f}% across transport levels. Good investment % is "
             f"much higher for Medium and High access, but transport is one of the four rules that define a good "
             f"investment, so that gap is built in.")

    else:
        g = df.groupby("Age_Group", observed=True)["Price_in_Lakhs"].mean()
        c = df["Age_of_Property"].corr(df["Price_in_Lakhs"])
        fig, ax = chart(6.5, 4)
        g.plot(kind="bar", color=GREEN, ax=ax)
        ax.set_title("Average price by property age group")
        ax.set_xlabel("Age group")
        ax.set_ylabel("Average price (lakhs)")
        plt.xticks(rotation=0)
        show(fig)
        note(f"The correlation between age and price is {c:.3f}: {strength(c)}.")

# ==================================================================
# ADVISOR (rule-based recommendations)
# ==================================================================
elif page == "advisor":
    page_head("05 / Advisor", "Find the best matches",
              "Set your budget and preferences. The advisor lists good investments that match, "
              "ranked by how far below the city median price per sqft they are.")

    with st.form("advisor_form"):
        f1, f2 = st.columns(2)
        with f1:
            budget = st.slider("Budget (lakhs)", int(df["Price_in_Lakhs"].min()), int(df["Price_in_Lakhs"].max()),
                               (100, 300))
            sel_cities = st.multiselect("Cities (leave empty for all)", cities, default=[])
            sel_types = st.multiselect("Property type", types, default=types)
        with f2:
            sel_bhk = st.multiselect("BHK", bhks, default=bhks)
            min_transport = st.selectbox("Minimum public transport", ["Low", "Medium", "High"], index=1)
            top_n = st.slider("Number of results", 3, 20, 8)
        go = st.form_submit_button("Find properties")

    if go:
        t_min = {"Low": 1, "Medium": 2, "High": 3}[min_transport]
        cand = df[(df["Good_Investment"] == 1) & df["Price_in_Lakhs"].between(*budget)
                  & df["Property_Type"].isin(sel_types) & df["BHK"].isin(sel_bhk)
                  & (df["Transport_Score"] >= t_min)]
        if sel_cities:
            cand = cand[cand["City"].isin(sel_cities)]
        st.session_state["advice"] = {
            "n": len(cand),
            "top": cand.sort_values(["Investment_Score", "Price_vs_City_Median", "Infrastructure_Score"],
                                    ascending=[False, True, False]).head(top_n)}

    if "advice" in st.session_state:
        adv = st.session_state["advice"]
        if adv["n"] == 0:
            st.warning("No good investments match these preferences. Try a wider budget or fewer filters.")
        else:
            st.markdown(f'<div class="tag">{adv["n"]:,} good investments match. Showing the top {len(adv["top"])}.</div>',
                        unsafe_allow_html=True)
            for _, row in adv["top"].iterrows():
                why = []
                disc = (1 - row["Price_vs_City_Median"]) * 100
                why.append(f"{disc:.0f}% below the city median price per sqft" if disc >= 1
                           else "priced at the city median per sqft")
                why.append(f"{row['Public_Transport_Accessibility'].lower()} transport access")
                why.append(f"{int(row['Nearby_Schools'])} schools and {int(row['Nearby_Hospitals'])} hospitals nearby")
                why.append(f"{int(row['Amenity_Count'])} amenities" + (", with security" if row["Security_Code"] == 1 else ""))
                st.markdown(H(f"""
                <div class="prop">
                  <div class="tag">{row['City']} &middot; {row['Locality']} &middot; rules met {int(row['Investment_Score'])} of 4</div>
                  <div class="t">{row['Property_Type']} &middot; {int(row['BHK'])} BHK &middot; {int(row['Size_in_SqFt']):,} sqft</div>
                  <div>Rs {row['Price_in_Lakhs']:.1f} lakhs &nbsp;(Rs {row['Price_per_SqFt']:,.0f} per sqft)</div>
                  <div class="why">{' &middot; '.join(why)}</div>
                </div>"""), unsafe_allow_html=True)
            cols = ["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
                    "Investment_Score"]
            st.download_button("Download these results (CSV)", data=adv["top"][cols].to_csv(index=False),
                               file_name="advisor_results.csv", mime="text/csv")

# ==================================================================
# PERFORMANCE
# ==================================================================
elif page == "performance":
    page_head("06 / Performance", "How the rules perform",
              "The good-investment verdict is rule-based. This page shows how many properties meet each rule, "
              "how scores are spread and how growth assumptions change the five-year price.")

    score_pct = df["Investment_Score"].value_counts(normalize=True).sort_index() * 100
    p1, p2 = st.columns(2)
    with p1:
        fig, ax = chart(6, 4)
        ax.bar(score_pct.index.astype(str), score_pct.values,
               color=[MUTED if i < 3 else BRIGHT for i in score_pct.index])
        ax.set_title("Share of properties by rules met")
        ax.set_xlabel("Rules met (out of 4)")
        ax.set_ylabel("Properties (%)")
        show(fig)
    names = {"Rule_Price": "Price at or below city median", "Rule_Transport": "Medium or high transport",
             "Rule_Schools_Hospitals": "Schools + hospitals >= 11", "Rule_Amenities_Security": "3+ amenities or security"}
    cover = pd.Series({n: df[c].mean() * 100 for c, n in names.items()})
    with p2:
        fig, ax = chart(6, 4)
        cover.sort_values().plot(kind="barh", color=GREEN, ax=ax)
        ax.set_title("Properties meeting each rule")
        ax.set_xlabel("Properties (%)")
        ax.set_ylabel("")
        show(fig)

    rows = []
    for c, n in names.items():
        met = df.loc[df[c], "Good_Investment"].mean() * 100
        not_met = df.loc[~df[c], "Good_Investment"].mean() * 100
        rows.append({"Rule": n, "Good investment % when met": round(met, 1),
                     "Good investment % when not met": round(not_met, 1), "Difference (points)": round(met - not_met, 1)})
    st.dataframe(pd.DataFrame(rows).sort_values("Difference (points)", ascending=False), hide_index=True)
    note(f"{df['Good_Investment'].mean() * 100:.1f}% of listings meet at least 3 of 4 rules. Because the label is built "
         f"from these rules, this table shows how strongly each rule drives the verdict. It is not a statistical model.")

    st.markdown('<div class="tag" style="margin-top:20px">Growth scenarios for the average listing</div>',
                unsafe_allow_html=True)
    avg = df["Price_in_Lakhs"].mean()
    g_rows = [{"Yearly growth": f"{g}%", "Price today (lakhs)": round(avg, 1),
               "Price after 5 years (lakhs)": round(avg * (1 + g / 100) ** 5, 1),
               "Gain (lakhs)": round(avg * ((1 + g / 100) ** 5 - 1), 1)} for g in [4, 6, 8, 10]]
    st.dataframe(pd.DataFrame(g_rows), hide_index=True)
    note("These growth rates are example assumptions. The dataset has no past prices to measure real appreciation.")

# ==================================================================
# ABOUT
# ==================================================================
else:
    page_head("07 / About", "About this project",
              "A data analyst application that helps investors judge properties, with clear rules and honest assumptions.")

    st.markdown(H(f"""
    <div class="card"><h4>Problem statement</h4>
    Help potential investors make real estate decisions by classifying whether a property is a good investment
    and estimating its price after five years, using {len(df):,} property listings across {df['City'].nunique()} cities
    and {df['State'].nunique()} states in India.</div>
    <div class="card"><h4>Approach</h4>
    1. Clean the data: missing values, duplicates, inconsistent floors, price per sqft recalculated in rupees.<br>
    2. Engineer features: price per sqft, school density score, infrastructure score, amenity count, price vs city median.<br>
    3. Analyse: 20 questions on price, size, location, features and infrastructure.<br>
    4. Classify: a property is a good investment when it meets at least 3 of 4 rules.<br>
    5. Estimate: five-year price from the current price and an adjustable yearly growth rate.<br>
    6. Present: this Streamlit application.</div>
    <div class="card"><h4>The four investment rules</h4>
    Price per sqft at or below the city median &middot; medium or high public transport &middot;
    schools plus hospitals of at least 11 &middot; 3 or more amenities, or security.</div>
    <div class="card"><h4>Assumptions and limitations</h4>
    &bull; No machine learning is used. The verdict is rule-based and the thresholds are analyst choices.<br>
    &bull; The dataset has no past prices, so the five-year estimate is a scenario based on a growth rate you choose, not a forecast learned from data.<br>
    &bull; Price, size and BHK are almost independent in this data, so most factors change price by only a few percent. The data looks synthetic, so findings should not be read as real market facts.<br>
    &bull; The same locality names appear in every city, so a locality result mixes properties from different cities.</div>
    """), unsafe_allow_html=True)

st.markdown(H(f'<div class="foot">{BRAND.upper()} &nbsp;/&nbsp; REAL DATA. CLEAR RULES. HONEST ASSUMPTIONS.</div>'),
            unsafe_allow_html=True)
