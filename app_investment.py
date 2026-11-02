"""
Real Estate Investment Analyzer - Streamlit app for data analysts (no machine learning)
Install once:  pip install streamlit pandas matplotlib seaborn
Run:           streamlit run app_investment.py
Needs cleaned_housing_data.csv (from the cleaning script) in the same folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Real Estate Investment Analyzer", layout="wide")
st.title("Real Estate Investment Analyzer")
st.caption("Rule-based analysis built from the housing data. No machine learning is used.")

AMENITIES = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]


# ------------------------------------------------------------------
# LOAD DATA AND ADD THE 4 INVESTMENT RULES (same rules as the cleaning script)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    d = pd.read_csv("cleaned_housing_data.csv.gz")
    d["Rule_Price"] = d["Price_vs_City_Median"] <= 1.0                    # at/below city median price per sqft
    d["Rule_Transport"] = d["Transport_Score"] >= 2                       # medium or high transport
    d["Rule_Schools_Hospitals"] = (d["Nearby_Schools"] + d["Nearby_Hospitals"]) >= 11
    d["Rule_Amenities_Security"] = (d["Amenity_Count"] >= 3) | (d["Security_Code"] == 1)
    return d


df = load_data()
city_median_pps = df.groupby("City")["Price_per_SqFt"].median()
cities = sorted(df["City"].unique())
types = sorted(df["Property_Type"].unique())
bhks = sorted(df["BHK"].unique())

with st.expander("How this app works (read first)"):
    st.markdown("""
- **Good Investment** = the property meets at least **3 of 4 rules**: price per sqft at or below the city median,
  medium/high public transport, schools + hospitals of at least 11, and 3+ amenities or security.
- **Confidence** = how many rules are met (out of 4), plus the share of similar listings
  (same city, type and BHK) that are Good Investments.
- **Price after 5 years** = today's price grown at a yearly rate that **you choose**.
  The dataset has no past prices, so the growth rate is an assumption, not something learned from the data.
- **Factor importance** = how much the Good Investment % changes when each rule is met.
""")

tab1, tab2, tab3, tab4 = st.tabs(["Check a Property", "Search Properties",
                                  "Market Insights", "Why This Result"])

# ==================================================================
# TAB 1: PROPERTY FORM + CLASSIFICATION + 5-YEAR PRICE
# ==================================================================
with tab1:
    st.subheader("Enter property details")

    with st.form("property_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            city = st.selectbox("City", cities)
            ptype = st.selectbox("Property Type", types)
            bhk = st.selectbox("BHK", bhks)
        with c2:
            size = st.number_input("Size (SqFt)", min_value=200, max_value=10000, value=1500, step=50)
            price = st.number_input("Price (Lakhs)", min_value=1.0, max_value=5000.0, value=250.0, step=5.0)
            growth = st.slider("Expected yearly price growth (%)", 1.0, 15.0, 6.0, 0.5)
        with c3:
            schools = st.slider("Nearby schools", int(df["Nearby_Schools"].min()),
                                int(df["Nearby_Schools"].max()), 5)
            hospitals = st.slider("Nearby hospitals", int(df["Nearby_Hospitals"].min()),
                                  int(df["Nearby_Hospitals"].max()), 5)
            transport = st.selectbox("Public transport access", ["Low", "Medium", "High"], index=1)

        amenities = st.multiselect("Amenities", AMENITIES, default=["Gym", "Pool"])
        security = st.radio("Security", ["Yes", "No"], horizontal=True)
        submitted = st.form_submit_button("Check investment")

    if submitted:
        pps = price * 100000 / size
        city_med = city_median_pps[city]
        transport_score = {"Low": 1, "Medium": 2, "High": 3}[transport]

        rules = pd.DataFrame({
            "Rule": ["Price per sqft at or below city median",
                     "Medium or High public transport",
                     "Schools + hospitals of at least 11",
                     "3+ amenities or security"],
            "Your property": [f"Rs {pps:,.0f} per sqft (city median Rs {city_med:,.0f})",
                              transport,
                              f"{schools + hospitals}",
                              f"{len(amenities)} amenities, security: {security}"],
            "Met": [pps <= city_med,
                    transport_score >= 2,
                    (schools + hospitals) >= 11,
                    (len(amenities) >= 3) or (security == "Yes")]})
        score = int(rules["Met"].sum())

        # Similar listings: same city + type + BHK (fall back to city + type if too few)
        peers = df[(df["City"] == city) & (df["Property_Type"] == ptype) & (df["BHK"] == bhk)]
        if len(peers) < 30:
            peers = df[(df["City"] == city) & (df["Property_Type"] == ptype)]

        st.session_state["result"] = {
            "city": city, "ptype": ptype, "price": price, "growth": growth, "score": score,
            "rules": rules, "peer_rate": peers["Good_Investment"].mean() * 100, "peer_n": len(peers)}

    if "result" in st.session_state:
        r = st.session_state["result"]
        st.divider()
        left, right = st.columns(2)

        # ---- Classification ----
        with left:
            st.subheader("Is this a Good Investment?")
            if r["score"] >= 3:
                st.success("YES - Good Investment")
            else:
                st.error("NO - Not a Good Investment")

            strength = {4: "Strong (all 4 rules met)", 3: "Borderline (3 of 4 rules met)",
                        2: "Borderline (2 of 4 rules met)", 1: "Weak (1 of 4 rules met)",
                        0: "Weak (no rules met)"}[r["score"]]
            st.metric("Rules met", f"{r['score']} / 4")
            st.progress(r["score"] / 4)
            st.write("Confidence:", strength)
            st.write(f"Similar listings in {r['city']} ({r['ptype']}): "
                     f"**{r['peer_rate']:.1f}%** are Good Investments "
                     f"(based on {r['peer_n']:,} listings).")
            st.dataframe(r["rules"].assign(Met=r["rules"]["Met"].map({True: "Met", False: "Not met"})),
                         hide_index=True)

        # ---- 5-year price ----
        with right:
            st.subheader("Estimated price after 5 years")
            years = list(range(0, 6))
            scenarios = {"Low": max(r["growth"] - 2, 0), "Base": r["growth"], "High": r["growth"] + 2}
            proj = pd.DataFrame({name: [r["price"] * (1 + g / 100) ** y for y in years]
                                 for name, g in scenarios.items()}, index=years)

            base = proj["Base"].iloc[-1]
            st.metric("Base estimate (Lakhs)", f"{base:,.1f}",
                      delta=f"+{base - r['price']:,.1f} vs today")
            summary = pd.DataFrame({
                "Yearly growth": [f"{g:.1f}%" for g in scenarios.values()],
                "Price in 5 years (Lakhs)": [round(proj[n].iloc[-1], 1) for n in scenarios]},
                index=list(scenarios.keys()))
            st.dataframe(summary)

            fig, ax = plt.subplots(figsize=(6, 4))
            proj.plot(marker="o", ax=ax)
            ax.set_title("Projected Price over 5 Years")
            ax.set_xlabel("Years from now")
            ax.set_ylabel("Price (Lakhs)")
            st.pyplot(fig)
            st.caption("The growth rate is your assumption. The dataset has no past prices to measure it.")

# ==================================================================
# TAB 2: SEARCH / FILTER PROPERTIES
# ==================================================================
with tab2:
    st.subheader("Filter properties")

    f1, f2 = st.columns(2)
    with f1:
        sel_cities = st.multiselect("City", cities, default=[])
        sel_types = st.multiselect("Property Type", types, default=types)
        sel_bhk = st.multiselect("BHK", bhks, default=bhks)
    with f2:
        size_min, size_max = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())
        price_min, price_max = int(df["Price_in_Lakhs"].min()), int(df["Price_in_Lakhs"].max())
        size_range = st.slider("Area (SqFt)", size_min, size_max, (size_min, size_max))
        price_range = st.slider("Price (Lakhs)", price_min, price_max, (price_min, price_max))
        only_good = st.checkbox("Show only Good Investments")

    found = df[df["Property_Type"].isin(sel_types) & df["BHK"].isin(sel_bhk)
               & df["Size_in_SqFt"].between(*size_range) & df["Price_in_Lakhs"].between(*price_range)]
    if sel_cities:
        found = found[found["City"].isin(sel_cities)]
    if only_good:
        found = found[found["Good_Investment"] == 1]

    k1, k2, k3 = st.columns(3)
    k1.metric("Matching properties", f"{len(found):,}")
    if len(found) > 0:
        k2.metric("Average price (Lakhs)", f"{found['Price_in_Lakhs'].mean():.1f}")
        k3.metric("Good Investment %", f"{found['Good_Investment'].mean() * 100:.1f}%")

        show_cols = ["City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                     "Price_per_SqFt", "Infrastructure_Score", "Good_Investment"]
        result = found[show_cols].sort_values("Price_per_SqFt").head(1000)
        st.write("Showing up to 1,000 properties, cheapest price per sqft first:")
        st.dataframe(result, hide_index=True)
        st.download_button("Download these results (CSV)", data=result.to_csv(index=False),
                           file_name="filtered_properties.csv", mime="text/csv")
    else:
        st.warning("No properties match these filters.")


# ==================================================================
# TAB 3: MARKET INSIGHTS (HEATMAP + TREND)
# ==================================================================
@st.cache_data
def make_heatmap_table(_d, metric, columns_by):
    d = _d.assign(Good_Investment_Pct=_d["Good_Investment"] * 100)
    return d.pivot_table(values=metric, index="City", columns=columns_by, aggfunc="mean")


with tab3:
    st.subheader("Location-wise heatmap")

    h1, h2 = st.columns(2)
    with h1:
        metric_label = st.selectbox("Measure", ["Average Price (Lakhs)", "Average Price per SqFt (Rs)",
                                                "Good Investment %"])
    with h2:
        by_label = st.selectbox("Compare across", ["Property_Type", "BHK"])

    metric_col = {"Average Price (Lakhs)": "Price_in_Lakhs",
                  "Average Price per SqFt (Rs)": "Price_per_SqFt",
                  "Good Investment %": "Good_Investment_Pct"}[metric_label]

    table = make_heatmap_table(df, metric_col, by_label)
    fig, ax = plt.subplots(figsize=(8, 12))
    sns.heatmap(table, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title(f"{metric_label} by City and {by_label}")
    ax.set_xlabel(by_label)
    ax.set_ylabel("City")
    st.pyplot(fig)

    st.subheader("Trend chart")
    fig, ax = plt.subplots(figsize=(10, 4))
    df.groupby("Year_Built")["Price_in_Lakhs"].mean().plot(marker="o", ax=ax)
    ax.set_title("Average Price by Year Built")
    ax.set_xlabel("Year Built")
    ax.set_ylabel("Average Price (Lakhs)")
    st.pyplot(fig)

    st.subheader("Top 10 cities by Good Investment %")
    top = (df.groupby("City")["Good_Investment"].mean() * 100).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 4))
    top.plot(kind="bar", color="green", ax=ax)
    ax.set_title("Top 10 Cities by Good Investment %")
    ax.set_xlabel("City")
    ax.set_ylabel("Good Investment (%)")
    plt.xticks(rotation=45)
    st.pyplot(fig)


# ==================================================================
# TAB 4: WHY THIS RESULT (FACTOR IMPORTANCE)
# ==================================================================
@st.cache_data
def rule_impact(_d):
    names = {"Rule_Price": "Price at or below city median",
             "Rule_Transport": "Medium or High transport",
             "Rule_Schools_Hospitals": "Schools + hospitals >= 11",
             "Rule_Amenities_Security": "3+ amenities or security"}
    rows = []
    for col, name in names.items():
        met = _d.loc[_d[col], "Good_Investment"].mean() * 100
        not_met = _d.loc[~_d[col], "Good_Investment"].mean() * 100
        rows.append({"Factor": name, "Good Investment % when met": round(met, 1),
                     "Good Investment % when not met": round(not_met, 1),
                     "Difference (points)": round(met - not_met, 1)})
    return pd.DataFrame(rows).sort_values("Difference (points)", ascending=False)


with tab4:
    st.subheader("Which factors matter most?")
    impact = rule_impact(df)
    st.dataframe(impact, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    impact.set_index("Factor")["Difference (points)"].plot(kind="barh", color="steelblue", ax=ax)
    ax.set_title("Impact of Each Factor on Good Investment %")
    ax.set_xlabel("Increase in Good Investment % when the factor is met")
    ax.set_ylabel("")
    st.pyplot(fig)

    st.info("The Good Investment label is built from these four rules, so this chart shows how strongly "
            "each rule drives the classification. It is not a statistical model.")
    st.caption("Overall, "
               f"{df['Good_Investment'].mean() * 100:.1f}% of the {len(df):,} listings are Good Investments.")
