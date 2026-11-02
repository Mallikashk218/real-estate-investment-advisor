"""
India Housing Prices - Streamlit EDA Dashboard
Install once:  pip install streamlit pandas matplotlib seaborn
Run:           streamlit run app.py
Needs cleaned_housing_data.csv in the same folder.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ------------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------------
st.set_page_config(page_title="India Housing EDA", layout="wide")
st.title("India Housing Prices - Exploratory Data Analysis")
st.caption("Price trends, size, safety and infrastructure vs investment")


# ------------------------------------------------------------------
# LOAD DATA (cached so it only loads once)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_housing_data.csv")


df = load_data()

# ------------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------------
st.sidebar.header("Filters")

cities = st.sidebar.multiselect("City", sorted(df["City"].unique()),
                                default=sorted(df["City"].unique()))
types = st.sidebar.multiselect("Property Type", sorted(df["Property_Type"].unique()),
                               default=sorted(df["Property_Type"].unique()))

df = df[df["City"].isin(cities) & df["Property_Type"].isin(types)]

if df.empty:
    st.warning("No data for the selected filters. Please select at least one city and one property type.")
    st.stop()

# ------------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(df):,}")
c2.metric("Avg Price (Lakhs)", f"{df['Price_in_Lakhs'].mean():.1f}")
c3.metric("Avg Price per SqFt (Rs)", f"{df['Price_per_SqFt'].mean():,.0f}")
c4.metric("Good Investment %", f"{df['Good_Investment'].mean() * 100:.1f}%")

# ------------------------------------------------------------------
# TABS - one for each analysis
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Price by City", "Size vs Investment",
                                  "Security vs Investment", "Infrastructure vs Price"])

# ---------- TAB 1: PRICE TRENDS BY CITY ----------
with tab1:
    st.subheader("Average Price by City")

    top_n = st.slider("Number of cities to show", 5, 20, 10)
    city_price = (df.groupby("City")["Price_in_Lakhs"].mean()
                    .sort_values(ascending=False).head(top_n))

    fig, ax = plt.subplots(figsize=(10, 5))
    city_price.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title(f"Top {top_n} Cities by Average Price")
    ax.set_xlabel("City")
    ax.set_ylabel("Average Price (Lakhs)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Average Price by Year Built")
    fig, ax = plt.subplots(figsize=(10, 4))
    df.groupby("Year_Built")["Price_in_Lakhs"].mean().plot(marker="o", ax=ax)
    ax.set_title("Average Price by Year Built")
    ax.set_xlabel("Year Built")
    ax.set_ylabel("Average Price (Lakhs)")
    st.pyplot(fig)

    st.info("Insight: Average prices are very similar across cities, so city alone "
            "does not explain price differences in this data.")

# ---------- TAB 2: SIZE VS INVESTMENT ----------
with tab2:
    st.subheader("Property Size vs Investment")

    df["Size_Band"] = pd.cut(df["Size_in_SqFt"], bins=[0, 1000, 2000, 3000, 4000, 5000],
                             labels=["<1000", "1000-2000", "2000-3000", "3000-4000", "4000-5000"])
    size_summary = df.groupby("Size_Band", observed=True).agg(
        Avg_Price_per_SqFt=("Price_per_SqFt", "mean"),
        Good_Investment_Pct=("Good_Investment", lambda x: x.mean() * 100)).round(1)

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        size_summary["Good_Investment_Pct"].plot(kind="bar", color="green", ax=ax)
        ax.set_title("Good Investment % by Size")
        ax.set_xlabel("Size (SqFt)")
        ax.set_ylabel("Good Investment (%)")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        size_summary["Avg_Price_per_SqFt"].plot(kind="bar", color="orange", ax=ax)
        ax.set_title("Average Price per SqFt by Size")
        ax.set_xlabel("Size (SqFt)")
        ax.set_ylabel("Price per SqFt (Rs)")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    st.dataframe(size_summary)
    st.info("Insight: Larger properties have a lower price per sqft, so they meet the "
            "'priced below city median' rule more often and show a higher Good Investment %.")

# ---------- TAB 3: SECURITY VS GOOD INVESTMENT ----------
with tab3:
    st.subheader("Security vs Good Investment")
    st.caption("The dataset has no crime rate column, so Security (Yes/No) is used as the safety measure.")

    security = df.groupby("Security")["Good_Investment"].mean().mul(100).round(1)
    security.index = ["No Security" if i == "No" else "Has Security" for i in security.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    security.plot(kind="bar", color=["red", "green"][:len(security)], ax=ax)
    ax.set_title("Good Investment % by Security")
    ax.set_xlabel("Security")
    ax.set_ylabel("Good Investment (%)")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    st.info("Insight: Properties with security have a higher Good Investment %. "
            "Note that Security is one of the rules used to build the label.")

# ---------- TAB 4: INFRASTRUCTURE VS PRICE ----------
with tab4:
    st.subheader("Infrastructure Score vs Price")

    corr = df["Infrastructure_Score"].corr(df["Price_in_Lakhs"])
    st.metric("Correlation (Infrastructure vs Price)", f"{corr:.3f}")

    df["Infra_Level"] = pd.qcut(df["Infrastructure_Score"], 4,
                                labels=["Low", "Medium", "High", "Very High"],
                                duplicates="drop")
    infra = df.groupby("Infra_Level", observed=True)["Price_in_Lakhs"].mean().round(1)

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        infra.plot(kind="bar", color="orange", ax=ax)
        ax.set_title("Average Price by Infrastructure Level")
        ax.set_xlabel("Infrastructure Level")
        ax.set_ylabel("Average Price (Lakhs)")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    with col_b:
        sample = df.sample(min(5000, len(df)), random_state=1)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=sample, x="Infrastructure_Score", y="Price_in_Lakhs",
                        alpha=0.4, ax=ax)
        ax.set_title("Infrastructure Score vs Price")
        ax.set_xlabel("Infrastructure Score")
        ax.set_ylabel("Price (Lakhs)")
        st.pyplot(fig)

    st.info("Insight: The correlation is close to zero, so higher infrastructure does not "
            "go with higher prices in this data.")

# ------------------------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------------------------
with st.expander("Show data preview"):
    st.dataframe(df.head(100))
