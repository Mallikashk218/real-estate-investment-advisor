"""
India Housing Prices - EDA Dashboard (20 questions, no machine learning)
Install once:  pip install streamlit pandas matplotlib seaborn
Run:           streamlit run app_eda_20_questions.py
Needs cleaned_housing_data.csv (from the cleaning script) in the same folder.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Housing EDA - 20 Questions", layout="wide")
st.title("India Housing Prices - Exploratory Data Analysis")
st.caption("20 questions answered in five parts. Select a part in the sidebar.")
sns.set_style("whitegrid")


@st.cache_data
def load_data():
    return pd.read_csv("cleaned_housing_data.csv")


df = load_data()


# ------------------------------------------------------------------
# SMALL HELPERS
# ------------------------------------------------------------------
def question(number, text):
    st.divider()
    st.subheader(f"Q{number}. {text}")


def strength(corr):
    """Turn a correlation number into plain words."""
    a = abs(corr)
    if a < 0.1:
        return "no meaningful relationship"
    if a < 0.3:
        return "a weak relationship"
    if a < 0.6:
        return "a moderate relationship"
    return "a strong relationship"


def gap_pct(series):
    """Gap between the highest and lowest value, as a % of the lowest."""
    return (series.max() - series.min()) / series.min() * 100


def new_chart(w=8, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
part = st.sidebar.radio("Select a part", [
    "A. What is in the data?",
    "B. Price and size",
    "C. Location",
    "D. Property features",
    "E. Infrastructure and investment"])

st.sidebar.metric("Properties", f"{len(df):,}")
st.sidebar.metric("Cities", df["City"].nunique())
st.sidebar.metric("States", df["State"].nunique())

# ==================================================================
# PART A: WHAT IS IN THE DATA?
# ==================================================================
if part.startswith("A"):
    st.header("Part A: What is in the data?")

    question(1, "What is the distribution of property prices?")
    fig, ax = new_chart()
    ax.hist(df["Price_in_Lakhs"], bins=30, color="skyblue", edgecolor="black")
    ax.set_title("Distribution of Property Prices")
    ax.set_xlabel("Price (Lakhs)")
    ax.set_ylabel("Number of Properties")
    st.pyplot(fig)
    p = df["Price_in_Lakhs"]
    st.info(f"Prices range from {p.min():.0f} to {p.max():.0f} lakhs. The mean is {p.mean():.1f} and the "
            f"median is {p.median():.1f}, so the spread is balanced with no strong skew "
            f"(skewness {p.skew():.2f}).")

    question(2, "What is the distribution of property sizes?")
    fig, ax = new_chart()
    ax.hist(df["Size_in_SqFt"], bins=30, color="lightgreen", edgecolor="black")
    ax.set_title("Distribution of Property Sizes")
    ax.set_xlabel("Size (SqFt)")
    ax.set_ylabel("Number of Properties")
    st.pyplot(fig)
    s = df["Size_in_SqFt"]
    st.info(f"Sizes range from {s.min():.0f} to {s.max():.0f} sqft, with a mean of {s.mean():,.0f} and a "
            f"median of {s.median():,.0f}. Small, mid-size and large properties are all common.")

    question(5, "Are there any outliers in price per sq ft or property size?")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["Price_per_SqFt", "Size_in_SqFt", "Price_in_Lakhs"]):
        sns.boxplot(y=df[col], ax=ax, color="orange")
        ax.set_title(f"Box Plot of {col}")
    st.pyplot(fig)
    rows = []
    for col in ["Price_per_SqFt", "Size_in_SqFt", "Price_in_Lakhs"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({"Column": col, "Outliers": n, "% of rows": round(n / len(df) * 100, 1),
                     "Upper limit": round(high)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.info("Price per sqft has outliers on the high side. They are small properties with high prices, "
            "so they are genuine values and not errors. Size and total price have none.")

    question(16, "How many properties belong to each owner type?")
    owner = df["Owner_Type"].value_counts()
    fig, ax = new_chart(6, 4)
    owner.plot(kind="bar", color="purple", ax=ax)
    ax.set_title("Number of Properties by Owner Type")
    ax.set_xlabel("Owner Type")
    ax.set_ylabel("Number of Properties")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    st.info("Share of listings: " + ", ".join(f"{k} {v / len(df) * 100:.1f}%" for k, v in owner.items()) + ".")

    question(17, "How many properties are available under each availability status?")
    status = df["Availability_Status"].value_counts()
    fig, ax = new_chart(5, 5)
    status.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_title("Properties by Availability Status")
    ax.set_ylabel("")
    st.pyplot(fig)
    st.info("Counts: " + ", ".join(f"{k} {v:,}" for k, v in status.items()) + ".")

# ==================================================================
# PART B: PRICE AND SIZE
# ==================================================================
elif part.startswith("B"):
    st.header("Part B: Price and size")

    question(4, "Is there a relationship between property size and price?")
    corr = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
    sample = df.sample(5000, random_state=1)
    fig, ax = new_chart()
    sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=0.4, ax=ax)
    ax.set_title("Property Size vs Price (sample of 5,000)")
    ax.set_xlabel("Size (SqFt)")
    ax.set_ylabel("Price (Lakhs)")
    st.pyplot(fig)
    st.info(f"The correlation is {corr:.3f}, which is {strength(corr)}. Bigger properties are not "
            f"priced higher in this data.")

    question(3, "How does price per sq ft vary by property type?")
    pps_type = df.groupby("Property_Type")["Price_per_SqFt"].agg(["mean", "median"]).round(0)
    fig, ax = new_chart(6, 4)
    pps_type["mean"].plot(kind="bar", color="teal", ax=ax)
    ax.set_title("Average Price per SqFt by Property Type")
    ax.set_xlabel("Property Type")
    ax.set_ylabel("Price per SqFt (Rs)")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    st.dataframe(pps_type)
    st.info(f"The gap between the highest and lowest property type is only "
            f"{gap_pct(pps_type['mean']):.1f}%, so property type has little effect on price per sqft.")

    question(11, "How are numeric features correlated with each other?")
    num_cols = ["Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt", "BHK", "Nearby_Schools",
                "Nearby_Hospitals", "Age_of_Property", "Infrastructure_Score",
                "Amenity_Count", "Transport_Score"]
    cm = df[num_cols].corr()
    fig, ax = new_chart(10, 8)
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Numeric Features")
    st.pyplot(fig)
    mask = np.triu(np.ones(cm.shape, dtype=bool), k=1)      # keep each pair once
    top = cm.where(mask).stack().sort_values(key=abs, ascending=False).head(6).round(2)
    top = top.reset_index()
    top.columns = ["Feature 1", "Feature 2", "Correlation"]
    st.write("Strongest pairs:")
    st.dataframe(top, hide_index=True)
    st.info("The strong pairs are built in: price per sqft is price divided by size, and the infrastructure "
            "score is made from schools, hospitals and transport. Almost every other pair is close to 0.")

# ==================================================================
# PART C: LOCATION
# ==================================================================
elif part.startswith("C"):
    st.header("Part C: Location")

    question(6, "What is the average price per sq ft by state?")
    state = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False)
    fig, ax = new_chart(12, 5)
    state.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Average Price per SqFt by State")
    ax.set_xlabel("State")
    ax.set_ylabel("Price per SqFt (Rs)")
    plt.xticks(rotation=60, ha="right")
    st.pyplot(fig)
    st.info(f"Highest: {state.index[0]} (Rs {state.iloc[0]:,.0f}). Lowest: {state.index[-1]} "
            f"(Rs {state.iloc[-1]:,.0f}). The gap across {len(state)} states is {gap_pct(state):.1f}%.")

    question(7, "What is the average property price by city?")
    city = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
    top_n = st.slider("Number of cities to show", 5, len(city), 15)
    fig, ax = new_chart(12, 5)
    city.head(top_n).plot(kind="bar", color="darkorange", ax=ax)
    ax.set_title(f"Top {top_n} Cities by Average Property Price")
    ax.set_xlabel("City")
    ax.set_ylabel("Average Price (Lakhs)")
    plt.xticks(rotation=60, ha="right")
    st.pyplot(fig)
    st.info(f"Highest: {city.index[0]} ({city.iloc[0]:.1f} lakhs). Lowest: {city.index[-1]} "
            f"({city.iloc[-1]:.1f} lakhs). The gap across {len(city)} cities is {gap_pct(city):.1f}%.")

    question(9, "How is BHK distributed across cities?")
    bhk = pd.crosstab(df["City"], df["BHK"], normalize="index") * 100
    fig, ax = new_chart(8, 12)
    sns.heatmap(bhk, annot=True, fmt=".0f", cmap="Blues", ax=ax)
    ax.set_title("BHK Distribution Across Cities (% of listings)")
    ax.set_xlabel("BHK")
    ax.set_ylabel("City")
    st.pyplot(fig)
    st.info(f"In every city each BHK makes up between {bhk.min().min():.1f}% and {bhk.max().max():.1f}% "
            f"of listings, so BHK is spread evenly.")

    question(8, "What is the median age of properties by locality?")
    age = df.groupby("Locality")["Age_of_Property"].median()
    fig, ax = new_chart()
    ax.hist(age, bins=range(int(age.min()), int(age.max()) + 2), color="gray", edgecolor="black")
    ax.set_title(f"Median Property Age by Locality ({len(age)} localities)")
    ax.set_xlabel("Median Age (years)")
    ax.set_ylabel("Number of Localities")
    st.pyplot(fig)
    st.info(f"The overall median age is {df['Age_of_Property'].median():.0f} years. Locality medians range "
            f"from {age.min():.0f} ({age.idxmin()}) to {age.max():.0f} ({age.idxmax()}). "
            f"Note: the same locality names appear in every city.")

    question(10, "What are the price trends for the top 5 most expensive localities?")
    top5 = df.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5)
    trend = (df[df["Locality"].isin(top5.index)]
             .groupby(["Year_Built", "Locality"])["Price_in_Lakhs"].mean().unstack())
    fig, ax = new_chart(11, 5)
    trend.plot(marker="o", ax=ax)
    ax.set_title("Average Price by Year Built - Top 5 Most Expensive Localities")
    ax.set_xlabel("Year Built")
    ax.set_ylabel("Average Price (Lakhs)")
    st.pyplot(fig)
    st.dataframe(top5.round(1).rename("Average price (Lakhs)"))
    st.info("The lines move up and down with no steady upward trend. Each point is based on only about "
            "15 listings, so the swings are mostly noise.")

# ==================================================================
# PART D: PROPERTY FEATURES
# ==================================================================
elif part.startswith("D"):
    st.header("Part D: Property features")

    question(14, "How does price vary by furnished status?")
    furn = df.groupby("Furnished_Status")["Price_in_Lakhs"].agg(["mean", "median"]).round(1)
    fig, ax = new_chart(7, 4.5)
    sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", ax=ax)
    ax.set_title("Price by Furnished Status")
    ax.set_xlabel("Furnished Status")
    ax.set_ylabel("Price (Lakhs)")
    st.pyplot(fig)
    st.dataframe(furn)
    st.info(f"Average prices differ by only {gap_pct(furn['mean']):.1f}% across furnished statuses, "
            f"so furnishing has little effect on price here.")

    question(15, "How does price per sq ft vary by property facing direction?")
    facing = df.groupby("Facing")["Price_per_SqFt"].mean().sort_values(ascending=False)
    fig, ax = new_chart(6, 4)
    facing.plot(kind="bar", color="brown", ax=ax)
    ax.set_title("Average Price per SqFt by Facing Direction")
    ax.set_xlabel("Facing")
    ax.set_ylabel("Price per SqFt (Rs)")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    st.info(f"Highest: {facing.index[0]} (Rs {facing.iloc[0]:,.0f}). Lowest: {facing.index[-1]} "
            f"(Rs {facing.iloc[-1]:,.0f}). The gap is {gap_pct(facing):.1f}%.")

    question(18, "Does parking space affect property price?")
    park = df.groupby("Parking_Space")["Price_in_Lakhs"].mean().round(1)
    fig, ax = new_chart(6, 4)
    park.plot(kind="bar", color="green", ax=ax)
    ax.set_title("Average Price by Parking Space")
    ax.set_xlabel("Parking Space")
    ax.set_ylabel("Average Price (Lakhs)")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    st.info("Average price: " + ", ".join(f"{k}: {v} lakhs" for k, v in park.items()) +
            f". The gap is {gap_pct(park):.1f}%, so parking has little effect.")

    question(19, "How do amenities affect price per sq ft?")
    rows = []
    for a in ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]:
        has = df["Amenities"].str.contains(a)
        rows.append({"Amenity": a, "With": df.loc[has, "Price_per_SqFt"].mean(),
                     "Without": df.loc[~has, "Price_per_SqFt"].mean()})
    amen = pd.DataFrame(rows).set_index("Amenity").round(0)
    amen["Difference"] = amen["With"] - amen["Without"]
    fig, ax = new_chart(8, 4.5)
    amen[["With", "Without"]].plot(kind="bar", ax=ax)
    ax.set_title("Price per SqFt With and Without Each Amenity")
    ax.set_xlabel("Amenity")
    ax.set_ylabel("Price per SqFt (Rs)")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    st.dataframe(amen)
    by_count = df.groupby("Amenity_Count")["Price_per_SqFt"].mean()
    st.info(f"Each amenity changes price per sqft by at most Rs {amen['Difference'].abs().max():,.0f}. "
            f"Going from {by_count.index.min()} to {by_count.index.max()} amenities changes it by "
            f"{(by_count.iloc[-1] - by_count.iloc[0]) / by_count.iloc[0] * 100:.1f}%.")

# ==================================================================
# PART E: INFRASTRUCTURE AND INVESTMENT
# ==================================================================
else:
    st.header("Part E: Infrastructure and investment")

    question(12, "How do nearby schools relate to price per sq ft?")
    c_sch = df["Nearby_Schools"].corr(df["Price_per_SqFt"])
    fig, ax = new_chart()
    df.groupby("Nearby_Schools")["Price_per_SqFt"].mean().plot(marker="o", ax=ax)
    ax.set_title("Average Price per SqFt by Number of Nearby Schools")
    ax.set_xlabel("Nearby Schools")
    ax.set_ylabel("Price per SqFt (Rs)")
    st.pyplot(fig)
    st.info(f"The correlation is {c_sch:.3f}, which is {strength(c_sch)}.")

    question(13, "How do nearby hospitals relate to price per sq ft?")
    c_hos = df["Nearby_Hospitals"].corr(df["Price_per_SqFt"])
    fig, ax = new_chart()
    df.groupby("Nearby_Hospitals")["Price_per_SqFt"].mean().plot(marker="o", color="red", ax=ax)
    ax.set_title("Average Price per SqFt by Number of Nearby Hospitals")
    ax.set_xlabel("Nearby Hospitals")
    ax.set_ylabel("Price per SqFt (Rs)")
    st.pyplot(fig)
    st.info(f"The correlation is {c_hos:.3f}, which is {strength(c_hos)}.")

    question(20, "How does public transport accessibility relate to price per sq ft or investment potential?")
    transport = df.groupby("Public_Transport_Accessibility").agg(
        Avg_Price_per_SqFt=("Price_per_SqFt", "mean"),
        Good_Investment_Pct=("Good_Investment", lambda x: x.mean() * 100)).round(1)
    transport = transport.loc[["Low", "Medium", "High"]]
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = new_chart(6, 4)
        transport["Avg_Price_per_SqFt"].plot(kind="bar", color="orange", ax=ax)
        ax.set_title("Price per SqFt by Transport Access")
        ax.set_xlabel("Public Transport Access")
        ax.set_ylabel("Price per SqFt (Rs)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    with col_b:
        fig, ax = new_chart(6, 4)
        transport["Good_Investment_Pct"].plot(kind="bar", color="green", ax=ax)
        ax.set_title("Good Investment % by Transport Access")
        ax.set_xlabel("Public Transport Access")
        ax.set_ylabel("Good Investment (%)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
    st.dataframe(transport)
    st.info(f"Price per sqft differs by only {gap_pct(transport['Avg_Price_per_SqFt']):.1f}% across transport "
            f"levels. Good Investment % is much higher for Medium and High access, but transport is one of "
            f"the rules that define Good Investment, so this gap is built in.")
