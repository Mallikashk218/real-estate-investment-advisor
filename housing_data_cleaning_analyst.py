#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#REAL ESTATE INVESTEMENT ADVISOR#
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#PROJECT SUMMARY:
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#The goal of this project is to build a data analyst application that helps potential investors make real estate decisions. 
# The analysis uses 250,000 property listings across 42 cities and 20 states in India, and focuses on how prices, locations, features and infrastructure relate to investment quality.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Key steps:

#Data Collection and Cleaning: The dataset has 250,000 rows and 23 columns. No missing values or duplicate rows were found. The Price_per_SqFt column was recalculated in rupees because
#  the original was rounded to 0.00 for many rows. In 116,304 rows (46%), the floor number was higher than the total floors, so these were flagged and corrected.

#Feature Engineering: New features were created, including price per sqft, School Density Score, Infrastructure Score (schools, hospitals and transport), amenity count, price compared 
# with the city median, and age group. Categorical columns were encoded and numeric columns were scaled. A binary “Good Investment” label was created: a property qualifies if it meets 
# at least 3 of 4 rules (price per sqft at or below the city median, medium or high transport access, schools plus hospitals of at least 11, and 3+ amenities or security). About 52% of
#  properties are Good Investments.

#Data Visualization: Histograms, box plots, bar charts, line charts, scatter plots and heatmaps were used to answer 20 questions on price, size, location, features and infrastructure.

#Application: A Streamlit app lets users enter property details, filter listings by city, area, price and BHK, see whether a property is a Good Investment, and view an estimated price 
# after 5 years under low, base and high growth scenarios.
 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Insights: The analysis revealed several key insights:

#Prices range from 10 to 500 lakhs and sizes from 500 to 5,000 sqft, both spread evenly across their ranges.
#Average price differs by only about 3% across cities (250.8 to 258.5 lakhs), and price per sqft by under 1% across property types.
#Size and price are not related (correlation close to 0), and price per sqft has outliers (7.9% of rows) that are genuine small, high-priced properties.
#Furnishing, facing direction, parking, amenities, schools, hospitals and transport show almost no effect on price per sqft.
#BHK, owner type and availability status are spread almost evenly across the data.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Recommendations: Judge a property by its price per sqft compared with the city median, then check transport access, nearby schools and hospitals, and amenities or security. Use the 5-year 
# estimate as a scenario range, not a guaranteed forecast, because the dataset has no past prices to measure real growth.

#This analysis gives investors a clear, rule-based way to compare properties, and the Streamlit application makes it easy to screen listings and test growth assumptions.

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Problem Statement
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#This dataset contains 250,000 residential property listings across 42 cities and 20 states in India, covering apartments, independent houses and villas. Analysing it helps us understand:

#Property Prices and Location: How prices, sizes and price per sqft vary by city, state and property type, and which features (such as BHK, amenities, furnishing and age) are linked to higher prices.

#Infrastructure and Investment Quality: Whether nearby schools, hospitals and public transport relate to price, 
# and which properties qualify as a Good Investment based on clear rules such as fair price per sqft, transport access, nearby facilities and amenities.

#Future Price Estimate: What a property may cost after 5 years, using its current price and an adjustable yearly growth rate, shown under low, base and high scenarios.

#Objective: To help potential investors make real estate decisions by classifying each property as a Good Investment or not, estimating its price after 5 years, 
# and presenting the results with clear charts and recommendations in an interactive Streamlit application, with MLflow used to track and compare the rule settings and growth assumptions tried.

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
India Housing Prices - Data Cleaning for Data Analysts (no machine learning)
Only pandas and numpy are needed.
Run in VS Code:  python housing_data_cleaning_analyst.py
"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
DATA_PATH = "india_housing_prices.csv"
OUTPUT_PATH = "cleaned_housing_data.csv"
CURRENT_YEAR = 2025

# ------------------------------------------------------------------
# STEP 1: LOAD THE DATA
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Rows and columns:", df.shape)
print(df.dtypes)

# ------------------------------------------------------------------
# STEP 2: MISSING VALUES
# ------------------------------------------------------------------
missing = df.isnull().sum()
print("\nMissing values:")
print(missing[missing > 0] if missing.sum() > 0 else "None found")

# Fill any gaps (numbers -> median, text -> most common value)
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(exclude="number").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ------------------------------------------------------------------
# STEP 3: DUPLICATES (ID is always unique, so we ignore it)
# ------------------------------------------------------------------
print( df.duplicated().sum())
df = df.drop_duplicates()

# ------------------------------------------------------------------
# STEP 4: FIX INCONSISTENT VALUES
# ------------------------------------------------------------------
# 4a. Remove extra spaces in text columns
for col in df.select_dtypes(exclude="number").columns:
    df[col] = df[col].astype(str).str.strip()

# 4b. Recalculate Price per SqFt in rupees
#     (the original column is in lakhs and rounded to 2 decimals, so it shows 0.00)
df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]).round(2)

# 4c. Floor_No cannot be higher than Total_Floors
wrong_floor = df["Floor_No"] > df["Total_Floors"]
print("Rows with Floor_No > Total_Floors:", wrong_floor.sum())
df["Floor_Data_Fixed"] = wrong_floor.astype(int)       # flag so you can filter later
df["Total_Floors"] = df[["Total_Floors", "Floor_No"]].max(axis=1)

# 4d. Make sure Age matches Year_Built
df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]

# ------------------------------------------------------------------
# STEP 5: OUTLIER CHECK (report only, nothing is removed)
# ------------------------------------------------------------------
print("\nOutlier check (IQR rule):")
for col in ["Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"  {col}: {n} outliers")

# ------------------------------------------------------------------
# STEP 6: NEW FEATURES
# ------------------------------------------------------------------
# Amenities
df["Amenity_Count"] = df["Amenities"].str.split(", ").str.len()

# Public transport: Low < Medium < High
df["Transport_Score"] = df["Public_Transport_Accessibility"].map(
    {"Low": 1, "Medium": 2, "High": 3})

# School Density Score (0 = fewest schools, 1 = most schools)
df["School_Density_Score"] = (df["Nearby_Schools"] - df["Nearby_Schools"].min()) / (
    df["Nearby_Schools"].max() - df["Nearby_Schools"].min())

# Infrastructure Score = schools + hospitals + transport (0 to 1)
df["Infrastructure_Score"] = ((df["Nearby_Schools"] / 10
                               + df["Nearby_Hospitals"] / 10
                               + df["Transport_Score"] / 3) / 3).round(3)

# Price compared with the city's usual price per sqft (1.0 = same as city median)
city_median = df.groupby("City")["Price_per_SqFt"].transform("median")
df["Price_vs_City_Median"] = (df["Price_per_SqFt"] / city_median).round(2)

# Position in the building (0 = ground floor, 1 = top floor)
df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"]).round(2)

# Age group for easy charts
df["Age_Group"] = pd.cut(df["Age_of_Property"], bins=[0, 10, 20, 30, 100],
                         labels=["0-10 yrs", "11-20 yrs", "21-30 yrs", "30+ yrs"])

# ------------------------------------------------------------------
# STEP 7: ENCODE CATEGORICAL COLUMNS (text -> numbers)
# ------------------------------------------------------------------
# Yes / No -> 1 / 0
df["Parking_Space_Code"] = df["Parking_Space"].map({"Yes": 1, "No": 0})
df["Security_Code"] = df["Security"].map({"Yes": 1, "No": 0})

# Ordered categories -> 0, 1, 2
df["Furnished_Code"] = df["Furnished_Status"].map(
    {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2})

# Location columns -> number codes (State, City) and number of listings (Locality in that city)
df["State_Code"] = df["State"].astype("category").cat.codes
df["City_Code"] = df["City"].astype("category").cat.codes
df["Locality_Listings"] = df.groupby(["City", "Locality"])["ID"].transform("count")

# Property_Type -> one column per type (1 / 0)
df = pd.concat([df, pd.get_dummies(df["Property_Type"], prefix="Type", dtype=int)], axis=1)

# ------------------------------------------------------------------
# STEP 8: SCALE NUMERIC COLUMNS (Min-Max: values become 0 to 1)
# New columns end with _scaled; the original columns are kept.
# ------------------------------------------------------------------
for col in ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
            "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals"]:
    df[col + "_scaled"] = ((df[col] - df[col].min()) / (df[col].max() - df[col].min())).round(4)

# ------------------------------------------------------------------
# STEP 9: GOOD INVESTMENT LABEL (domain rules)
# The data has no past prices, so an appreciation rate cannot be calculated.
# A property gets 1 point for each rule it meets, and is a Good Investment
# when it meets at least 3 of the 4 rules. Change the numbers to suit you.
# ------------------------------------------------------------------
rule1 = df["Price_vs_City_Median"] <= 1.0                       # priced at or below city median
rule2 = df["Transport_Score"] >= 2                              # medium or high transport
rule3 = (df["Nearby_Schools"] + df["Nearby_Hospitals"]) >= 11   # enough schools + hospitals
rule4 = (df["Amenity_Count"] >= 3) | (df["Security_Code"] == 1) # good amenities or security

df["Investment_Score"] = rule1.astype(int) + rule2.astype(int) + rule3.astype(int) + rule4.astype(int)
df["Good_Investment"] = (df["Investment_Score"] >= 3).astype(int)

print("\nGood Investment split:")
print(df["Good_Investment"].value_counts())
print((df["Good_Investment"].value_counts(normalize=True) * 100).round(1))

# ------------------------------------------------------------------
# STEP 10: SAVE
# ------------------------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)
print("\nSaved:", OUTPUT_PATH, "| rows and columns:", df.shape)

# ------------------------------------------------------------------
# STEP 11: QUICK ANALYSIS (examples of what you can do next)
# ------------------------------------------------------------------
print("\nGood Investment % by Property Type:")
print((df.groupby("Property_Type")["Good_Investment"].mean() * 100).round(1))

print("\nTop 5 cities by Good Investment %:")
print((df.groupby("City")["Good_Investment"].mean() * 100).round(1).sort_values(ascending=False).head(5))

print("\nAverage price per sqft by Property Type:")
print(df.groupby("Property_Type")["Price_per_SqFt"].mean().round(0))
