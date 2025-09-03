# =============================
# Step 1: Import Libraries
# =============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from google.colab import files

# =============================
# Step 2: Upload CSV File
# =============================
print("📂 Please upload the 'wondr_diamonds_sales.csv' file...")
uploaded = files.upload()

# Load into DataFrame
import io
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# Preview dataset
print("\n✅ Dataset Uploaded Successfully!")
print(df.head())

# =============================
# Step 3: Basic Statistics
# =============================
print("\n📊 Dataset Info:")
print(df.describe())
print("\nColumns:", df.columns.tolist())

# =============================
# Step 4: Visualization 1 – Sales by Occasion
# =============================
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Occasion", order=df['Occasion'].value_counts().index, palette="viridis")
plt.title("Sales Count by Occasion")
plt.xticks(rotation=45)
plt.show()

# =============================
# Step 5: Visualization 2 – Revenue by City
# =============================
city_revenue = df.groupby("City")["Price"].sum().sort_values(ascending=False)
plt.figure(figsize=(8,5))
city_revenue.plot(kind='bar', color="teal")
plt.title("Total Revenue by City")
plt.ylabel("Revenue (INR)")
plt.show()

# =============================
# Step 6: Customer Segmentation (Clustering)
# =============================
features = df[["Age","Price"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(features)

# Visualization 3: Clusters
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="Age", y="Price", hue="Cluster", palette="deep", s=80)
plt.title("Customer Segmentation by Age & Price")
plt.xlabel("Customer Age")
plt.ylabel("Purchase Value (INR)")
plt.show()

# =============================
# Step 7: Persona Summary with Meaning
# =============================
print("\n🧑‍🤝‍🧑 Customer Personas (Cluster Summary & Meaning):")

for cluster in df["Cluster"].unique():
    subset = df[df["Cluster"]==cluster]
    avg_age = round(subset["Age"].mean(),1)
    avg_spend = round(subset["Price"].mean(),0)
    top_city = subset["City"].mode()[0]
    top_occasion = subset["Occasion"].mode()[0]
    top_channel = subset["Channel"].mode()[0]

    # Persona Meaning Logic
    if avg_age < 30 and avg_spend < 120000:
        persona = "💎 Young Budget Buyers – Young professionals making affordable gifting or festival purchases"
    elif avg_age < 40 and avg_spend >= 120000 and avg_spend < 250000:
        persona = "💍 Mid-range Occasion Buyers – Couples/families buying for weddings & anniversaries, moderately price-sensitive"
    else:
        persona = "👑 Premium Luxury Buyers – High-spending, often older customers from metro cities buying wedding/heritage jewelry"

    # Print Persona Summary
    print(f"\n--- Persona {cluster} ---")
    print("Average Age:", avg_age)
    print("Average Spend (INR):", avg_spend)
    print("Top City:", top_city)
    print("Top Occasion:", top_occasion)
    print("Preferred Channel:", top_channel)
    print("Meaning:", persona)
