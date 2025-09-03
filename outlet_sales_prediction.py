# =============================
# Step 1: Install Libraries
# =============================
!pip install prophet pandas matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from google.colab import files
import io

# =============================
# Step 2: Upload Dataset
# =============================
print("📂 Please upload 'wondr_diamonds_outlet_sales.csv' file...")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[filename]))
df["Month"] = pd.to_datetime(df["Month"])

print("\n✅ Dataset Loaded Successfully")
print(df.head())

# =============================
# Step 3: Total Sales Trend (All Outlets Combined)
# =============================
df_total = df.groupby("Month")["Sales_Lakhs"].sum().reset_index()

plt.figure(figsize=(12,6))
plt.plot(df_total["Month"], df_total["Sales_Lakhs"], color="teal", linewidth=2)
plt.title("📈 Overall Diamond Jewellery Sales (All Outlets)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Sales (Lakhs INR)")
plt.grid(True, alpha=0.3)
plt.show()

# =============================
# Step 4: Top 5 Outlets Performance
# =============================
top_outlets = df.groupby("Outlet")["Sales_Lakhs"].sum().sort_values(ascending=False).head(5).index
df_top = df[df["Outlet"].isin(top_outlets)]

plt.figure(figsize=(12,6))
sns.lineplot(data=df_top, x="Month", y="Sales_Lakhs", hue="Outlet", linewidth=2)
plt.title("🏬 Top 5 Performing Outlets (2020-2025)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Sales (Lakhs INR)")
plt.legend(title="Outlet")
plt.show()

# =============================
# Step 5: Forecast Example for One Outlet
# =============================
outlet_id = "Outlet_01"  # Change if needed
df_outlet = df[df["Outlet"]==outlet_id][["Month","Sales_Lakhs"]].rename(columns={"Month":"ds","Sales_Lakhs":"y"})

model = Prophet()
model.fit(df_outlet)

future = model.make_future_dataframe(periods=6, freq="MS")
forecast = model.predict(future)

# Visualization: Forecast
plt.figure(figsize=(12,6))
plt.plot(df_outlet["ds"], df_outlet["y"], label="Historical Sales", color="blue")
plt.plot(forecast["ds"], forecast["yhat"], label="Predicted Sales", color="red", linestyle="--")
plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                 color="pink", alpha=0.3, label="Prediction Range")
plt.title(f"🔮 Sales Forecast for {outlet_id} (Next 6 Months)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Sales (Lakhs INR)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# =============================
# Step 6: Easy Explanation
# =============================
print("\n📊 Easy Insights:")
print("1. The first chart shows how total jewellery sales grew across all outlets.")
print("2. The second chart highlights the Top 5 outlets – useful for benchmarking.")
print(f"3. The forecast chart shows how {outlet_id} is expected to perform in the next 6 months.")
print("   • Blue line = Actual sales history")
print("   • Red dashed line = Predicted sales")
print("   • Pink shaded area = Safe range (high/low possible values)")
print("👉 Managers can use this to plan inventory, promotions, and staffing for peak seasons.")
