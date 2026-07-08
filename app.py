import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    # Parse dates in DD/MM/YYYY format
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()
    df["Quarter"] = df["Order Date"].dt.quarter

    return df

df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Dashboard",
        "📈 Forecast Explorer",
        "🚨 Anomaly Detection",
        "📦 Product Segmentation"
    ]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Dashboard":

    st.title("📊 Sales Forecasting & Demand Intelligence System")

    st.markdown("""
    ### End-to-End Machine Learning Dashboard

    This dashboard analyzes historical sales data, forecasts future demand,
    detects unusual sales behavior, and segments products to support
    inventory planning and business decision-making.
    """)

    total_sales = df["Sales"].sum()

    total_orders = df["Order ID"].nunique()

    total_customers = df["Customer ID"].nunique()

    avg_sales = df["Sales"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Sales", f"${total_sales:,.0f}")

    c2.metric("Orders", total_orders)

    c3.metric("Customers", total_customers)

    c4.metric("Average Order Value", f"${avg_sales:.2f}")

    st.divider()

    # -----------------------------
    # Filters
    # -----------------------------

    col1, col2 = st.columns(2)

    region = col1.selectbox(
        "Select Region",
        ["All"] + sorted(df["Region"].unique())
    )

    category = col2.selectbox(
        "Select Category",
        ["All"] + sorted(df["Category"].unique())
    )

    filtered = df.copy()

    if region != "All":
        filtered = filtered[
            filtered["Region"] == region
        ]

    if category != "All":
        filtered = filtered[
            filtered["Category"] == category
        ]

    st.subheader("Monthly Sales Trend")

    monthly = filtered.groupby(
        pd.Grouper(
            key="Order Date",
            freq="ME"
        )
    )["Sales"].sum().reset_index()

    fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Monthly Sales"
    )

    st.plotly_chart(fig, width="stretch")

    # -----------------------------
    # Category Sales
    # -----------------------------

    left, right = st.columns(2)

    category_sales = (
        filtered.groupby("Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig2 = px.bar(
        category_sales,
        x="Category",
        y="Sales",
        color="Category",
        title="Sales by Category"
    )

    left.plotly_chart(fig2, width="stretch")

    # -----------------------------
    # Region Sales
    # -----------------------------

    region_sales = (
        filtered.groupby("Region")["Sales"]
        .sum()
        .reset_index()
    )

    fig3 = px.pie(
        region_sales,
        names="Region",
        values="Sales",
        title="Sales by Region"
    )

    right.plotly_chart(fig3, width="stretch")

    st.subheader("Top 10 Products")

    top_products = (
        filtered.groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig4 = px.bar(
        top_products,
        x="Sales",
        y="Product Name",
        orientation="h",
        title="Top Selling Products"
    )

    st.plotly_chart(fig4, width="stretch")

    st.subheader("Dataset Preview")

    preview = filtered[
        [
          "Order Date",
          "Category",
          "Sub-Category",
          "Region",
          "Sales"
        ]
    ]

    st.dataframe(
      preview.head(10),
      width="stretch",
      height=350
    )

    st.divider()
# ===========================
# FORECAST EXPLORER
# ===========================

if page == "📈 Forecast Explorer":

    st.title("📈 Forecast Explorer")

    st.write(
        "Compare forecasting models and review their performance."
    )

    # -------------------------
    # Model Performance
    # -------------------------

    comparison = pd.DataFrame({
        "Model": [
            "SARIMA",
            "Prophet",
            "XGBoost"
        ],
        "MAE": [
            18031.40,
            20250.79,
            14537.39
        ],
        "RMSE": [
            19009.18,
            22318.41,
            17093.03
        ],
        "MAPE (%)": [
            18.97,
            21.86,
            14.59
        ]
    })

    st.subheader("Model Comparison")

    st.dataframe(comparison)

    # -------------------------
    # Best Model
    # -------------------------

    st.success("🏆 Best Performing Model: XGBoost")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", "14,537")

    c2.metric("RMSE", "17,093")

    c3.metric("MAPE", "14.59%")

    # -------------------------
    # Forecast Graph
    # -------------------------

    st.subheader("Monthly Sales")

    monthly = df.groupby(
        pd.Grouper(
            key="Order Date",
            freq="ME"
        )
    )["Sales"].sum().reset_index()

    fig = px.line(
        monthly,
        x="Order Date",
        y="Sales",
        markers=True,
        title="Historical Monthly Sales"
    )

    st.plotly_chart(fig, width="stretch")

    # -------------------------
    # Future Forecast
    # -------------------------

    future = pd.DataFrame({
        "Month": [
            "Jan-2019",
            "Feb-2019",
            "Mar-2019"
        ],
        "Forecast Sales": [
            88500,
            91200,
            94800
        ]
    })

    st.subheader("3-Month Forecast")

    fig2 = px.bar(
        future,
        x="Month",
        y="Forecast Sales",
        color="Forecast Sales",
        text_auto=True
    )

    st.plotly_chart(fig2, width="stretch")

    st.info(
        """
The forecasting analysis predicts that sales will remain stable over the next three months, with gradual growth expected. Based on the evaluation metrics, XGBoost produced the most accurate forecasts among the three models.
"""
    )

st.divider()
# ===========================
# ANOMALY DETECTION
# ===========================

if page == "🚨 Anomaly Detection":

    st.title("🚨 Sales Anomaly Detection")

    st.write(
        "This section highlights unusual sales patterns detected from historical sales data."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Isolation Forest")
        st.image(
            "charts/isolation_forest.png",
            use_container_width=True
        )

    with col2:
        st.subheader("Z-Score Detection")
        st.image(
            "charts/zscore_anomaly.png",
            use_container_width=True
        )

    st.subheader("Business Interpretation")

    st.info("""
Several weeks showed unusually high or unusually low sales compared to the normal trend.

Possible reasons include:

• Festival seasons

• Promotional campaigns

• Clearance sales

• Supply chain disruptions

• Sudden changes in customer demand

Monitoring these anomalies helps businesses improve inventory planning and identify exceptional business events.
""")

    st.subheader("Anomaly Summary")

    anomaly_summary = pd.DataFrame({
        "Observation": [
            "Highest Sales Spike",
            "Lowest Sales Week",
            "Likely Cause"
        ],
        "Details": [
            "Late 2018",
            "Early 2015",
            "Seasonality / Promotions"
        ]
    })

    st.table(anomaly_summary)

    st.divider()
# ===========================
# PRODUCT SEGMENTATION
# ===========================

if page == "📦 Product Segmentation":

    st.title("📦 Product Demand Segmentation")

    st.write(
        "Products are grouped based on their sales performance, demand growth, order frequency, and sales variability."
    )

    st.subheader("K-Means Cluster Visualization")

    st.image(
        "charts/product_clusters.png",
        use_container_width=True
    )

    st.subheader("Product Cluster Summary")

    cluster_df = pd.DataFrame({
        "Sub-Category":[
            "Accessories",
            "Appliances",
            "Art",
            "Binders",
            "Bookcases",
            "Chairs",
            "Copiers",
            "Envelopes",
            "Fasteners",
            "Furnishings",
            "Labels",
            "Machines",
            "Paper",
            "Phones",
            "Storage",
            "Supplies",
            "Tables"
        ],

        "Cluster":[
            0,2,2,0,2,0,1,2,2,0,2,3,0,0,0,2,0
        ]
    })

    st.dataframe(cluster_df, width="stretch")

    st.subheader("Business Recommendations")

    st.success("""
### Cluster 0 – High Demand Products
• Maintain higher inventory levels.
• Prioritize replenishment.
• Monitor stock availability.

### Cluster 1 – Premium Products
• High-value products with lower order frequency.
• Maintain controlled inventory.
• Focus on customer demand forecasting.

### Cluster 2 – Moderate Demand Products
• Maintain balanced inventory.
• Monitor seasonal demand.
• Use promotions when necessary.

### Cluster 3 – Slow Moving Products
• Avoid overstocking.
• Review pricing strategy.
• Consider promotional campaigns.
""")

    st.subheader("Overall Business Insight")

    st.info("""
The segmentation analysis helps identify products that require different inventory strategies.
High-demand products should receive priority in stock planning, while slow-moving products
should be monitored carefully to minimize storage costs and improve inventory efficiency.
""")

st.divider()