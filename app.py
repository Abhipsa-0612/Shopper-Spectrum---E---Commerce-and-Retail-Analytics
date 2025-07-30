import streamlit as st
import pandas as pd
import joblib
import gdown
import os

st.title("🛒 Shopper Spectrum - Customer Insights & Recommendations")

# Step 1: Load item_similarity.pkl from Google Drive
item_file = "item_similarity.pkl"

if not os.path.exists(item_file):
  # Replace with your actual Google Drive sharable link (file > share > anyone with link)
    url = 'https://drive.google.com/file/d/1u8ZsGFasaSnSUiXNIZ_vum7P-rIoy6lV/view?usp=sharing'  # <-- replace YOUR_FILE_ID
    gdown.download(url, item_file, quiet=False)

# Step 2: Load Models
kmeans = joblib.load("rfm_kmeans_model.pkl")
scaler = joblib.load("rfm_scaler.pkl")
item_similarity = pd.read_pickle(item_file)

# Step 3: Interface Tabs
tab1, tab2 = st.tabs(["Product Recommender", "Customer Segment Predictor"])

# Product Recommendation Tab
with tab1:
    st.header("Product Recommendation System")
    product_code = st.text_input("Enter Product Stock Code (e.g., 85123A)")

    if st.button("Get Recommendations"):
        if product_code in item_similarity.columns:
            similar = item_similarity[product_code].sort_values(ascending=False)[1:6]
            st.write("Top 5 similar products:")
            for item in similar.index:
                st.markdown(f"- {item}")
        else:
            st.warning("Product not found!")

# Customer Segmentation Tab
with tab2:
    st.header("Customer Segmentation")

    rec = st.number_input("Recency (in days)", min_value=0)
    freq = st.number_input("Frequency (number of purchases)", min_value=0)
    mon = st.number_input("Monetary (total amount spent)", min_value=0.0)

    if st.button("Predict Segment"):
        user_data = scaler.transform([[rec, freq, mon]])
        cluster = kmeans.predict(user_data)[0]

        # Basic logic (you can enhance this based on your data analysis)
        if cluster == 0:
            segment = "High-Value Customer"
        elif cluster == 1:
            segment = "Regular Customer"
        elif cluster == 2:
            segment = "At-Risk Customer"
        else:
            segment = "Occasional Buyer"

        st.success(f" Predicted Segment: {segment}")
