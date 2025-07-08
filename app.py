
import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from utils import extract_features_single, load_gallery_images

knn = joblib.load("knn_model.pkl")
svm = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

st.sidebar.title("🦐 Navigation")
menu = st.sidebar.radio("Choose Page:", ["📷 Predict", "🖼️ Gallery", "📊 Performance"])

if menu == "📷 Predict":
    st.title("📷 Upload and Predict Shrimp")
    uploaded_file = st.file_uploader("Upload shrimp image...", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB').resize((128,128))
        img_np = np.array(img)
        st.image(img, caption="Input Image", use_container_width=True)
        features = extract_features_single(img_np).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        knn_pred = knn.predict(features_pca)[0]
        svm_pred = svm.predict(features_pca)[0]
        st.markdown("---")
        st.markdown(f"### 🔵 KNN Prediction: {'✅ **Healthy**' if knn_pred==0 else '❌ **Disease**'}")
        st.markdown(f"### 🔴 SVM Prediction: {'✅ **Healthy**' if svm_pred==0 else '❌ **Disease**'}")

elif menu == "🖼️ Gallery":
    st.title("🖼️ Shrimp Gallery")
    st.subheader("✅ Healthy Shrimp Examples")
    healthy_images = load_gallery_images("1. Healthy", "Healthy")
    cols = st.columns(3)
    for i, (img, label) in enumerate(healthy_images):
        with cols[i%3]:
            st.image(img, caption=f"✅ {label}", use_container_width=True)
    st.markdown("---")
    st.subheader("❌ Disease Shrimp Examples")
    Disease_images = load_gallery_images("3. WSSV", "Disease") + load_gallery_images("4. WSSV_BG", "Disease")
    cols = st.columns(3)
    for i, (img, label) in enumerate(Disease_images):
        with cols[i%3]:
            st.image(img, caption=f"❌ {label}", use_container_width=True)

elif menu == "📊 Performance":
    st.title("📊 Model Performance Analysis")
    all_images = []
    for folder, label_val in [("1. Healthy", 0), ("2. BG", 1), ("3. WSSV", 1), ("4. WSSV_BG", 1)]:
        path = os.path.join("data_udang", folder)
        for fname in os.listdir(path)[:10]:
            img_path = os.path.join(path, fname)
            try:
                img = Image.open(img_path).convert('RGB').resize((128,128))
                all_images.append( (np.array(img), label_val) )
            except:
                continue
    X = np.array([extract_features_single(im).flatten() for im, _ in all_images])
    y = np.array([label for _, label in all_images])
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    knn_preds = knn.predict(X_pca)
    svm_preds = svm.predict(X_pca)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("KNN Confusion Matrix")
        cm_knn = confusion_matrix(y, knn_preds)
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax1)
        st.pyplot(fig1)
        st.code(classification_report(y, knn_preds))

    with col2:
        st.subheader("SVM Confusion Matrix")
        cm_svm = confusion_matrix(y, svm_preds)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', ax=ax2)
        st.pyplot(fig2)
        st.code(classification_report(y, svm_preds))
