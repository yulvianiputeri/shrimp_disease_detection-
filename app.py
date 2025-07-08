
import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from utils import extract_features_single, load_gallery_images

knn = joblib.load("knn_model.pkl")
svm = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

def count_images_per_class():
    counts = {}
    for folder in ["1. Healthy", "2. BG", "3. WSSV", "4. WSSV_BG"]:
        path = os.path.join("data_udang", folder)
        try:
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
            counts[folder] = len(files)
        except FileNotFoundError:
            counts[folder] = 0
    return counts

counts = count_images_per_class()
st.sidebar.title("🦐 Navigation")
menu = st.sidebar.radio("Choose Page:", ["📷 Predict", "🖼️ Gallery", "🖼 Pipeline", "📊 Model Evaluation"])
st.sidebar.markdown("## 🗂 Data Summary")
for folder, count in counts.items():
    st.sidebar.write(f"{folder}: **{count} images**")

def load_test_dataset():
    all_images = []
    for folder, label_val in [("1. Healthy", 0), ("2. BG", 1), ("3. WSSV", 1), ("4. WSSV_BG", 1)]:
        path = os.path.join("data_udang", folder)
        for fname in os.listdir(path)[:15]:
            img_path = os.path.join(path, fname)
            try:
                img = Image.open(img_path).convert('RGB').resize((128,128))
                all_images.append( (np.array(img), label_val) )
            except:
                continue
    return all_images

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
        knn_proba = knn.predict_proba(features_pca)[0][1]
        svm_proba = svm.predict_proba(features_pca)[0][1]
        st.markdown("---")
        st.markdown(f"### 🔵 KNN: {'✅ **Healthy**' if knn_pred==0 else '❌ **Diseased**'} ({100*(1-knn_proba):.1f}% Healthy)")
        st.markdown(f"### 🔴 SVM: {'✅ **Healthy**' if svm_pred==0 else '❌ **Diseased**'} ({100*(1-svm_proba):.1f}% Healthy)")

elif menu == "🖼️ Gallery":
    st.title("🖼 Shrimp Gallery")
    st.subheader("Dataset Summary")
    for folder, count in counts.items():
        st.write(f"{folder}: **{count} images**")
    st.markdown("---")
    cols = st.columns(3)
    healthy_images = load_gallery_images("1. Healthy", "Healthy")
    for i, (img, label) in enumerate(healthy_images):
        with cols[i%3]:
            st.image(img, caption=f"✅ {label}", use_container_width=True)
    st.markdown("---")
    Disease_images = load_gallery_images("3. WSSV", "Disease") + load_gallery_images("4. WSSV_BG", "Disease")
    for i, (img, label) in enumerate(Disease_images):
        with cols[i%3]:
            st.image(img, caption=f"❌ {label}", use_container_width=True)

elif menu == "🖼 Pipeline":
    st.title("🖼 Image Processing Pipeline")
    uploaded_file = st.file_uploader("Upload shrimp image for pipeline demo...", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.subheader("(a) Original Image")
        st.image(img, caption="(a) Original", use_container_width=True)
        resized = img.resize((128,128))
        st.subheader("(b) Resized Image")
        st.image(resized, caption="(b) Resized", use_container_width=True)
        img_np = np.array(resized)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        st.subheader("(c) Contrast Enhanced")
        st.image(enhanced, caption="(c) Enhanced", use_container_width=True)
        marked = resized.copy()
        draw = ImageDraw.Draw(marked)
        draw.rectangle([(30,50), (90,100)], outline="red", width=3)
        st.subheader("(d) Marked Infected Area")
        st.image(marked, caption="(d) Marked suspicious area", use_container_width=True)

elif menu == "📊 Model Evaluation":
    st.title("📊 Model Evaluation on Test Dataset")
    test_data = load_test_dataset()
    X = np.array([extract_features_single(im) for im, _ in test_data])
    y = np.array([label for _, label in test_data])
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    knn_preds = knn.predict(X_pca)
    svm_preds = svm.predict(X_pca)
    knn_probs = knn.predict_proba(X_pca)[:,1]
    svm_probs = svm.predict_proba(X_pca)[:,1]

    st.header("🔹 Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("KNN")
        cm_knn = confusion_matrix(y, knn_preds)
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.subheader("SVM")
        cm_svm = confusion_matrix(y, svm_preds)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', ax=ax2)
        st.pyplot(fig2)

    st.header("📈 ROC Curve")
    fpr_knn, tpr_knn, _ = roc_curve(y, knn_probs)
    fpr_svm, tpr_svm, _ = roc_curve(y, svm_probs)
    auc_knn = auc(fpr_knn, tpr_knn)
    auc_svm = auc(fpr_svm, tpr_svm)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={auc_knn:.2f})")
    ax.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={auc_svm:.2f})", color='red')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    st.header("📊 Precision, Recall, F1-Score Bar Plot")
    report_knn = pd.DataFrame(classification_report(y, knn_preds, output_dict=True)).transpose()
    report_svm = pd.DataFrame(classification_report(y, svm_preds, output_dict=True)).transpose()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("KNN")
        fig_knn, ax_knn = plt.subplots(figsize=(8,4))
        report_knn.iloc[:-3, :3].plot(kind='bar', ax=ax_knn)
        ax_knn.set_title("KNN Metrics")
        st.pyplot(fig_knn)
    with col4:
        st.subheader("SVM")
        fig_svm, ax_svm = plt.subplots(figsize=(8,4))
        report_svm.iloc[:-3, :3].plot(kind='bar', ax=ax_svm, color=['red','orange','purple'])
        ax_svm.set_title("SVM Metrics")
        st.pyplot(fig_svm)

    st.header("📋 Classification Report Table")
    st.subheader("KNN Classification Report")
    st.dataframe(report_knn.style.format(precision=2))
    st.subheader("SVM Classification Report")
    st.dataframe(report_svm.style.format(precision=2))
