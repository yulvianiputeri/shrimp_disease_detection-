
import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
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

st.sidebar.title("🦐 Navigation")
menu = st.sidebar.radio("Choose Page:", [
    "📷 Predict",
    "🖼️ Gallery",
    "📊 Confusion Matrix",
    "📈 ROC Curve",
    "📉 Precision/Recall Bar Plot"
])

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
        st.markdown("---")
        st.markdown(f"### 🔵 KNN Prediction: {'✅ **Healthy**' if knn_pred==0 else '❌ **Sick**'}")
        st.markdown(f"### 🔴 SVM Prediction: {'✅ **Healthy**' if svm_pred==0 else '❌ **Sick**'}")

elif menu == "🖼️ Gallery":
    st.title("🖼️ Shrimp Gallery")
    st.subheader("✅ Healthy Shrimp Examples")
    healthy_images = load_gallery_images("1. Healthy", "Healthy")
    cols = st.columns(3)
    for i, (img, label) in enumerate(healthy_images):
        with cols[i%3]:
            st.image(img, caption=f"✅ {label}", use_container_width=True)
    st.markdown("---")
    st.subheader("❌ Sick Shrimp Examples")
    sick_images = load_gallery_images("3. WSSV", "Sick") + load_gallery_images("4. WSSV_BG", "Sick")
    cols = st.columns(3)
    for i, (img, label) in enumerate(sick_images):
        with cols[i%3]:
            st.image(img, caption=f"❌ {label}", use_container_width=True)

elif menu == "📊 Confusion Matrix":
    st.title("📊 Confusion Matrix KNN & SVM")
    test_data = load_test_dataset()
    X = np.array([extract_features_single(im).flatten() for im, _ in test_data])
    y = np.array([label for _, label in test_data])
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

elif menu == "📈 ROC Curve":
    st.title("📈 ROC Curve Comparison")
    test_data = load_test_dataset()
    X = np.array([extract_features_single(im).flatten() for im, _ in test_data])
    y = np.array([label for _, label in test_data])
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    knn_probs = knn.predict_proba(X_pca)[:,1]
    svm_probs = svm.predict_proba(X_pca)[:,1]

    fpr_knn, tpr_knn, _ = roc_curve(y, knn_probs)
    fpr_svm, tpr_svm, _ = roc_curve(y, svm_probs)
    auc_knn = auc(fpr_knn, tpr_knn)
    auc_svm = auc(fpr_svm, tpr_svm)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(fpr_knn, tpr_knn, label=f'KNN (AUC={auc_knn:.2f})')
    ax.plot(fpr_svm, tpr_svm, label=f'SVM (AUC={auc_svm:.2f})', color='red')
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve KNN vs SVM")
    ax.legend()
    st.pyplot(fig)

elif menu == "📉 Precision/Recall Bar Plot":
    st.title("📉 Precision, Recall, F1-Score")
    test_data = load_test_dataset()
    X = np.array([extract_features_single(im).flatten() for im, _ in test_data])
    y = np.array([label for _, label in test_data])
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    knn_preds = knn.predict(X_pca)
    svm_preds = svm.predict(X_pca)
    
    report_knn = pd.DataFrame(classification_report(y, knn_preds, output_dict=True)).transpose()
    report_svm = pd.DataFrame(classification_report(y, svm_preds, output_dict=True)).transpose()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("KNN")
        fig_knn, ax_knn = plt.subplots(figsize=(8,4))
        report_knn.iloc[:-3, :3].plot(kind='bar', ax=ax_knn)
        ax_knn.set_title("KNN: Precision, Recall, F1-Score per Class")
        st.pyplot(fig_knn)
    with col2:
        st.subheader("SVM")
        fig_svm, ax_svm = plt.subplots(figsize=(8,4))
        report_svm.iloc[:-3, :3].plot(kind='bar', ax=ax_svm, color=['red','orange','purple'])
        ax_svm.set_title("SVM: Precision, Recall, F1-Score per Class")
        st.pyplot(fig_svm)
