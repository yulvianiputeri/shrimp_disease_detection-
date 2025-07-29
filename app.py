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

try:
    knn = joblib.load("knn_model.pkl")
    svm = joblib.load("svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")

    X_test_eval = joblib.load("X_test.pkl")
    y_test_eval = joblib.load("y_test.pkl")

    st.sidebar.success("Models and test data loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("Model or test data files not found. Please run 'train_model.py' first.")
    st.stop() 

def count_images_per_class():
    """Menghitung jumlah gambar per kelas dalam dataset."""
    counts = {}
    base_path = "data_udang"
    for folder in ["1. Healthy", "2. BG", "3. WSSV", "4. WSSV_BG"]:
        path = os.path.join(base_path, folder)
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

def load_test_dataset_for_evaluation():
    """
    Memuat subset gambar dari dataset untuk evaluasi model.
    Menggunakan subset kecil (15 gambar per folder) untuk demo cepat.
    """
    all_images = []
    base_path = "data_udang"

    label_map_eval = {"1. Healthy": 0, "2. BG": 1, "3. WSSV": 1, "4. WSSV_BG": 1}

    for folder, label_val in label_map_eval.items():
        path = os.path.join(base_path, folder)
        if not os.path.exists(path):
            st.warning(f"Test data folder not found: {path}. Skipping.")
            continue
        
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.png','.jpeg'))][:15]
        for fname in files:
            img_path = os.path.join(path, fname)
            try:
                img = Image.open(img_path).convert('RGB').resize((128,128))
                all_images.append( (np.array(img), label_val) )
            except Exception as e:
                st.warning(f"Could not load test image {img_path}: {e}")
                continue
    return all_images


if menu == "📷 Predict":
    st.title("📷 Upload and Predict Shrimp")
    uploaded_file = st.file_uploader("Upload shrimp image...", type=["jpg","png","jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB').resize((128,128))
        img_np = np.array(img)
        
        st.image(img, caption="Input Image", use_container_width=True)
        
        # Ekstraksi fitur, scaling, dan PCA
        features = extract_features_single(img_np).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Prediksi dengan KNN dan SVM
        knn_pred = knn.predict(features_pca)[0]
        svm_pred = svm.predict(features_pca)[0]
        
        # Probabilitas prediksi
        knn_proba = knn.predict_proba(features_pca)[0]
        svm_proba = svm.predict_proba(features_pca)[0]

        st.markdown("---")
        st.markdown(f"### 🔵 KNN: {'✅ **Healthy**' if knn_pred==0 else '❌ **Diseased**'} ({100*knn_proba[0]:.1f}% Healthy, {100*knn_proba[1]:.1f}% Diseased)")
        st.markdown(f"### 🔴 SVM: {'✅ **Healthy**' if svm_pred==0 else '❌ **Diseased**'} ({100*svm_proba[0]:.1f}% Healthy, {100*svm_proba[1]:.1f}% Diseased)")

elif menu == "🖼️ Gallery":
    st.title("🖼 Shrimp Gallery")
    st.subheader("Dataset Summary")
    for folder, count in counts.items():
        st.write(f"{folder}: **{count} images**")
    
    st.markdown("---")
    st.subheader("Healthy Shrimp Examples")
    cols_healthy = st.columns(3)
    healthy_images = load_gallery_images("1. Healthy", "Healthy")
    for i, (img, label) in enumerate(healthy_images):
        with cols_healthy[i%3]:
            st.image(img, caption=f"✅ {label}", use_container_width=True)
    
    st.markdown("---")
    st.subheader("Diseased Shrimp Examples")
    cols_disease = st.columns(3)

    Disease_images = load_gallery_images("3. WSSV", "Disease") + load_gallery_images("4. WSSV_BG", "Disease")
    for i, (img, label) in enumerate(Disease_images):
        with cols_disease[i%3]:
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
        st.image(resized, caption="(b) Resized (128x128)", use_container_width=True)
        
        img_np = np.array(resized)
        
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        st.subheader("(c) Contrast Enhanced Image")
        st.image(enhanced, caption="(c) Contrast Enhanced (CLAHE)", use_container_width=True)
        
        marked = resized.copy()
        draw = ImageDraw.Draw(marked)

        draw.rectangle([(30,50), (90,100)], outline="red", width=3)
        st.subheader("(d) Marked Suspicious Area")
        st.image(marked, caption="(d) Marked suspicious area (example)", use_container_width=True)

elif menu == "📊 Model Evaluation":
    st.title("📊 Model Evaluation on Test Dataset")
    
    X = X_test_eval
    y = y_test_eval
    st.write(f"Evaluating on {len(y)} samples from the held-out test set.")

    # Prediksi dan probabilitas
    knn_preds = knn.predict(X) 
    svm_preds = svm.predict(X) 
    knn_probs = knn.predict_proba(X)[:,1] 
    svm_probs = svm.predict_proba(X)[:,1] 

    st.header("🔹 Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("KNN")
        cm_knn = confusion_matrix(y, knn_preds)
        fig1, ax1 = plt.subplots(figsize=(6,5))
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                    xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        st.pyplot(fig1)
    with col2:
        st.subheader("SVM")
        cm_svm = confusion_matrix(y, svm_preds)
        fig2, ax2 = plt.subplots(figsize=(6,5))
        sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', ax=ax2,
                    xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        st.pyplot(fig2)

    st.header("📈 ROC Curve")
    fpr_knn, tpr_knn, _ = roc_curve(y, knn_probs)
    fpr_svm, tpr_svm, _ = roc_curve(y, svm_probs)
    auc_knn = auc(fpr_knn, tpr_knn)
    auc_svm = auc(fpr_svm, tpr_svm)
    
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={auc_knn:.2f})", color='blue')
    ax.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={auc_svm:.2f})", color='red')
    ax.plot([0,1], [0,1], 'k--', label='Random Classifier') # Garis referensi
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.header("📊 Precision, Recall, F1-Score Bar Plot")

    report_knn_dict = classification_report(y, knn_preds, output_dict=True, zero_division=0)
    report_svm_dict = classification_report(y, svm_preds, output_dict=True, zero_division=0)

    report_knn = pd.DataFrame(report_knn_dict).transpose()
    report_svm = pd.DataFrame(report_svm_dict).transpose()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("KNN Metrics")

        metrics_to_plot_knn = report_knn.loc[['0', '1', 'macro avg'], ['precision', 'recall', 'f1-score']]
        fig_knn, ax_knn = plt.subplots(figsize=(8,5))
        metrics_to_plot_knn.plot(kind='bar', ax=ax_knn, cmap='viridis')
        ax_knn.set_title("KNN Classification Metrics")
        ax_knn.set_ylabel("Score")
        ax_knn.set_xticklabels(['Healthy (0)', 'Diseased (1)', 'Macro Avg'], rotation=45, ha='right')
        ax_knn.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig_knn)
    with col4:
        st.subheader("SVM Metrics")
        metrics_to_plot_svm = report_svm.loc[['0', '1', 'macro avg'], ['precision', 'recall', 'f1-score']]
        fig_svm, ax_svm = plt.subplots(figsize=(8,5))
        metrics_to_plot_svm.plot(kind='bar', ax=ax_svm, cmap='plasma')
        ax_svm.set_title("SVM Classification Metrics")
        ax_svm.set_ylabel("Score")
        ax_svm.set_xticklabels(['Healthy (0)', 'Diseased (1)', 'Macro Avg'], rotation=45, ha='right')
        ax_svm.legend(loc='lower right')
        plt.tight_layout()
        st.pyplot(fig_svm)

    st.header("📋 Classification Report Table")
    st.subheader("KNN Classification Report")
    st.dataframe(report_knn.style.format(precision=2))
    st.subheader("SVM Classification Report")
    st.dataframe(report_svm.style.format(precision=2))
