import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import load_css, extract_features_single, load_dataset_features

# Page config
st.set_page_config(page_title="🦐 Shrimp Disease Detection", page_icon="🦐", layout="wide")
load_css()

# Load models
@st.cache_resource
def load_models():
    return joblib.load("knn_model.pkl"), joblib.load("svm_model.pkl"), joblib.load("scaler.pkl"), joblib.load("pca.pkl")

knn, svm, scaler, pca = load_models()

# Sidebar
st.sidebar.title("🦐 Navigation")
menu = st.sidebar.selectbox("Choose Page:", ["🔍 Detection", "🖼️ Gallery", "📊 Performance"])

# Detection page (sama seperti sebelumnya)
if menu == "🔍 Detection":
    st.markdown('<h1 class="main-header">🦐 Shrimp Disease Detection</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📤 Upload Shrimp Image", type=["jpg","png","jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            img = Image.open(uploaded_file).convert('RGB').resize((128,128))
            st.image(img, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            img_np = np.array(img)
            features = extract_features_single(img_np).reshape(1, -1)
            features_scaled = scaler.transform(features)
            features_pca = pca.transform(features_scaled)
            
            knn_pred = knn.predict(features_pca)[0]
            svm_pred = svm.predict(features_pca)[0]
            
            knn_proba = max(knn.predict_proba(features_pca)[0]) * 100
            svm_proba = max(svm.predict_proba(features_pca)[0]) * 100
            
            # Results
            knn_status = "Healthy" if knn_pred == 0 else "Diseased"
            svm_status = "Healthy" if svm_pred == 0 else "Diseased"
            
            card_class = "healthy-card" if knn_pred == 0 else "diseased-card"
            st.markdown(f'<div class="{card_class}"><h3>🔵 KNN: {knn_status}</h3><p>Confidence: {knn_proba:.1f}%</p></div>', unsafe_allow_html=True)
            
            card_class = "healthy-card" if svm_pred == 0 else "diseased-card"
            st.markdown(f'<div class="{card_class}"><h3>🔴 SVM: {svm_status}</h3><p>Confidence: {svm_proba:.1f}%</p></div>', unsafe_allow_html=True)

# Gallery page (sama seperti sebelumnya)
elif menu == "🖼️ Gallery":
    st.markdown('<h1 class="main-header">🖼️ Shrimp Gallery</h1>', unsafe_allow_html=True)
    
    df = pd.read_csv("shrimp_dataset.csv")
    
    # Split gallery
    st.subheader("✅ Contoh Udang Sehat")
    healthy_df = df[df['label'] == 'Healthy'].sample(6)
    cols = st.columns(3)
    for i, (_, row) in enumerate(healthy_df.iterrows()):
        img = Image.open(row['image_path']).convert('RGB').resize((128,128))
        with cols[i%3]:
            st.image(img, caption="Sehat", use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("❌ Contoh Udang Sakit")
    diseased_df = df[df['label'] != 'Healthy'].sample(6)
    cols = st.columns(3)
    for i, (_, row) in enumerate(diseased_df.iterrows()):
        img = Image.open(row['image_path']).convert('RGB').resize((128,128))
        with cols[i%3]:
            st.image(img, caption="Sakit", use_container_width=True)

# Performance page - UPDATED dengan visualisasi lengkap
elif menu == "📊 Performance":
    st.markdown('<h1 class="main-header">📊 Model Performance Analysis</h1>', unsafe_allow_html=True)
    
    with st.spinner("🔄 Loading performance data..."):
        X, y, _ = load_dataset_features()
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        
        knn_preds = knn.predict(X_pca)
        svm_preds = svm.predict(X_pca)
    
    # Calculate metrics
    def calculate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        return accuracy, precision, recall, f1
    
    knn_metrics = calculate_metrics(y, knn_preds)
    svm_metrics = calculate_metrics(y, svm_preds)
    
    # 1. METRIC CARDS
    st.markdown("### 🎯 Performance Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔵 KNN Model")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Accuracy", f"{knn_metrics[0]:.3f}")
            st.metric("Recall", f"{knn_metrics[2]:.3f}")
        with metrics_col2:
            st.metric("Precision", f"{knn_metrics[1]:.3f}")
            st.metric("F1-Score", f"{knn_metrics[3]:.3f}")
    
    with col2:
        st.markdown("#### 🔴 SVM Model")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Accuracy", f"{svm_metrics[0]:.3f}")
            st.metric("Recall", f"{svm_metrics[2]:.3f}")
        with metrics_col2:
            st.metric("Precision", f"{svm_metrics[1]:.3f}")
            st.metric("F1-Score", f"{svm_metrics[3]:.3f}")
    
    # 2. COMPARISON CHARTS
    st.markdown("### 📊 Model Comparison")
    
    # Create comparison data
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'KNN': knn_metrics,
        'SVM': svm_metrics
    })
    
    # Bar chart comparison
    fig_bar = go.Figure(data=[
        go.Bar(name='KNN', x=metrics_df['Metric'], y=metrics_df['KNN'], marker_color='#3498db'),
        go.Bar(name='SVM', x=metrics_df['Metric'], y=metrics_df['SVM'], marker_color='#e74c3c')
    ])
    fig_bar.update_layout(
        title="Model Performance Comparison",
        barmode='group',
        height=400,
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 3. CONFUSION MATRICES
    st.markdown("### 🎯 Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔵 KNN Confusion Matrix")
        cm_knn = confusion_matrix(y, knn_preds)
        
        # Plotly heatmap for KNN
        fig_knn = go.Figure(data=go.Heatmap(
            z=cm_knn,
            x=['Healthy', 'Diseased'],
            y=['Healthy', 'Diseased'],
            colorscale='Blues',
            text=cm_knn,
            texttemplate="%{text}",
            textfont={"size": 20}
        ))
        fig_knn.update_layout(
            title="KNN Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        st.plotly_chart(fig_knn, use_container_width=True)
        
        # Classification report
        st.text("KNN Classification Report:")
        report_knn = classification_report(y, knn_preds, target_names=['Healthy', 'Diseased'])
        st.text(report_knn)
    
    with col2:
        st.markdown("#### 🔴 SVM Confusion Matrix")
        cm_svm = confusion_matrix(y, svm_preds)
        
        # Plotly heatmap for SVM
        fig_svm = go.Figure(data=go.Heatmap(
            z=cm_svm,
            x=['Healthy', 'Diseased'],
            y=['Healthy', 'Diseased'],
            colorscale='Reds',
            text=cm_svm,
            texttemplate="%{text}",
            textfont={"size": 20}
        ))
        fig_svm.update_layout(
            title="SVM Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        st.plotly_chart(fig_svm, use_container_width=True)
        
        # Classification report
        st.text("SVM Classification Report:")
        report_svm = classification_report(y, svm_preds, target_names=['Healthy', 'Diseased'])
        st.text(report_svm)
    
    # 4. DETAILED METRICS TABLE
    st.markdown("### 📋 Detailed Metrics Table")
    detailed_metrics = pd.DataFrame({
        'Model': ['KNN', 'SVM'],
        'Accuracy': [f"{knn_metrics[0]:.4f}", f"{svm_metrics[0]:.4f}"],
        'Precision': [f"{knn_metrics[1]:.4f}", f"{svm_metrics[1]:.4f}"],
        'Recall': [f"{knn_metrics[2]:.4f}", f"{svm_metrics[2]:.4f}"],
        'F1-Score': [f"{knn_metrics[3]:.4f}", f"{svm_metrics[3]:.4f}"]
    })
    st.dataframe(detailed_metrics, use_container_width=True)
    
    # 5. RADAR CHART
    st.markdown("### 🕸️ Model Performance Radar")
    
    fig_radar = go.Figure()
    
    # KNN radar
    fig_radar.add_trace(go.Scatterpolar(
        r=knn_metrics,
        theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        fill='toself',
        name='KNN',
        marker_color='#3498db'
    ))
    
    # SVM radar
    fig_radar.add_trace(go.Scatterpolar(
        r=svm_metrics,
        theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        fill='toself',
        name='SVM',
        marker_color='#e74c3c'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Performance Radar Chart",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # 6. WINNER ANNOUNCEMENT
    st.markdown("### 🏆 Best Model")
    if knn_metrics[0] > svm_metrics[0]:
        st.success(f"🥇 **KNN** is the winner with {knn_metrics[0]:.1%} accuracy!")
    elif svm_metrics[0] > knn_metrics[0]:
        st.success(f"🥇 **SVM** is the winner with {svm_metrics[0]:.1%} accuracy!")
    else:
        st.info("🤝 It's a tie! Both models perform equally well.")