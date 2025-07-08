# utils.py
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.feature import local_binary_pattern
import mahotas
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

def load_css():
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def extract_features_single(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    radius, n_points, METHOD = 3, 24, 'uniform'
    lbp = local_binary_pattern(gray, n_points, radius, METHOD)
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    glcm = mahotas.features.haralick(gray).mean(axis=0)
    return np.hstack([hist, glcm])

@st.cache_data
def load_dataset_features():
    df = pd.read_csv("shrimp_dataset.csv")
    features, labels, paths = [], [], []
    for idx, row in df.iterrows():
        img = Image.open(row['image_path']).convert('RGB').resize((128,128))
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, 24, 3, 'uniform')
        hist, _ = np.histogram(lbp, bins=np.arange(0, 27), range=(0, 26))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        glcm = mahotas.features.haralick(gray).mean(axis=0)
        features.append(np.hstack([hist, glcm]))
        labels.append(0 if row['label']=="Healthy" else 1)
        paths.append(row['image_path'])
    return np.array(features), np.array(labels), paths