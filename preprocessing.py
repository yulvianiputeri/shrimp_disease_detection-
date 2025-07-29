import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
import mahotas  
from PIL import Image, ImageEnhance

base_path = 'data_udang'
folders = ['1. Healthy', '2. BG', '3. WSSV', '4. WSSV_BG']
label_map = {'1. Healthy': 0, '2. BG': 1, '3. WSSV': 1, '4. WSSV_BG': 1}
TARGET_SIZE = (128, 128)

def augment_image(img_pil):

    augmented_images = [img_pil] 

    # Flip Horizontal
    augmented_images.append(img_pil.transpose(Image.FLIP_LEFT_RIGHT))

    augmented_images.append(img_pil.rotate(10))
    augmented_images.append(img_pil.rotate(-10))

    # Perubahan Kecerahan
    enhancer = ImageEnhance.Brightness(img_pil)
    augmented_images.append(enhancer.enhance(0.8)) 
    augmented_images.append(enhancer.enhance(1.2)) 

    # Perubahan Kontras
    enhancer = ImageEnhance.Contrast(img_pil)
    augmented_images.append(enhancer.enhance(0.8)) 
    augmented_images.append(enhancer.enhance(1.2)) 

    return augmented_images

def load_images_with_augmentation():
    """
    Memuat gambar dari folder dan melakukan augmentasi data.
    """
    images = []
    labels = []
    filenames = [] 

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' not found. Skipping.")
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            try:
                img_pil = Image.open(img_path).convert('RGB').resize(TARGET_SIZE)
                
                # Lakukan augmentasi
                augmented_imgs = augment_image(img_pil)
                
                for aug_img in augmented_imgs:
                    images.append(np.array(aug_img))
                    labels.append(label_map[folder])
                    filenames.append(img_file) 

            except Exception as e:
                print(f"Error loading or augmenting {img_path}: {str(e)}")
                continue
    return np.array(images), np.array(labels), filenames

def extract_features(images):
    """
    Ekstraksi fitur LBP dan Haralick (GLCM) dari kumpulan gambar.
    """
    lbp_features = []
    glcm_features = []
    
    # Parameter LBP 
    radius = 3
    n_points = 8 * radius
    METHOD = 'uniform'

    for img_np in images:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # LBP
        lbp = local_binary_pattern(gray, n_points, radius, METHOD)
        hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        lbp_features.append(hist)

        # Haralick (GLCM)
        if gray.shape[0] < 2 or gray.shape[1] < 2: 
            glcm_props = np.zeros(13) #
        else:
            glcm_props = mahotas.features.haralick(gray).mean(axis=0)
        glcm_features.append(glcm_props)

    return np.hstack([np.array(lbp_features), np.array(glcm_features)])

def load_and_preprocess_data():
    """
    Memuat, mengaugmentasi, mengekstrak fitur, menskalakan, menerapkan PCA,
    dan membagi data menjadi set pelatihan dan pengujian.
    """
    images, labels, _ = load_images_with_augmentation()
    print(f"Total images after augmentation: {len(images)}")
    
    features = extract_features(images)
    print(f"Features shape: {features.shape}")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"Features scaled shape: {features_scaled.shape}")

   
    n_components_to_use = min(50, features.shape[1] - 1) 
    pca = PCA(n_components=n_components_to_use, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    print(f"Features PCA shape: {features_pca.shape}")
    print(f"Explained variance ratio by PCA: {np.sum(pca.explained_variance_ratio_):.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        features_pca, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler, pca

