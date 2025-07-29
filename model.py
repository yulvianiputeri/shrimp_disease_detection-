from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV 

def train_knn_optimized(X_train, y_train, X_test, y_test):
    """
    Melatih model KNN dengan optimasi hyperparameter menggunakan GridSearchCV.
    Mengembalikan akurasi, laporan klasifikasi, prediksi, label sebenarnya, dan model terbaik.
    """
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'] 
    }
    knn = KNeighborsClassifier()
    grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1, verbose=1) 
    grid_knn.fit(X_train, y_train)
    
    best_knn = grid_knn.best_estimator_
    
    knn_calib = CalibratedClassifierCV(best_knn, cv=3, method='isotonic') 

    preds = knn_calib.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    
    print(f"Best KNN parameters: {grid_knn.best_params_}")
    return acc, report, preds, y_test, knn_calib

def train_svm_optimized(X_train, y_train, X_test, y_test):
    """
    Melatih model SVM dengan optimasi hyperparameter menggunakan GridSearchCV.
    Mengembalikan akurasi, laporan klasifikasi, prediksi, label sebenarnya, dan model terbaik.
    """
    param_grid_svm = {
        'C': [0.01, 0.1, 1, 10, 100], 
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1], 
        'kernel': ['rbf', 'linear'] 
    }
    svm = SVC(probability=True, random_state=42) 
    grid_svm = GridSearchCV(svm, param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1, verbose=1) 
    grid_svm.fit(X_train, y_train)
    
    best_svm = grid_svm.best_estimator_
    
    svm_calib = CalibratedClassifierCV(best_svm, cv=3, method='isotonic') 
    svm_calib.fit(X_train, y_train)

    preds = svm_calib.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    
    print(f"Best SVM parameters: {grid_svm.best_params_}")
    return acc, report, preds, y_test, svm_calib

def bandingkan_model(X_train, y_train, X_test, y_test):
    """
    Melatih dan membandingkan model KNN dan SVM yang telah dioptimasi.
    """
    hasil = {}

    print("\n--- Training KNN Model ---")
    acc_knn, report_knn, preds_knn, y_test_knn, model_knn = train_knn_optimized(X_train, y_train, X_test, y_test)
    hasil['KNN'] = {
        'akurasi': acc_knn,
        'laporan': report_knn,
        'prediksi': preds_knn,
        'model': model_knn
    }
    print(f"KNN Accuracy: {acc_knn:.4f}")
    print("KNN Classification Report:\n", report_knn)

    print("\n--- Training SVM Model ---")
    acc_svm, report_svm, preds_svm, y_test_svm, model_svm = train_svm_optimized(X_train, y_train, X_test, y_test)
    hasil['SVM'] = {
        'akurasi': acc_svm,
        'laporan': report_svm,
        'prediksi': preds_svm,
        'model': model_svm
    }
    print(f"SVM Accuracy: {acc_svm:.4f}")
    print("SVM Classification Report:\n", report_svm)

    return hasil

