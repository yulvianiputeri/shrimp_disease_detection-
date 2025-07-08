
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

def train_knn_optimized(X_train, y_train, X_test, y_test):
    param_grid_knn = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    grid_knn = GridSearchCV(knn, param_grid_knn, cv=3, scoring='accuracy', n_jobs=-1)
    grid_knn.fit(X_train, y_train)
    best_knn = grid_knn.best_estimator_
    preds = best_knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return acc, report, preds, y_test, best_knn

def train_svm_optimized(X_train, y_train, X_test, y_test):
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear']
    }
    svm = SVC(probability=True, random_state=42)
    grid_svm = GridSearchCV(svm, param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1)
    grid_svm.fit(X_train, y_train)
    best_svm = grid_svm.best_estimator_
    preds = best_svm.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return acc, report, preds, y_test, best_svm

def bandingkan_model(X_train, y_train, X_test, y_test):
    hasil = {}

    # KNN
    acc_knn, report_knn, preds_knn, y_test_knn, model_knn = train_knn_optimized(X_train, y_train, X_test, y_test)
    hasil['KNN'] = {
        'akurasi': acc_knn,
        'laporan': report_knn,
        'prediksi': preds_knn,
        'model': model_knn
    }

    # SVM
    acc_svm, report_svm, preds_svm, y_test_svm, model_svm = train_svm_optimized(X_train, y_train, X_test, y_test)
    hasil['SVM'] = {
        'akurasi': acc_svm,
        'laporan': report_svm,
        'prediksi': preds_svm,
        'model': model_svm
    }

    return hasil
