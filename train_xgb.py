import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
import numpy as np 
from sklearn.decomposition import PCA
import pickle
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns

import threading
import psutil
import time
import csv

monitoring = True
monitor_data = []

def preprocess_data(df, fit=False, scaler=None, pca=None):
    label_col = df['Label'] if 'Label' in df.columns else None

    # Loại bỏ đi những cột không có ý nghĩa
    non_numeric_cols = ['Label', 'frame.time', 'wlan.bssid', 'wlan.da', 'wlan.ra', 'wlan.sa', 
                        'wlan.ta', 'radiotap.present.tsft', 'radiotap.rxflags', 'wlan.fc.ds']
    
    df_numeric = df.drop(columns=[col for col in non_numeric_cols if col in df.columns], errors='ignore')
    
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    df_numeric = df_numeric.fillna(df_numeric.mean())

    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    # Bước 1.2.3: Xử lý các giá trị NaN còn lại (ví dụ: thay thế bằng 0, hoặc giá trị trung bình/trung vị)
    # Thay thế NaN bằng 0 là một cách đơn giản. Tùy thuộc vào dữ liệu thực tế, bạn có thể muốn dùng mean/median.
    df_numeric = df_numeric.fillna(df_numeric.mean())
    
    # Loại bỏ cột có phương sai bằng 0 (Phương sai = 0 có nghĩa là hằng số, không có trọng số trong quá trình huấn luyện)
    df_numeric = df_numeric.loc[:, df_numeric.var() > 0]
    
    # Chuẩn hóa (Normalization)
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric)
    else:
        X_scaled = scaler.transform(df_numeric)
    
    # PCA
    if fit:
        pca = PCA()
        pca.fit(X_scaled)
        # Tìm số thành phần đạt 95% phương sai
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
        pca = PCA(n_components=n_components_95)
        X_pca = pca.fit_transform(X_scaled)
    else:
        X_pca = pca.transform(X_scaled)
    
    return X_pca, label_col, scaler, pca, df_numeric.columns

def load_pca_configuration(df):
    print("Số dòng trong file gốc:", len(df))

    try:
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        pca = pickle.load(open('pca.pkl', 'rb'))
        print("Đã tải Scaler và PCA từ file.")
        X_pca, label_col,  scaler, pca, feature_names = preprocess_data(df, fit=False, scaler=scaler, pca=pca)
    except FileNotFoundError:
        print("Không tìm thấy file Scaler hoặc PCA. Fit lại trên toàn bộ dữ liệu.")
        X_pca, label_col, scaler, pca, feature_names = preprocess_data(df, fit=True)
        pickle.dump(scaler, open('scaler.pkl', 'wb'))
        pickle.dump(pca, open('pca.pkl', 'wb'))

    return X_pca, label_col, scaler, pca, feature_names

def monitor_system():
    while monitoring:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        ram_used_mb = memory.used / (1024 * 1024)
        monitor_data.append((cpu_percent, ram_used_mb))
        time.sleep(1)

if __name__ == "__main__":
    df = pd.read_csv('combined_shuffled_dataset_cleaned.csv')
    print("Số dòng trong file gốc:", len(df))

    X_pca, label_col, scaler, pca, feature_names  = load_pca_configuration(df)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_col)

    # --- CHIA DỮ LIỆU THÀNH TẬP HUẤN LUYỆN VÀ KIỂM ĐỊNH ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\nKích thước tập huấn luyện X: {X_train.shape}")
    print(f"Kích thước tập kiểm định X: {X_test.shape}")
    print(f"Kích thước tập huấn luyện y: {y_train.shape}")
    print(f"Kích thước tập kiểm định y: {y_test.shape}")
    print(f"Tỉ lệ lớp trong y_train (0/1): {np.bincount(y_train) / len(y_train)}")
    print(f"Tỉ lệ lớp trong y_test (0/1): {np.bincount(y_test) / len(y_test)}")

    # --- PHẦN 2: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH XGBOOST ---
    """ Các siêu tham số của mô hình XGBoost 
        objective: Hàm mục tiêu ('binary:logistic' cho phân loại nhị phân với đầu ra xác suất)
        n_estimators: Số lượng cây boosting
        learning_rate: Tốc độ học (shrinkage)
        max_depth: Độ sâu tối đa của mỗi cây con
        subsample: Tỷ lệ lấy mẫu con của dữ liệu (hàng) cho mỗi cây
        colsample_bytree: Tỷ lệ lấy mẫu con của thuộc tính (cột) cho mỗi cây
        gamma: Mức giảm mất mát tối thiểu cần thiết để thực hiện một phân chia.
        reg_alpha (L1) và reg_lambda (L2): Tham số regularization cho trọng số lá.
        eval_metric: Metric để đánh giá trong quá trình huấn luyện (để theo dõi early stopping)
        use_label_encoder=False: Để tránh cảnh báo deprecation
        tree_method='hist': Sử dụng thuật toán xây dựng cây dựa trên histogram, nhanh hơn cho dữ liệu lớn.
        n_jobs: Số lượng nhân CPU để sử dụng (-1 nghĩa là sử dụng tất cả)
    """

    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',  # Thích hợp cho phân loại nhị phân
        n_estimators=300,             # Số lượng cây boosting
        learning_rate=0.05,           # Tốc độ học
        max_depth=6,                  # Độ sâu tối đa của mỗi cây
        subsample=0.7,                # Sử dụng 70% mẫu ngẫu nhiên cho mỗi cây
        colsample_bytree=0.7,         # Sử dụng 70% thuộc tính ngẫu nhiên cho mỗi cây
        gamma=0.1,                    # Minimum loss reduction for a split
        reg_alpha=0.005,              # L1 regularization on weights
        reg_lambda=1,                 # L2 regularization on weights
        use_label_encoder=False,      # Tắt cảnh báo về label encoder (đã deprecated)
        eval_metric='logloss',        # Metric đánh giá (có thể là 'auc', 'error')
        random_state=42,
        n_jobs=-1                     # Sử dụng tất cả các nhân CPU
    )

    print("\n-------------------- Bắt đầu huấn luyện với mô hình XGBoost -------------------- ")
    t = threading.Thread(target=monitor_system)
    t.start()
    t0 = time.time()
    xgb_classifier.fit(X_train, y_train,
                    eval_set=[(X_test, y_test)], # Tập validation để theo dõi hiệu suất
                    early_stopping_rounds=20,    # Dừng nếu không có cải thiện trong 20 vòng liên tiếp
                    verbose=False)               # Tắt hiển thị chi tiết từng vòng huấn luyện
    
    monitoring = False
    t.join()
    with open('log_train_xgb.csv', mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cpu_percent", "ram_used_mb"])
        writer.writerows(monitor_data)
    print("-------------------- Kết thúc quá trình huấn luyện mô hình vf à lưu mô hình --------------------")
    xgb_classifier.save_model("xgb_model.json")  # Hoặc .bin cũng được
    t1=time.time()
    print(f"Huấn luyện hoàn tất. Số lượng cây thực tế: {xgb_classifier.best_iteration + 1 if hasattr(xgb_classifier, 'best_iteration') else xgb_classifier.n_estimators} cây.")
    print(f'Thời gian huấn luyện: {(t1-t0)} (seconds)')

    print("\n--------------------Load mô hình đã train và đánh giá mô hình XGBoost trên tập Test -------------------- ")
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.load_model("xgb_model.json") 
    
    y_pred_encoded = xgb_classifier.predict(X_test)
    # Dự đoán xác suất cho lớp dương (lớp 1, tương ứng với label_encoder.classes_[1])
    y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]

    # **BƯỚC QUAN TRỌNG: CHUYỂN ĐỔI y_test VÀ y_pred TRỞ LẠI NHÃN GỐC**
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)

    # Tính toán các metric đánh giá
    accuracy = accuracy_score(y_test_original, y_pred_original)
    auc_score = roc_auc_score(y_test_original, y_pred_proba) # AUC cần xác suất và nhãn gốc

    print(f"\nĐộ chính xác của mô hình XGBoost trên tập kiểm định: {accuracy:.4f}")
    print(f"AUC Score của mô hình XGBoost trên tập kiểm định: {auc_score:.4f}")

    # Hiển thị báo cáo phân loại chi tiết với NHÃN GỐC
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test_original, y_pred_original, target_names=label_encoder.classes_))

    # Hiển thị Ma trận nhầm lẫn với NHÃN GỐC
    conf_matrix = confusion_matrix(y_test_original, y_pred_original)
    print("\nMa trận nhầm lẫn:")
    print(conf_matrix)

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, # Sử dụng nhãn gốc cho trục X
                yticklabels=label_encoder.classes_) # Sử dụng nhãn gốc cho trục Y
    plt.title('Ma trận nhầm lẫn (XGBoost)')
    plt.ylabel('Giá trị thực tế')
    plt.xlabel('Giá trị dự đoán')
    plt.savefig('Confusion Matrix của XGBoost.png', dpi=500, bbox_inches='tight') # Lưu dưới dạng PNG với 500 DPI
    plt.show()


    print("\n-------------------- Độ quan trọng của thuộc tính (Feature Importances) với mô hình XGBoost --------------------")
    feature_importances = xgb_classifier.feature_importances_
    feature_names = feature_names

    sorted_indices = np.argsort(feature_importances)[::-1]

    print("\nĐộ quan trọng của thuộc tính (Feature Importances):")
    for i in sorted_indices:
        print(f"- {feature_names[i]}: {feature_importances[i]:.4f}")

    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in sorted_indices],
            [feature_importances[i] for i in sorted_indices],
            color='skyblue',
            linewidth=0) # Vẫn giữ linewidth=0 để khắc phục lỗi trước đó nếu có

    plt.xlabel("Độ quan trọng")
    plt.title("Độ quan trọng của thuộc tính trong XBGoost")
    plt.gca().invert_yaxis()
    plt.savefig('Độ quan trọng của thuộc tính trong XBGoost.png', dpi=500, bbox_inches='tight') # Lưu dưới dạng PNG với 500 DPI
    plt.show()