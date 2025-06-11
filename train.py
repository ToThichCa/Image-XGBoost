import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau,LambdaLR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import seaborn as sns 
import threading
import psutil
import time
import csv
monitoring = True
monitor_data = []
warnings.filterwarnings('ignore')

def monitor_system():
    while monitoring:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        ram_used_mb = memory.used / (1024 * 1024)
        monitor_data.append((cpu_percent, ram_used_mb))
        time.sleep(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def calculate_accuracy(outputs, labels):
    preds = (outputs >= 0.5).float()
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy

def plot_and_save(train_losses,val_losses,train_accuracies,val_accuracies,file_name_figure = "unknow1.png"):
    plt.figure(figsize=(12, 5))
    
    train_losses = torch.tensor(train_losses).cpu().numpy()
    val_losses = torch.tensor(val_losses).cpu().numpy()
    
    train_accuracies = torch.tensor(train_accuracies).cpu().numpy()
    val_accuracies = torch.tensor(val_accuracies).cpu().numpy()
    
    # Biểu đồ loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Biểu đồ accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(file_name_figure,dpi=500, bbox_inches='tight')
    plt.show()

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

def train_test_split_cus(X,y_encoded,batch_size = 512):
    # Chia dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    print("Số mẫu train:", len(X_train))
    print("Số mẫu test:", len(X_test))
    # Chia tập train thành train và validation (lấy 20% của tập train làm valid còn lại là train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print("Số mẫu validation:", len(X_val))
    
    # Chuyển dữ liệu sang Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    
    # Tạo DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class MLP(nn.Module):
    def __init__(self, input_dim, dropout = 0.1):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout)
        )
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

def training(model, train_loader,val_loader,
              criterion, optimizer,scheduler,warmup_scheduler,class_weights_tensor,
              model_name ="best_mlp_model.pth",
              num_epochs = 10):
    # Huấn luyện mô hình
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    # Đo time 
    t1 = time.time()
    
    for epoch in range(num_epochs):
        # Huấn luyện
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            weights = class_weights_tensor[y_batch.long().squeeze()]
            loss = criterion(outputs, y_batch)
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_acc += calculate_accuracy(outputs, y_batch) * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
    
        # Đánh giá trên validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_acc += calculate_accuracy(outputs, y_batch) * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
        # Cập nhật learning rate
        scheduler.step(val_loss)
        warmup_scheduler.step()
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stop_msg = f"Early stopping triggered at epoch {epoch+1}"
                print(stop_msg)
                break

    print("-------------------- Kết thúc quá trình huấn luyện mô hình -------------------- ")
    t2 = time.time()
    print('Thời gian training là: {} seconds'.format((t2 - t1)))

    return train_losses, train_accuracies, val_losses, val_accuracies

if __name__ == "__main__":
    # Đọc data được clean -> transform data
    df = pd.read_csv('combined_shuffled_dataset_cleaned.csv')
    X_pca, label_col, scaler, pca, feature_names  = load_pca_configuration(df)

    # Tách đặc trưng (X) và nhãn (y), sau đó mã hóa nhãn
    X = X_pca
    y = label_col
    num_features = X_pca.shape[1]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("Classes:", label_encoder.classes_)

    # Chia train, val, test loader
    train_loader, val_loader, test_loader = train_test_split_cus(X,y_encoded,batch_size = 512)

    # Tính class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights)

    # Khởi tạo mô hình
    input_dim = X.shape[1]
    model = MLP(input_dim, dropout = 0.1).to(device)
    
    # Khởi tạo hàm loss, optimizer và scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-5)

    # Khởi tạo Warm-up scheduler cho quá trình training
    def lr_lambda(epoch):
        if epoch < 5:
            return epoch / 5.0
        return 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)

    # Traing 
    print("-------------------- Bắt đầu huấn luyện với mô hình MLP -------------------- ")
    t = threading.Thread(target=monitor_system)
    t.start()
    train_losses, train_accuracies, val_losses, val_accuracies = training(model,train_loader, val_loader, criterion,optimizer,scheduler,warmup_scheduler,
                                                                        class_weights_tensor,"best_mlp_model.pth")
    monitoring = False
    t.join()
    with open('log_train_mlp.csv', mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cpu_percent", "ram_used_mb"])
            writer.writerows(monitor_data)

    print("\n-------------------- Đánh giá mô hình MLP trên tập Test -------------------- ")
    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = (outputs >= 0.5).float().cpu().numpy().flatten()
            test_preds.extend(preds)
            test_labels.extend(y_batch.cpu().numpy().flatten())
            
    # In kết quả đánh giá Accuracy và AUC
    print("\nAccuracy trên tập test:", accuracy_score(test_labels, test_preds))
    print("\nAUC Score trên tập test:", roc_auc_score(test_labels, test_preds))

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=label_encoder.classes_))

    # Hiển thị Confusion Matrix
    conf_matrix = confusion_matrix(test_preds, test_labels)
    print("\nMa trận nhầm lẫn:")
    print(conf_matrix)

    # Trực quan hóa Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dự đoán Lớp Evil_Twin', 'Dự đoán Lớp Normal'],
                yticklabels=['Thực tế Lớp Evil_Twin', 'Thực tế Lớp Normal'])
    plt.title('Ma trận nhầm lẫn (Multi-layer Perceptron)')
    plt.ylabel('Giá trị thực tế')
    plt.xlabel('Giá trị dự đoán')
    plt.savefig('Confusion Matrix của Multi-layer Perceptron.png', dpi=500, bbox_inches='tight') # Lưu dưới dạng PNG với 500 DPI
    plt.show()

    plot_and_save(train_losses, val_losses, train_accuracies,val_accuracies, "mlp_plot.png")

    print("\n-------------------- Inference 1 sample với mô hình MLP -------------------- ")
    # Đọc dữ liệu gốc
    df = pd.read_csv('combined_shuffled_dataset_cleaned.csv')
    print("Số dòng trong file gốc:", len(df))
    X_pca, label_col, scaler, pca, feature_names  = load_pca_configuration(df)
    # Chuyển sang PCA hoặc chuyển 1 samples sang dạng đã được PCA

    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_col)
    print("Classes:", label_encoder.classes_)

    # Chọn ngẫu nhiên một mẫu từ DataFrame
    sample_idx = np.random.randint(0, len(df))
    sample_df = df.iloc[sample_idx:sample_idx+1]

    # Lấy đặc trưng gốc của mẫu (chỉ các cột feature_names)
    sample_X_raw = sample_df[feature_names].values

    # Transform mẫu: Scale và PCA
    # Bước 1: Scale dữ liệu bằng scaler đã huấn luyện
    sample_X_scaled = scaler.transform(sample_X_raw)

    # Bước 2: Áp dụng PCA để giảm chiều
    sample_X_pca = pca.transform(sample_X_scaled)

    # Lấy nhãn thực tế của mẫu
    sample_y = label_col.iloc[sample_idx]
    sample_y_encoded = y_encoded[sample_idx]

    # Chuyển mẫu PCA sang Tensor
    sample_X_tensor = torch.tensor(sample_X_pca, dtype=torch.float32).to(device)

    # Khởi tạo mô hình FT-Transformer
    num_features = X_pca.shape[1]
    model = MLP(input_dim).to(device)

    # Tải trọng số mô hình
    model.load_state_dict(torch.load('best_mlp_model.pth'))
    model.eval()

    # Thực hiện inference và tính thời gian
    start_time = time.perf_counter()
    with torch.no_grad():
        pred_prob = model(sample_X_tensor)
    end_time = time.perf_counter()

    # Chuyển xác suất thành nhãn
    pred_prob = pred_prob.cpu().numpy()[0, 0]
    pred_label = 1 if pred_prob >= 0.5 else 0
    pred_label_name = label_encoder.inverse_transform([pred_label])[0]

    # Tính thời gian inference (mili-giây)
    inference_time_ms = (end_time - start_time) * 1000

    # In kết quả
    print("\nInference trên một mẫu từ file gốc:")
    print(f"Index mẫu: {sample_idx}")
    print(f"Đặc trưng gốc ({len(feature_names)} cột): {sample_df[feature_names].values.flatten()}")
    print(f"Đặc trưng PCA ({num_features} cột): {sample_X_pca.flatten()}")
    print(f"Nhãn thực tế: {sample_y}")
    print(f"Nhãn dự đoán: {pred_label_name}")
    print(f"Xác suất (Evil_Twin): {pred_prob:.4f}")
    print(f"Thời gian inference: {inference_time_ms:.4f} (seconds)")