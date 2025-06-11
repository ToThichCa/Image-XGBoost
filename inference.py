
import torch.nn as nn
import numpy as np 
import pandas as pd
import time 
import pickle
import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.preprocessing import LabelEncoder

import threading
import psutil
import time
import csv
monitoring = True
monitor_data = []
def monitor_system():
    while monitoring:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        ram_used_mb = memory.used / (1024 * 1024)
        monitor_data.append((cpu_percent, ram_used_mb))
        time.sleep(1)
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
    
if __name__ == "__main__":
    df_real = pd.read_csv(
    "capture_HUST_C7.csv",
        quotechar='"',
        error_bad_lines = False,
        low_memory=False
    )
    df_real_clone = df_real

    df_real_clone['frame.time'] = pd.to_datetime(
        df_real_clone['frame.time'].str.replace(' GTB Standard Time', '', regex=False),
        format="%b %d- %Y %H:%M:%S.%f",
        errors='coerce'
    )

    clone = pd.concat([
        df_real_clone,
        df_real_clone['frame.time'].dt.year.rename('frame_year'),
        df_real_clone['frame.time'].dt.month.rename('frame_month'),
        df_real_clone['frame.time'].dt.day.rename('frame_day'),
        df_real_clone['frame.time'].dt.hour.rename('frame_hour'),
        df_real_clone['frame.time'].dt.minute.rename('frame_minute'),
        df_real_clone['frame.time'].dt.second.rename('frame_second'),
        df_real_clone['frame.time'].dt.dayofweek.rename('frame_dayofweek')
    ], axis=1)

    df_time_cleaned = clone.copy()

    df_real_keep = df_time_cleaned[['frame.len', 'frame.number', 'frame.time_delta', 'radiotap.length',
       'wlan.duration', 'wlan.fc.frag', 'wlan.fc.order', 'wlan.fc.moredata',
       'wlan.fc.protected', 'wlan.fc.pwrmgt', 'wlan.fc.retry',
       'wlan.fc.subtype', 'wlan_radio.duration', 'wlan_radio.data_rate',
       'wlan_radio.signal_dbm', 'frame_hour', 'frame_second']]

    df_real_keep = df_real_keep.fillna(0)

    scaler = pickle.load(open('scaler.pkl', 'rb'))
    pca = pickle.load(open('pca.pkl', 'rb'))    
    label_col  = pickle.load(open('label_col.pkl', 'rb'))

    sample_idx = np.random.randint(0, len(df_real_keep))
    sample_df = df_real_keep.iloc[sample_idx:sample_idx+1]

    # Lấy đặc trưng gốc của mẫu (chỉ các cột feature_names)
    feature_names = df_real_keep.columns
    sample_X_raw = sample_df[feature_names].values
    sample_X_raw
    # Transform mẫu: Scale và PCA
    # Bước 1: Scale dữ liệu bằng scaler đã huấn luyện
    sample_X_scaled = scaler.transform(sample_X_raw)
    
    # Bước 2: Áp dụng PCA để giảm chiều
    sample_X_pca = pca.transform(sample_X_scaled)
    
    # Chuyển mẫu PCA sang Tensor
    sample_X_tensor = torch.tensor(sample_X_pca, dtype=torch.float32).to(device)
    
    # Khởi tạo mô hình FT-Transformer
    num_features = sample_X_pca.shape[1]
    model = MLP(num_features).to(device)
    
    # Tải trọng số mô hình
    model.load_state_dict(torch.load('mlp_model.pth'))
    model.eval()
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_col)

    # Thực hiện inference và tính thời gian
    start_time = time.perf_counter()
    t = threading.Thread(target=monitor_system)
    t.start()
    with torch.no_grad():
        pred_prob = model(sample_X_tensor)
    end_time = time.perf_counter()
    monitoring = False
    t.join()
    with open('log_infer_mlp.csv', mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cpu_percent", "ram_used_mb"])
        writer.writerows(monitor_data)

    # Chuyển xác suất thành nhãn
    pred_prob = pred_prob.cpu().numpy()[0, 0]
    pred_label = 1 if pred_prob >= 0.5 else 0
    pred_label_name = label_encoder.inverse_transform([pred_label])[0]

    # Tính thời gian inference (seconds)
    inference_time_ms = (end_time - start_time) * 1000
    print(f'Inference time : {inference_time_ms} miliseconds')
    print(f'Xác suất đự doán : {pred_label}')
    print(f'Nhãn đự doán : {pred_label_name}')

