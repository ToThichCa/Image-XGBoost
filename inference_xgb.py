import numpy as np 
import pandas as pd
import time 
import pickle
import xgboost as xgb

if __name__ == "__main__":
    df_real = pd.read_csv(
    "capture_HUST_C7.csv",
        quotechar='"',
        error_bad_lines=False,
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
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.load_model("xgb_model.json") 
    t0= time.time()
    y_pred = xgb_classifier.predict(sample_X_pca)
    t1= time.time()
    
    print(f'Xác suất dự đoán: {y_pred}')
    print(f"Thời gian inference: {(t1-t0)*1000}(miliseconds)")
