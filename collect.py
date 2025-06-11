import pyshark
import pandas as pd
from datetime import datetime
import numpy as np

interface = 'Wi-Fi'

cap = pyshark.LiveCapture(
    interface=interface,
    use_json=True,  
    include_raw=False
)

data = []

for pkt in cap.sniff_continuously(packet_count=100):  
    try:
        frame = pkt.frame_info
        radiotap = pkt.get_multiple_layers('radiotap')[0] if 'radiotap' in pkt else None
        wlan = pkt.get_multiple_layers('wlan')[0] if 'wlan' in pkt else None
        wlan_radio = pkt.get_multiple_layers('wlan_radio')[0] if 'wlan_radio' in pkt else None

        timestamp = float(frame.time_epoch) if hasattr(frame, 'time_epoch') else None
        dt = datetime.fromtimestamp(timestamp) if timestamp else None

        row = {
            'frame.len': getattr(frame, 'len', None),
            'frame.number': getattr(frame, 'number', None),
            'frame.time_delta': getattr(frame, 'time_delta', None),
            'radiotap.length': getattr(radiotap, 'length', None) if radiotap else None,
            'wlan.duration': getattr(wlan, 'duration', None) if wlan else None,
            'wlan.fc.frag': getattr(wlan, 'fc_frag', None) if wlan else None,
            'wlan.fc.order': getattr(wlan, 'fc_order', None) if wlan else None,
            'wlan.fc.moredata': getattr(wlan, 'fc_more_data', None) if wlan else None,
            'wlan.fc.protected': getattr(wlan, 'fc_protected', None) if wlan else None,
            'wlan.fc.pwrmgt': getattr(wlan, 'fc_pwr_mgt', None) if wlan else None,
            'wlan.fc.retry': getattr(wlan, 'fc_retry', None) if wlan else None,
            'wlan.fc.subtype': getattr(wlan, 'fc_subtype', None) if wlan else None,
            'wlan_radio.duration': getattr(wlan_radio, 'duration', None) if wlan_radio else None,
            'wlan_radio.data_rate': getattr(wlan_radio, 'data_rate', None) if wlan_radio else None,
            'wlan_radio.signal_dbm': getattr(wlan_radio, 'signal_dbm', None) if wlan_radio else None,
            'frame_hour': dt.hour if dt else None,
            'frame_second': dt.second if dt else None,
        }
        data.append(row)

    except Exception as e:
        print(f"Lỗi xử lý gói: {e}")
df = pd.DataFrame(data)
df.to_csv('pyshark.csv', index=False)

