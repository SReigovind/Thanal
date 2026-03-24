import os
import csv
import numpy as np
from datetime import datetime

class HealthAnalyzer:
    def __init__(self, camera_id="cam_default", stress_threshold_pct=15.0):
        self.camera_id = camera_id
        # Creates a unique history file for each camera
        self.csv_file = f"{self.camera_id}_health_history.csv"
        self.stress_threshold_pct = stress_threshold_pct
        
        if not os.path.exists(self.csv_file):
            self._create_csv()

    def _create_csv(self):
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Avg_Green", "Avg_VNIR", "Health_Ratio", "Status"])

    def clear_history(self):
        self._create_csv()
        print(f"🧹 Cleared history in {self.csv_file}")

    def analyze_and_log(self, rgb_image, vnir_image, leaf_mask):
        g_channel = rgb_image[:, :, 1].astype(np.float32)
        leaf_g = g_channel[leaf_mask > 0]
        leaf_vnir = vnir_image[leaf_mask > 0]
        
        if len(leaf_vnir) == 0:
            return {"status": "No Leaf Detected", "avg_g": 0, "avg_vnir": 0, "ratio": 0, "ready": False}

        avg_g = np.mean(leaf_g)
        avg_vnir = np.mean(leaf_vnir)
        current_ratio = avg_vnir / (avg_g + 1e-5)

        history_ratios = []
        with open(self.csv_file, mode='r') as file:
            reader = list(csv.reader(file))
            if len(reader) > 1:
                history_ratios = [float(row[3]) for row in reader[1:]]
        
        history_ratios.append(current_ratio) 
        total_scans = len(history_ratios)

        stats = {
            "avg_g": avg_g,
            "avg_vnir": avg_vnir,
            "ratio": current_ratio,
            "ready": False
        }

        status = "Calibrating..."

        if total_scans < 5:
            status = f"Calibrating ({total_scans}/5)..."
        else:
            stats["ready"] = True
            
            baseline = np.mean(history_ratios[0:5])
            global_avg = np.mean(history_ratios)
            current_5_avg = np.mean(history_ratios[-5:])
            
            if total_scans >= 10:
                prev_5_avg = np.mean(history_ratios[-10:-5])
            else:
                prev_5_avg = baseline 

            def calc_diff(new_val, old_val):
                return ((new_val - old_val) / old_val) * 100

            stats["vs_baseline"] = calc_diff(current_ratio, baseline)
            stats["vs_prev_check"] = calc_diff(current_ratio, prev_5_avg)
            stats["recent5_vs_global"] = calc_diff(current_5_avg, global_avg)

            if stats["vs_baseline"] <= -self.stress_threshold_pct:
                status = "ALERT: STRESS"
            else:
                status = "Healthy Tracking"

        stats["status"] = status

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, f"{avg_g:.2f}", f"{avg_vnir:.2f}", f"{current_ratio:.4f}", status])

        return stats