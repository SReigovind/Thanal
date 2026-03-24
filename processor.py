import cv2
import os
import time
import threading
import numpy as np
from datetime import datetime
from PIL import Image

from inference import VNIREngine 
from analyzer import HealthAnalyzer
from notifier import WhatsAppNotifier

class ImageProcessorMulti:
    def __init__(self, model_path="ThanalModel.onnx", output_dir="monitoring_logs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("⚙️ Loading Edge NVR VNIR Engine (ONNX Runtime)...")
        self.engine = VNIREngine(model_path=model_path, device="cpu")
        
        self.analyzers = {}
        
        print("📱 Initializing WhatsApp Notifier...")
        self.notifier = WhatsAppNotifier()
        self.alert_cooldowns = {} 
        self.ALERT_COOLDOWN_SECONDS = 60 

    def get_analyzer(self, camera_id):
        if camera_id not in self.analyzers:
            print(f"📁 Initializing tracking history for {camera_id}")
            self.analyzers[camera_id] = HealthAnalyzer(camera_id=camera_id)
        return self.analyzers[camera_id]

    def clear_history(self, camera_id):
        analyzer = self.get_analyzer(camera_id)
        analyzer.clear_history()
        if camera_id in self.alert_cooldowns:
            del self.alert_cooldowns[camera_id]

    def process_frame(self, frame, camera_id="Unknown Camera", save_image=True):
        analyzer = self.get_analyzer(camera_id)
        
        frame_256 = cv2.resize(frame, (256, 256))
        
        hsv_frame = cv2.cvtColor(frame_256, cv2.COLOR_BGR2HSV)
        total_pixels = 256 * 256
        
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_hsv_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        
        green_hsv_mask = cv2.morphologyEx(green_hsv_mask, cv2.MORPH_CLOSE, kernel_large)
        green_hsv_mask = cv2.morphologyEx(green_hsv_mask, cv2.MORPH_OPEN, kernel_small)
        
        green_contours, _ = cv2.findContours(green_hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_green_area = 0
        best_green_cnt = None
        
        if green_contours:
            best_green_cnt = max(green_contours, key=cv2.contourArea)
            max_green_area = cv2.contourArea(best_green_cnt)

        lower_yellow = np.array([15, 70, 70]) 
        upper_yellow = np.array([30, 255, 255])
        yellow_hsv_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
        
        yellow_hsv_mask = cv2.morphologyEx(yellow_hsv_mask, cv2.MORPH_CLOSE, kernel_large)
        yellow_hsv_mask = cv2.morphologyEx(yellow_hsv_mask, cv2.MORPH_OPEN, kernel_small)
        
        yellow_contours, _ = cv2.findContours(yellow_hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_yellow_area = 0
        best_yellow_cnt = None
        
        if yellow_contours:
            best_yellow_cnt = max(yellow_contours, key=cv2.contourArea)
            max_yellow_area = cv2.contourArea(best_yellow_cnt)

        leaf_state = "NONE"
        leaf_mask = np.zeros((256, 256), dtype=np.uint8)
        contour_bound = np.zeros((256, 256), dtype=np.uint8)
        min_area_required = total_pixels * 0.05 

        if max_green_area >= max_yellow_area and max_green_area >= min_area_required:
            leaf_state = "GREEN"
            cv2.drawContours(contour_bound, [best_green_cnt], -1, 255, -1)
            leaf_mask = cv2.bitwise_and(green_hsv_mask, contour_bound)
            
        elif max_yellow_area > max_green_area and max_yellow_area >= min_area_required:
            leaf_state = "YELLOW_BROWN"
            cv2.drawContours(contour_bound, [best_yellow_cnt], -1, 255, -1)
            leaf_mask = cv2.bitwise_and(yellow_hsv_mask, contour_bound)

        masked_bgr = cv2.bitwise_and(frame_256, frame_256, mask=leaf_mask)
        masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

        # ==========================================
        # 310(H) x 235(W) SAFE-ZONE PANEL GENERATION
        # ==========================================
        panel = np.zeros((310, 235, 3), dtype=np.uint8)
        
        cv2.putText(panel, f"[{camera_id}]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        stats = {
            "status": "No Target Found", "avg_vnir": 0.0, "avg_g": 0.0, "ratio": 0.0,
            "ready": False, "vs_baseline": 0.0, "vs_prev_check": 0.0, "recent5_vs_global": 0.0
        }

        vnir_size = 160
        vnir_bgr = np.zeros((vnir_size, vnir_size, 3), dtype=np.uint8) 

        if leaf_state == "GREEN":
            pil_image = Image.fromarray(masked_rgb)
            vnir_result = self.engine.predict(pil_image)
            vnir_arr = np.array(vnir_result).astype(np.float32)
            
            stats = analyzer.analyze_and_log(masked_rgb, vnir_arr, leaf_mask)
            
            vnir_bgr = cv2.cvtColor(vnir_arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            vnir_bgr = cv2.resize(vnir_bgr, (vnir_size, vnir_size))
            
            status_color = (0, 0, 255) if "ALERT" in stats["status"] else (0, 255, 0)
            
            cv2.putText(panel, stats["status"], (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            cv2.putText(panel, f"VNIR: {stats.get('avg_vnir', 0):.1f} | Green: {stats.get('avg_g', 0):.1f}", (10, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(panel, f"Cur Ratio: {stats.get('ratio', 0):.3f}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if stats.get("ready"):
                def get_color(val): return (0, 255, 0) if val >= 0 else (0, 0, 255)
                cv2.putText(panel, "--- TEMPORAL TRENDS ---", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
                
                base_val = stats['vs_baseline']
                cv2.putText(panel, "Vs Baseline:", (10, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                if isinstance(base_val, str):
                    cv2.putText(panel, base_val, (135, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.putText(panel, f"{base_val:+.1f}%", (135, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.4, get_color(base_val), 1)

                prev_val = stats['vs_prev_check']
                cv2.putText(panel, "Vs Prev 5:", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                if isinstance(prev_val, str):
                    cv2.putText(panel, prev_val, (135, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.putText(panel, f"{prev_val:+.1f}%", (135, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, get_color(prev_val), 1)

                glob_val = stats['recent5_vs_global']
                cv2.putText(panel, "Recent vs Glob:", (10, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                if isinstance(glob_val, str):
                    cv2.putText(panel, glob_val, (135, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                else:
                    cv2.putText(panel, f"{glob_val:+.1f}%", (135, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.4, get_color(glob_val), 1)
            else:
                cv2.putText(panel, "Calibrating baseline (Need 5)...", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        elif leaf_state == "YELLOW_BROWN":
            cv2.putText(vnir_bgr, "AI BYPASSED", (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
            stats["status"] = "CRITICAL: Visual Stress"
            stats["vs_baseline"] = "CRITICAL!"
            cv2.putText(panel, stats["status"], (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        else: 
            cv2.putText(vnir_bgr, "NO TARGET", (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
            cv2.putText(panel, stats["status"], (10, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)

        panel[30:190, 37:197] = vnir_bgr

        # 1. Save the image to disk first so the Notifier can find it
        filepath = None
        if save_image:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filepath = os.path.join(self.output_dir, f"{camera_id.replace(' ', '_')}_{timestamp}.jpg")
            cv2.imwrite(filepath, panel)

        # 2. Dispatch WhatsApp Alert with the generated filepath
        if "ALERT" in stats["status"] or "CRITICAL" in stats["status"]:
            current_time = time.time()
            last_alert_time = self.alert_cooldowns.get(camera_id, 0)
            if current_time - last_alert_time > self.ALERT_COOLDOWN_SECONDS:
                print(f"⚠️ Stress detected on {camera_id}! Dispatching WhatsApp image alert...")
                threading.Thread(
                    target=self.notifier.send_stress_alert, 
                    args=(camera_id, stats["status"], stats.get("vs_baseline", 0.0), filepath),
                    daemon=True
                ).start()
                self.alert_cooldowns[camera_id] = current_time

        return panel, stats

if __name__ == "__main__":
    print("🚀 Running Processor Multi in Standalone Folder Demo Mode...")
    
    processor = ImageProcessorMulti(model_path="ThanalModel.onnx", output_dir="demo_outputs")
    cam_name = "Demo"
    processor.clear_history(cam_name) 
    
    input_folder = "demo_inputs"
    os.makedirs(input_folder, exist_ok=True)
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder) 
        if f.lower().endswith(valid_exts)
    ]
    
    if not images:
        print(f"⚠️ Add images to '{input_folder}' to test the demo.")
    else:
        for img_path in sorted(images): 
            print(f"📸 Processing: {img_path}")
            frame = cv2.imread(img_path)
            
            if frame is None:
                continue
                
            dashboard, stats = processor.process_frame(frame, camera_id=cam_name, save_image=True)
            
            cv2.imshow("Thanal Dashboard Viewer", dashboard)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()