import cv2
import time
import os
import requests  # We use requests to grab a static image
import numpy as np
from datetime import datetime
from PIL import Image

# Import our custom Engine
from inference import VNIREngine

# ===========================
# CONFIGURATION
# ===========================
# NOTE: Use /shot.jpg for a single clean image capture
CAMERA_URL = "http://10.26.1.132:8080/shot.jpg" 
MODEL_PATH = "ThanalModel.pth"
OUTPUT_FOLDER = "monitoring_logs"
SCAN_INTERVAL = 5  # Seconds between each capture

# ===========================
# SETUP
# ===========================
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize the Brain
engine = VNIREngine(model_path=MODEL_PATH, device="cpu")

print(f"📡 System Initialized. Target: {CAMERA_URL}")
print(f"✅ Scanning every {SCAN_INTERVAL} seconds...")

# ===========================
# MAIN LOOP
# ===========================
try:
    while True:
        try:
            # 1. Fetch Image from Phone (Snapshot method)
            # This is more stable than cv2.VideoCapture for IP cams
            response = requests.get(CAMERA_URL, timeout=5)
            
            if response.status_code == 200:
                # Convert bytes to Numpy Array
                image_array = np.array(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(image_array, -1)
                
                # 2. Convert for AI (OpenCV uses BGR, we need RGB)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                # 3. Generate VNIR (Using our Engine module)
                vnir_result = engine.predict(pil_image)

                # 4. Save Logic
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                vnir_filename = os.path.join(OUTPUT_FOLDER, f"vnir_{timestamp}.png")
                vnir_result.save(vnir_filename)
                
                print(f"[{timestamp}] 📸 Captured & Analyzed -> {vnir_filename}")

                # 5. Live Preview
                cv2.imshow("Real-Time Plant Monitor (RGB)", frame)
                vnir_cv = np.array(vnir_result)
                cv2.imshow("Virtual NIR Sensor", vnir_cv)
                
                # Handle Quit
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            else:
                print("⚠️ Failed to fetch image (Status Code Error)")

        except Exception as e:
            print(f"⚠️ Connection Error: {e}. Retrying...")

        # Wait Loop (Simple sleep is fine for snapshot method)
        time.sleep(SCAN_INTERVAL)

except KeyboardInterrupt:
    print("\n🛑 Manually Stopped.")

finally:
    cv2.destroyAllWindows()
    print("✅ System Shutdown Cleanly.")