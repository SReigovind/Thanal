import cv2
import time
import requests
import numpy as np
from processor import ImageProcessorMulti

# ===========================
# CONFIGURATION
# ===========================
HUB_IP = "10.11.2.238"  # 🛑 UPDATE THIS to your Mac/Hub's Wi-Fi IP

HUB_USB_URL = f"http://{HUB_IP}:5000/cam/usb"
HUB_IP_URL = f"http://{HUB_IP}:5000/cam/ip"

TDM_INTERVAL = 20 
TDM_OFFSET = 10   

CAM1_NAME = "USB Camera"
CAM2_NAME = "IP Camera"

def create_blank_panel(camera_id):
    """Creates a 310(H) x 235(W) placeholder panel for the safe zone."""
    panel = np.zeros((310, 235, 3), dtype=np.uint8) 
    cv2.putText(panel, f"Waiting for", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.putText(panel, f"{camera_id}...", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    return panel

def fetch_hub_frame(url, cam_name):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            image_array = np.array(bytearray(response.content), dtype=np.uint8)
            return True, cv2.imdecode(image_array, -1)
        else:
            print(f"⚠️ {cam_name} returned status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Hub network error for {cam_name}: {e}")
    return False, None

def main():
    print("📡 Initializing Wireless TDM Edge Kiosk...")
    processor = ImageProcessorMulti() 
    
    print(f"🧹 Resetting baseline histories for {CAM1_NAME} and {CAM2_NAME}...")
    processor.clear_history(CAM1_NAME)
    processor.clear_history(CAM2_NAME)
    
    print(f"🔗 Linking to IoT Hub at {HUB_IP}:5000...")
    
    # --- UI KIOSK SETUP ---
    cv2.namedWindow("Thanal UI", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Thanal UI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    start_time = time.time()
    last_usb_time = start_time - TDM_INTERVAL 
    last_ip_time = start_time - TDM_INTERVAL + TDM_OFFSET 

    dash_usb = create_blank_panel(CAM1_NAME)
    dash_ip = create_blank_panel(CAM2_NAME)
    
    print("\n⏳ Wireless TDM Schedule Started...")
    
    try:
        while True:
            current_time = time.time()
            
            # --- TDM SLOT 1: USB CAMERA ---
            if current_time - last_usb_time >= TDM_INTERVAL:
                print(f"[{time.strftime('%H:%M:%S')}] 📡 Fetching {CAM1_NAME} from Hub...")
                success, frame = fetch_hub_frame(HUB_USB_URL, CAM1_NAME)
                
                if success and frame is not None:
                    dash_usb, _ = processor.process_frame(frame, camera_id=CAM1_NAME, save_image=True)
                
                last_usb_time = current_time

            # --- TDM SLOT 2: IP CAMERA ---
            if current_time - last_ip_time >= TDM_INTERVAL:
                print(f"[{time.strftime('%H:%M:%S')}] 📡 Fetching {CAM2_NAME} from Hub...")
                success, frame = fetch_hub_frame(HUB_IP_URL, CAM2_NAME)
                
                if success and frame is not None:
                    dash_ip, _ = processor.process_frame(frame, camera_id=CAM2_NAME, save_image=True)
                    
                last_ip_time = current_time

            # --- RENDER UNIFIED KIOSK DASHBOARD ---
            # 1. Stack the two 235px wide panels horizontally to make a 470x310 viewport
            unified_dashboard = np.hstack((dash_usb, dash_ip))
            
            # 2. Wrap it in a 5px black border on all sides to make it 480x320 and bypass Pi overscan
            padded_dashboard = cv2.copyMakeBorder(
                unified_dashboard, 
                top=5, bottom=5, left=5, right=5, 
                borderType=cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]
            )
            
            cv2.imshow("Thanal UI", padded_dashboard)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Edge Kiosk System stopped gracefully.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()