import os
import base64
import requests
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

class WhatsAppNotifier:
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_whatsapp_number = os.getenv('TWILIO_FROM_NUMBER')
        self.to_whatsapp_number = os.getenv('TWILIO_TO_NUMBER')
        self.imgbb_api_key = os.getenv('IMGBB_API_KEY')
        
        if not all([self.account_sid, self.auth_token, self.from_whatsapp_number, self.to_whatsapp_number]):
            raise ValueError("❌ Missing Twilio credentials! Check your .env file.")
        if not self.imgbb_api_key:
            print("⚠️ Warning: IMGBB_API_KEY not found. Images will not be sent with alerts.")
            
        self.client = Client(self.account_sid, self.auth_token)

    def upload_to_imgbb(self, image_path):
        """Uploads a local image to ImgBB and returns the public direct URL."""
        if not self.imgbb_api_key or not image_path or not os.path.exists(image_path):
            return None
            
        try:
            with open(image_path, "rb") as file:
                url = "https://api.imgbb.com/1/upload"
                payload = {
                    "key": self.imgbb_api_key,
                    "image": base64.b64encode(file.read()),
                }
                res = requests.post(url, data=payload)
                if res.status_code == 200:
                    return res.json()['data']['url']
        except Exception as e:
            print(f"⚠️ Image upload failed: {e}")
        return None

    def send_stress_alert(self, camera_id, status, vs_baseline, image_path=None):
        """Sends a formatted WhatsApp alert, attaching the dashboard image if available."""
        if isinstance(vs_baseline, str):
            baseline_text = vs_baseline 
        else:
            baseline_text = f"{vs_baseline:+.1f}%" 

        message_body = (
            f"🚨 *Thanal Edge Alert* 🚨\n\n"
            f"📍 *Location:* {camera_id}\n"
            f"⚠️ *Status:* {status}\n"
            f"📉 *Drop vs Baseline:* {baseline_text}\n\n"
            f"Please review the attached VNIR synthesis map."
        )

        public_image_url = None
        if image_path:
            print("☁️ Uploading alert image to secure cloud...")
            public_image_url = self.upload_to_imgbb(image_path)

        kwargs = {
            "body": message_body,
            "from_": self.from_whatsapp_number,
            "to": self.to_whatsapp_number
        }
        
        if public_image_url:
            kwargs["media_url"] = [public_image_url]

        try:
            message = self.client.messages.create(**kwargs)
            print(f"📱 WhatsApp Alert sent successfully! SID: {message.sid}")
        except Exception as e:
            print(f"❌ Failed to send WhatsApp Alert: {e}")

if __name__ == "__main__":
    print("🧪 Testing Secure Twilio + ImgBB WhatsApp Integration...")
    try:
        notifier = WhatsAppNotifier()
        # Create a blank dummy image just to test the upload functionality
        import cv2
        import numpy as np
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite("test_upload.jpg", dummy_img)
        
        notifier.send_stress_alert("Test_Camera", "CRITICAL: Visually Stressed", "CRITICAL!", image_path="test_upload.jpg")
    except ValueError as e:
        print(e)