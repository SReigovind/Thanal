import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ==========================================
# 1. DEFINE THE MODEL ARCHITECTURE
# (Must match the training code EXACTLY)
# ==========================================

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.e1 = conv_block(3, 64)
        self.pool = nn.MaxPool2d(2)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)

        self.b = conv_block(512, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.d1 = conv_block(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.d2 = conv_block(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.d3 = conv_block(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.d4 = conv_block(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        c1 = self.e1(x)
        p1 = self.pool(c1)
        c2 = self.e2(p1)
        p2 = self.pool(c2)
        c3 = self.e3(p2)
        p3 = self.pool(c3)
        c4 = self.e4(p3)
        p4 = self.pool(c4)

        b = self.b(p4)

        u1 = self.up1(b)
        x4 = self.att1(g=u1, x=c4)
        cat1 = torch.cat((u1, x4), dim=1)
        ud1 = self.d1(cat1)

        u2 = self.up2(ud1)
        x3 = self.att2(g=u2, x=c3)
        cat2 = torch.cat((u2, x3), dim=1)
        ud2 = self.d2(cat2)

        u3 = self.up3(ud2)
        x2 = self.att3(g=u3, x=c2)
        cat3 = torch.cat((u3, x2), dim=1)
        ud3 = self.d3(cat3)

        u4 = self.up4(ud3)
        x1 = self.att4(g=u4, x=c1)
        cat4 = torch.cat((u4, x1), dim=1)
        ud4 = self.d4(cat4)

        return self.out(ud4)

# ==========================================
# 2. SETUP MODEL & INFERENCE
# ==========================================

# Use CPU for the demo (easier compatibility)
device = torch.device("cpu")
model_path = "ThanalModel.pth"

print(f"Loading model from {model_path}...")
model = AttentionUNet().to(device)

# Load weights (map_location ensures it loads on CPU even if trained on GPU)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Ensure 'model.pth' matches the AttentionUNet architecture.")

model.eval()

# Define Preprocessing (Must match training size)
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define the Prediction Function
def predict_vnir(input_img):
    if input_img is None:
        return None
        
    # 1. Preprocess
    # Convert PIL to Tensor
    input_tensor = img_transform(input_img).unsqueeze(0).to(device)
    
    # 2. Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # 3. Post-process
    # Clamp values to 0-1 range
    output_tensor = torch.clamp(output_tensor, 0, 1)
    
    # Remove batch dimension and convert to Numpy
    # Squeeze reduces (1, 1, 256, 256) to (256, 256)
    output_array = output_tensor.squeeze().cpu().numpy()
    
    # Convert to PIL Image (L mode for Grayscale)
    # Multiply by 255 to get back to 8-bit integer range
    output_image = Image.fromarray((output_array * 255).astype(np.uint8), mode='L')
    
    return output_image

# ==========================================
# 3. BUILD GRADIO INTERFACE
# ==========================================

with gr.Blocks() as demo:
    gr.Markdown("# 🌱 Virtual NIR Estimation Demo")
    gr.Markdown("Upload a standard RGB image of a leaf, and this AI will generate its corresponding Near-Infrared (NIR) view.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input RGB Image")
            run_btn = gr.Button("Generate NIR", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Estimated Virtual NIR")
            
    run_btn.click(fn=predict_vnir, inputs=input_image, outputs=output_image)
    
    gr.Markdown("### How it works")
    gr.Markdown("This model uses an **Attention U-Net** architecture trained on paired RGB-NIR imagery. It analyzes leaf texture and color to predict structural reflectance in the 700nm-1000nm band.")

# Launch the app
if __name__ == "__main__":
    demo.launch()