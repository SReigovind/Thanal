import onnxruntime as ort
from PIL import Image
import numpy as np
import os

class VNIREngine:
    def __init__(self, model_path="ThanalModel.onnx", device="cpu"):
        print(f"⚙️ Initializing ONNX VNIR Engine on {device}...")
        
        if not os.path.exists(model_path):
            print(f"❌ ONNX file not found: {model_path}")
            return
            
        # Initialize ONNX Runtime Session (Optimized for CPU)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load the graph
        self.ort_session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        
        # Get input and output names dynamically from the graph
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

    def predict(self, pil_image):
        """
        Takes a PIL RGB Image, returns a PIL Grayscale VNIR Image
        Uses pure NumPy instead of torchvision to save RAM on the Pi.
        """
        # 1. Preprocess (Equivalent to Resize(256, 256) + ToTensor())
        img_resized = pil_image.resize((256, 256))
        
        # Convert to numpy array and scale to [0, 1]
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        
        # Change data layout from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        img_np = np.transpose(img_np, (2, 0, 1))
        
        # Add batch dimension to make it (1, 3, 256, 256)
        input_tensor = np.expand_dims(img_np, axis=0)
        
        # 2. Inference using ONNX Runtime
        outputs = self.ort_session.run([self.output_name], {self.input_name: input_tensor})
        output_tensor = outputs[0]
        
        # 3. Post-process
        output_clipped = np.clip(output_tensor, 0, 1)
        output_array = np.squeeze(output_clipped) # Remove batch and channel dims
        
        # Convert back to PIL Image
        vnir_image = Image.fromarray((output_array * 255).astype(np.uint8), mode='L')
        return vnir_image