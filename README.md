# Project Thanal: Edge-Optimized VNIR Crop Monitoring 🌿
*An e-Yantra National Finals Project*

Edge-optimized deployment of Project Thanal, a low-cost, software-driven Virtual Near-Infrared (VNIR) crop monitoring system. It leverages an ONNX-accelerated vision pipeline and a multi-camera IoT hub architecture to detect early-stage plant stress on resource-constrained hardware.

## 📖 Project Overview
Thanal democratizes precision agriculture for small-to-medium farmers by utilizing standard RGB webcams to synthesize VNIR maps. This allows for the detection of invisible, early-stage physiological plant stress days before visible symptoms (like chlorosis) occur, eliminating the need for expensive multispectral hardware.

## ⚙️ How It Works (The Pipeline)
The system operates through a compute-aware, multi-stage pipeline designed specifically for edge devices:

1. **Dynamic Leaf Isolation:** Incoming frames pass through an HSV color-space cascade and morphological filtering to isolate true leaf tissue and reject background noise (dirt, shadows, equipment).
2. **Compute-Aware Routing:** * If severe visual chlorosis (yellowing/browning) is detected, the system flags a **Critical Alert** and intelligently bypasses the heavy AI inference to save edge compute cycles.
   * If the leaf appears visually healthy (green), the isolated tissue is routed to the AI engine.
3. **VNIR Synthesis:** The ONNX-optimized `UNet-Attention` model synthesizes a highly accurate Grayscale VNIR map from the isolated RGB input.
4. **Temporal Health Tracking:** The system calculates a VNIR-to-Green ratio and compares it against the plant's own rolling 5-scan historical baseline. This local normalization mitigates false flags caused by changing environmental lighting across different times of the day.

## 🏗️ System Architecture (Decoupled NVR)
To prevent network bottlenecks and thread-locking on the primary edge device (Raspberry Pi 4), Thanal utilizes a professional Network Video Recorder (NVR) / IoT Hub topology:

* **The IoT Video Gateway (Hub):** A centralized machine (PC/Mac) handles the messy hardware drivers, continuously clears physical USB buffers, and fetches IP camera streams. It acts as a middleman, serving the latest frames instantly via a lightweight HTTP Flask server.
* **The Edge Orchestrator (Pi 4):** The edge device requests frames from the Gateway over Wi-Fi using **Time-Division Multiplexing (TDM)**. It queries one camera feed every 20 seconds, staggered by a 10-second offset. This strict scheduling guarantees the Pi's CPU never attempts to run parallel AI inferences, preventing thermal throttling and RAM overflow.

## 🧠 Key Engineering Innovations
* **Graph-Optimized Inference:** The original 130MB PyTorch model was traced and exported to a static **ONNX** execution graph. The edge device runs entirely on `onnxruntime` (C++ backend), freeing up over 1GB of RAM by completely removing PyTorch dependencies.
* **Hardware Agnostic:** Because the Gateway hub abstracts the camera hardware into simple HTTP endpoints, the Edge node can process feeds from USB webcams, ESP32-Cams, or enterprise IP cameras simultaneously without altering the core inference code.