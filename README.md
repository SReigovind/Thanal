# 🌿 ThanalEdge

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Edge AI](https://img.shields.io/badge/Edge-AI-green)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![Use Case](https://img.shields.io/badge/Use--Case-Plant%20Health%20Monitoring-brightgreen)

**ThanalEdge** is an edge-native plant health intelligence system that combines computer vision and VNIR (Virtual NIR) inference to detect early signs of crop stress in real time.

It is designed for **low-power edge environments** (e.g., Raspberry Pi, field kiosks) and provides **continuous monitoring, temporal analysis, and instant alerts**.

---

## 🌱 Core Idea

Plants reflect light differently when stressed—especially in the **near-infrared spectrum**.

ThanalEdge:

* Synthesizes VNIR data from RGB images using an AI model
* Tracks changes in plant reflectance over time
* Detects **invisible early stress signals before visible symptoms appear**

---

## 🧠 System Architecture

```id="arch01"
Camera Input → Leaf Segmentation → VNIR Inference → Health Analysis → Alert System
```

### Pipeline Breakdown

1. **Image Acquisition**

   * Captured from USB or IP cameras via an IoT hub

2. **Leaf Segmentation**

   * HSV-based detection of:

     * 🌿 Healthy (green)
     * 🍂 Stressed (yellow/brown)

3. **VNIR Inference Engine**

   * ONNX model converts RGB → VNIR
   * Optimized for CPU-only edge devices

4. **Health Analyzer**

   * Computes:

     * Average green intensity
     * Average VNIR intensity
     * VNIR/Green ratio
   * Maintains historical trends per camera

5. **Decision Engine**

   * Evaluates deviations from baseline
   * Detects early stress conditions

6. **Notification Layer**

   * Sends WhatsApp alerts with optional image snapshots

---

## 📊 Health Intelligence Model

ThanalEdge does not rely on a single frame—it builds **temporal understanding**.

### Key Metrics

* **VNIR / Green Ratio**
* **Baseline (first 5 scans)**
* **Rolling averages (last 5 scans)**
* **Global trend comparison**

### Decision Logic

* 📉 Significant drop vs baseline → `ALERT: STRESS`
* 🍂 Yellow/Brown detection → `CRITICAL: Visual Stress`
* 📈 Stable trends → `Healthy Tracking`
* ⏳ Initial phase → `Calibrating`

---

## 🖥️ Edge Dashboard

Each processed frame produces a compact **310×235 monitoring panel** containing:

* VNIR visualization map
* Current health status
* Numerical metrics
* Temporal trend indicators

This enables real-time monitoring on **low-resolution kiosk displays**.

---

## 📡 Multi-Camera TDM System

ThanalEdge supports multiple cameras using a **Time Division Multiplexing (TDM)** approach:

* Alternates between camera streams at fixed intervals
* Maintains independent health histories per camera
* Displays a unified dual-panel dashboard

---

## ⚠️ Smart Alerting

* Triggered on:

  * VNIR-based stress detection
  * Visual stress (yellow/brown leaves)
* Includes:

  * Location (camera ID)
  * Stress severity
  * % deviation from baseline
  * Optional VNIR image snapshot
* Built-in cooldown system to prevent alert spam

---

## 🧾 Persistent Tracking

Each camera maintains its own:

* 📄 CSV log file
* 📈 Historical ratios
* 🧠 Independent baseline calibration

This allows long-term monitoring and trend analysis across multiple zones.

---

## 🧩 Module Overview

| Module             | Responsibility                     |
| ------------------ | ---------------------------------- |
| `processor.py`     | Core pipeline orchestration        |
| `inference.py`     | ONNX VNIR model execution          |
| `analyzer.py`      | Temporal health tracking & logging |
| `notifier.py`      | WhatsApp alert system              |
| `tdm.py`           | Multi-camera edge dashboard        |

---

## ⚡ Design Philosophy

* **Edge-first** → No cloud dependency required
* **Lightweight** → Optimized for CPU + low RAM
* **Real-time** → Instant processing & alerts
* **Temporal-aware** → Focus on trends, not snapshots
* **Fail-safe** → Works even with partial/noisy inputs

---

## 🌍 Use Cases

* Smart agriculture 🌾
* Greenhouse monitoring 🌱
* Plantation health tracking 🌴
* Research & phenotyping 🔬
* Remote farm surveillance 📡

---

## 🔮 Future Scope

* 🧠 Improved VNIR model with larger, more diverse training data
* 🌐 Integration with IoT sensors for stress cause identification
* 🤖 Automated responses (irrigation, nutrients, environment control)

---