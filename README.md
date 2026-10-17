# 🛡️ SafeGuard Vision AI: Edge-Accelerated HSE Safety, Fall Detection & Active Threat Intelligence

[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-GPU%20Accelerated-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Custom%20Trained-00FFFF)](https://ultralytics.com/)
[![Edge AI](https://img.shields.io/badge/Deployment-Edge%20%26%20CCTV-green)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-throughput edge computer vision framework designed for industrial health & safety (HSE), personal protective equipment (PPE) compliance, fall event alerts, and physical security threat detection.

---

## 📸 System Capabilities & Detections

### 1. Construction Site PPE & Helmet Live Compliance
Simultaneous multi-worker detection tracking Helmet and Safety Vest compliance with live counter overlays.
![PPE & Helmet Compliance Live Count](assets/ppe_helmet_compliance_live_count.png)

### 2. Real-Time Human Fall Detection
Instant alert triggering upon rapid vertical velocity and aspect-ratio collapse (**98% accuracy**).
![Human Fall Detection Alert](assets/human_fall_detection_alert.png)

### 3. Active Threat & Weapon Identification
Sub-second detection of firearms and bladed weapons in indoor and retail environments.
![Weapon Detection Indoor](assets/weapon_threat_detection_indoor.png)
![Weapon Detection Retail](assets/weapon_threat_detection_retail.png)

### 4. Dwell Time & Loitering Alerts
Zone intrusion and loitering tracking with duration counters.
![Loitering Detection Alert](assets/loitering_alert.png)

### 5. Crowd Density Estimation
Spatial clustering algorithm estimating pedestrian volume and gathering alerts.
![Crowd Density Estimation](assets/crowd_density_estimation.png)

---

## 🔬 Core Capabilities

### 1. PPE Spatial Intersection Attribution
Rather than just detecting loose helmets, the system implements an IoU and centroid containment algorithm (`is_ppe_on_person`):
$$\text{Intersection}(B_{\text{person}}, B_{\text{gear}}) > 0 \quad \land \quad C_{\text{gear}} \in B_{\text{person}}$$
Accurately maps helmets and vests to unique person track IDs to generate compliant vs non-compliant audit numbers.

### 2. Human Fall Event Trigger (98% Accuracy)
Monitors bounding box bounding geometry:
- Aspect ratio inversion: $\frac{\text{Width}}{\text{Height}} > 1.2$
- Vertical centroid acceleration: $\frac{\Delta C_y}{\Delta t} > V_{\text{thresh}}$
- Ground proximity validation to eliminate false positives from sitting or tying shoelaces.

### 3. Edge-Optimized ONNX Runtime Inference
Optimized with FP16 precision via ONNX Runtime GPU, achieving sub-20ms per-frame latency on edge devices.

---

## 🛠️ Quickstart

```bash
# Clone the repository
git clone https://github.com/mh-masum6875/workplace-safety-edge-vision.git
cd workplace-safety-edge-vision

# Install dependencies
pip install -r requirements.txt
```

### Running PPE Compliance Detection:
```bash
python src/ppe_compliance_tracker.py --source data/construction_site.mp4
```

### Running Fall Detection (ONNX GPU):
```bash
python src/fall_detection_onnx.py --source data/workplace_feed.mp4 --weights models/fall_detection.onnx
```

### Running Weapon Detection:
```bash
python src/weapon_detection_onnx.py --source data/security_feed.mp4 --weights models/weapon_detection.onnx
```

---

## 📄 License
This project is licensed under the MIT License.
