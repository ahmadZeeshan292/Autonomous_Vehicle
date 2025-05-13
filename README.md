# 🛣️ Autonomous Vehicle: Lane Detection, Object Detection & Directional Control

![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-orange)

An end-to-end classical computer vision-based autonomous driving system that performs **lane detection**, **dynamic object detection**, and **HUD-based directional feedback** using motion and color segmentation all in real time.

---

## 📑 Table of Contents

- [🔍 Project Overview](#-project-overview)
- [📊 System Architecture](#-system-architecture)
- [🧠 Algorithms Used](#-algorithms-used)
- [🚗 Pipeline Breakdown](#-pipeline-breakdown)
  - [Lane Detection](#lane-detection)
  - [Object Detection](#object-detection)
  - [Directional Control](#directional-control)
- [⚙️ Performance Optimization](#️-performance-optimization)
- [🧪 Demo and Testing](#-demo-and-testing)

---

## 🔍 Project Overview

This project simulates the behavior of an autonomous vehicle in off-road or controlled environments by integrating three key components:

- **Lane Detection** using bird's eye view transformations and color thresholding.
- **Object Detection** through motion vector tracking and clustering.
- **HUD-based Directional Control** for real-time driver assistance based on road curvature and detected obstacles.

---

## 📊 System Architecture

The system follows a dual-pipeline architecture:

- **Left pipeline** for lane detection and feedback generation.
- **Right pipeline** for motion-based object detection and ROI analysis.

Both systems communicate with a central display system (HUD) to render directional cues and safety warnings.

![Pipeline Flowchart](Pipeline_Flowchart.jpg)

---

## 🧠 Algorithms Used

| Component         | Algorithm(s) Used                                                   |
|------------------|----------------------------------------------------------------------|
| Lane Detection    | Perspective Transform, Thresholding, Polynomial Regression (2nd order) |
| Object Detection  | Shi-Tomasi Corner Detection, Lucas-Kanade Optical Flow, DBSCAN       |
| Control Logic     | Curvature Estimation, ROI Intersection Checks                        |

---

## 🚗 Pipeline Breakdown

### Lane Detection

- Convert incoming frame to bird's eye view.
- Apply color thresholding to isolate white and yellow lane lines.
- Reproject to front view and check for lane presence.
- If lanes are detected, fit a 2nd order polynomial and estimate lane curvature.
- Display directional feedback on HUD based on curvature and lane alignment.

### Object Detection

- Detect motion points using **Shi-Tomasi**.
- Track motion vectors using **Lucas-Kanade Optical Flow**.
- Apply **DBSCAN clustering** to group similar motion patterns.
- Generate boundary boxes and highlight any within the region of interest (ROI).
- Display safety warnings on HUD if ROI contains dynamic objects.

### Directional Control

- If lanes are detected: provide curvature-based directional arrows.
- If objects in ROI: overlay real-time warnings for safe navigation.

## ⚙️ Performance Optimization

To enable faster and near real-time processing, the code introduces a **scaling factor**. This factor scales down:

- The **input video resolution** for lighter processing.
- The **features (motion points, lane masks, etc.)**, ensuring operations like optical flow and polynomial fitting remain efficient.

This design enables the system to be portable and deployable on lower-end hardware or embedded devices.

---

## 🧪 Demo and Testing

You can see the complete working of the autonomous vehicle system in the demo video below:

🎥 **[▶ Watch Demo Video](Output_video.mp4)**




