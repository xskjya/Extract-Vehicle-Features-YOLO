# **Extract-Vehicle-Features-YOLO**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Object%20Detection-red.svg)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Supervision-Annotation%20%26%20Tracking-brightgreen.svg)](https://github.com/roboflow/supervision)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## **Overview**
This is a YOLO-based video vehicle speed estimation extraction based on cameras
Use cases include:
---

## **Demo Video**

Watch the full demonstration on YouTube:


---

## **Features**

* **Real-Time Vehicle Detection** using YOLO.
* **Accurate Multi-Object Tracking** with ByteTrack.
* **Vehicle Counting** using configurable line zones (In/Out counts).
* **Speed Estimation** using **Perspective Transformation**.
* Supports **custom YOLO models** and various camera perspectives.
---

## **Project Structure**

```
Vehicle-Speed-Estimation-and-Counting-YOLO-Supervision
│
├── data
│   ├── vehicles.mp4
│   ├── vehicles_output.mp4
│   ├── frame.png
│   └── annotated_frame.png
│
├── models
│   ├── yolov8n.pt
│   └── VisDrone_YOLO_x2.pt
│
├── src
│   ├── annotator.py
│   ├── speed_estimator.py
│   ├── view_transformer.py
│   └── __init__.py
│
├── config.py
├── main.py
├── requirements.txt
└── Vehicle_Speed_Estimation_main.ipynb
```

---

## **Installation**


## **How to Run**


---

## **Future Enhancements**

---

## **Acknowledgments**

* [YOLO by Ultralytics](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [Supervision by Roboflow](https://github.com/roboflow/supervision)
* [NumPy](https://numpy.org/)

---

