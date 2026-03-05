# 🚦 Urban Traffic Multi-Stage Detection and Analysis

This project is an AI-powered video monitoring system for urban and pedestrian traffic. It transforms standard cameras into smart sensors capable of analyzing the environment in real time, detecting not only vehicles but also attributes like color, license plates, pedestrian gender, and pet breeds.

## ✨ Key Features

* **Multi-Class Detection and Tracking:** Simultaneous identification of multiple instances: vehicles (cars, trucks, buses), pedestrians, and animals, under various lighting conditions. The BoT-SORT algorithm is used to associate a unique ID to each object, transforming independent detections into coherent trajectories to prevent multiple counting.
* **Vehicle Analysis (Color & LPR):** Color determination is handled using the K-Means Clustering algorithm on histograms in the HSV color space. License plates are extracted using image processing techniques (Upscaling, Sharpening) and the EasyOCR network.
* **Pedestrian Analysis:** Estimating the gender of detected persons using a Convolutional Neural Network (CNN) based on the GoogleNet/Caffe architecture.
* **Pet Identification:** Classifying the breed of animals using the pre-trained MobileNetV2 architecture, applying validation filters to eliminate erroneous classifications.
* **Data Persistence:** Structured storage of detections in an SQLite relational database, using a debouncing logic to write data at regular intervals and optimize storage space.

## 🧠 System Architecture (Multi-Stage Detection)

The project implements the "Divide et Impera" (Divide and Conquer) algorithmic paradigm to balance both precision and the performance required for real-time execution. The data pipeline is organized sequentially:

1.  **Stage 1 (Global Detection):** The YOLOv11s main neural network scans the entire video frame to generate Bounding Boxes and identify the primary class.
2.  **ROI (Region of Interest) Extraction:** Based on the provided coordinates, the system performs a cropping operation and applies spatial filtering to remove "noisy" data.
3.  **Stage 2 (Local Analysis):** The cropped images are distributed to specialized analysis modules for each type of attribute (K-Means, EasyOCR, MobileNetV2, etc.).
4.  **Aggregation and Persistence:** Results are condensed into a single data object and saved into the database.



## 🛠️ Technologies and Frameworks Used

* **Language:** Python 3.10
* **Computer Vision:** OpenCV (cv2)
* **Object Detection & Tracking:** Ultralytics YOLO framework implementing YOLOv11 (the `yolo11s.pt` model)
* **Deep Learning Backend:** PyTorch & TorchVision (CUDA acceleration supported)
* **OCR (Optical Character Recognition):** EasyOCR
* **Database & ORM:** SQLite & SQLAlchemy

## 📸 Visual Demonstration
<img width="953" height="422" alt="image" src="https://github.com/user-attachments/assets/4d89104f-36d2-4202-9b68-d68f6d775662" />
<img width="380" height="361" alt="image" src="https://github.com/user-attachments/assets/64abf123-bbdb-43c8-99b1-813ae0b34a10" />
<img width="989" height="567" alt="image" src="https://github.com/user-attachments/assets/15cd590e-fae7-4524-9e0f-f5bf3e31f0d6" />
<img width="827" height="914" alt="image" src="https://github.com/user-attachments/assets/94c9d1b4-171c-470d-8a04-66c54f8dd23d" />
<img width="974" height="116" alt="image" src="https://github.com/user-attachments/assets/418c3d08-f468-41ce-b3db-091524268433" />





