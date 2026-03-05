# [cite_start]🚦 Urban Traffic Multi-Stage Detection & Analysis [cite: 4]

[cite_start]This project represents an advanced software system for real-time video monitoring and analysis of urban and pedestrian traffic, using Artificial Intelligence (AI) and Computer Vision technologies[cite: 8]. [cite_start]Developed in the context of accelerated urbanization and the transition to the "Smart City" concept, the system transforms an ordinary video camera into a sensor capable of understanding the surrounding environment[cite: 6, 23].

[cite_start]Unlike conventional systems that are limited to counting vehicles, this solution performs a more detailed analysis[cite: 9]. [cite_start]The system is capable of automatically extracting secondary attributes (color, license plate, pedestrian gender, pet breed)[cite: 10].

## ✨ Key Features

- [cite_start]**Multi-Class Detection and Tracking:** Simultaneous identification of multiple instances: vehicles (cars, trucks, buses), pedestrians, and animals, under various lighting conditions[cite: 14]. [cite_start]The BoT-SORT algorithm is used to associate a unique ID to each object, transforming independent detections into coherent trajectories to prevent multiple counting[cite: 35, 36].
- [cite_start]**Vehicle Analysis (Color & LPR):** Color determination is handled using the K-Means Clustering algorithm on histograms in the HSV color space[cite: 43, 77]. [cite_start]License plates are extracted using image processing techniques (Upscaling, Sharpening) and the EasyOCR network[cite: 44, 98, 99].
- [cite_start]**Pedestrian Analysis:** Estimating the gender of detected persons using a Convolutional Neural Network (CNN) based on the GoogleNet/Caffe architecture[cite: 45, 46].
- [cite_start]**Pet Identification:** Classifying the breed of animals using the pre-trained MobileNetV2 architecture, applying validation filters to eliminate erroneous classifications[cite: 47, 48].
- [cite_start]**Data Persistence:** Structured storage of detections in an SQLite relational database, using a debouncing logic to write data at regular intervals and optimize storage space[cite: 17, 51, 52, 101].

## [cite_start]🧠 System Architecture (Multi-Stage Detection) [cite: 24, 25]

[cite_start]The project implements the "Divide et Impera" (Divide and Conquer) algorithmic paradigm to balance both precision and the performance required for real-time execution[cite: 12, 26]. [cite_start]The data pipeline is organized sequentially[cite: 30, 31]:

1. [cite_start]**Stage 1 (Global Detection):** The YOLOv11s main neural network scans the entire video frame to generate Bounding Boxes and identify the primary class[cite: 28, 33, 34].
2. [cite_start]**ROI (Region of Interest) Extraction:** Based on the provided coordinates, the system performs a cropping operation and applies spatial filtering to remove "noisy" data[cite: 37, 38, 39].
3. [cite_start]**Stage 2 (Local Analysis):** The cropped images are distributed to specialized analysis modules for each type of attribute (K-Means, EasyOCR, MobileNetV2, etc.)[cite: 29, 41].
4. [cite_start]**Aggregation and Persistence:** Results are condensed into a single data object and saved into the database[cite: 49, 50, 52].

_(Insert architecture flowchart here)_

## [cite_start]🛠️ Technologies and Frameworks Used [cite: 60]

- [cite_start]**Language:** Python 3.10 [cite: 55]
- [cite_start]**Computer Vision:** OpenCV (cv2) [cite: 62]
- [cite_start]**Object Detection & Tracking:** Ultralytics YOLO framework implementing YOLOv11 (the `yolo11s.pt` model) [cite: 61, 66]
- [cite_start]**Deep Learning Backend:** PyTorch & TorchVision, configured to use CUDA hardware acceleration for optimized inference times [cite: 63]
- [cite_start]**OCR (Optical Character Recognition):** EasyOCR [cite: 65]
- [cite_start]**Database & ORM:** SQLite & SQLAlchemy [cite: 64]

## 📸 Visual Demonstration
