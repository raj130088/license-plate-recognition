# 🚗 License Plate Recognition System

A computer vision project that detects vehicles and extracts license plate text from video using deep learning and OCR.

---

## 📌 Overview

This project performs **automatic license plate recognition (ALPR)** on video input. It uses object detection, tracking, and OCR to:

* Detect license plates in each frame
* Track plates across frames
* Extract text using OCR
* Stabilize predictions using temporal consistency

---

## ✨ Features

* 🔍 **Real-time license plate detection** using YOLO
* 🎯 **Object tracking** with BoT-SORT
* 🔤 **OCR-based text extraction** using fast-plate-ocr
* 🧠 **Text stabilization** using voting & similarity matching
* 🎥 **Annotated video output** with bounding boxes and plate text

---

## 🛠️ Tech Stack

* **Python**
* **OpenCV**
* **Ultralytics YOLO**
* **fast-plate-ocr**
* **NumPy**
* **tqdm**

---

## 📂 Project Structure

```
license-plate-recognition/
│── main_script.py          # Main pipeline
│── utility.py              # OCR + stabilization logic
│── requirements.txt
│── saved_models/
│   ├── license_plate_best.pt
│   └── ...
```

---

## ⚙️ How It Works

1. **Video Input**

   * Reads frames from input video

2. **Detection & Tracking**

   * YOLO detects license plates
   * BoT-SORT assigns unique IDs to each plate

3. **OCR Processing**

   * Cropped plates passed to OCR model
   * Text cleaned and validated using regex

4. **Text Stabilization**

   * Maintains history of predictions
   * Uses similarity grouping + voting
   * Outputs most stable plate number

5. **Visualization**

   * Draws bounding boxes
   * Displays zoomed plate
   * Overlays stabilized text

---

## ▶️ Demo

### 📥 Input Video

[Watch Input Video]([https://drive.google.com/file/d/121OnrtPOnqlY88N4whB6I0JqN30rd8q1/view?usp=drive_link](https://drive.google.com/file/d/121OnrtPOnqlY88N4whB6I0JqN30rd8q1/view?usp=drivesdk))

### 📤 Output Video

[Watch Output Video]([https://drive.google.com/file/d/1t_ufDwxa9Dd1Fj5G8UrvCleALIfhaFBb/view?usp=drive_link](https://drive.google.com/file/d/1t_ufDwxa9Dd1Fj5G8UrvCleALIfhaFBb/view?usp=drivesdk))

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python main_script.py
```

---

## ⚠️ Notes

* Large files (videos, models) are not included in this repo
* Download them separately and place them in the correct directories

---

## 🧠 Challenges Faced

* Handling OCR inaccuracies due to motion blur and lighting
* Stabilizing predictions across frames
* Managing large video/model files for GitHub

---

## 🔮 Future Improvements

* Deploy as a web app (Streamlit / Flask)
* Improve OCR accuracy with custom training
* Add support for multiple plate formats
* Optimize for real-time performance (GPU)

---

## 🙌 Acknowledgements

* Ultralytics YOLO
* fast-plate-ocr
* OpenCV

