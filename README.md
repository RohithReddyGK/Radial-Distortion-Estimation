# Computer Vision - Radial Distortion Estimation & Correction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![React](https://img.shields.io/badge/React-18.2.0-blue)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3.5.3-blue)
![Flask](https://img.shields.io/badge/Flask-2.3.3-lightgrey)
![Render](https://img.shields.io/badge/Deployment-Render-blue)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-blue)
[![Live Demo](https://img.shields.io/badge/Live-Demo-green)](https://radial-distortion-estimator.vercel.app/)

---

## 🧠 Project Overview
This project demonstrates **Camera Radial Distortion Estimation & Correction** using **Computer Vision techniques**.  
It uses OpenCV’s calibration pipeline to detect checkerboard corners, estimate intrinsic and distortion parameters, and visualize distortion correction.

Users can:
- Upload a **checkerboard or planar grid image**
- Specify **inner corners (cols × rows)**
- Run calibration directly from the web interface
- **View each step visually**
- **Download the full calibration report as a PDF**
- **Open and test the notebook in Google Colab**

---

## 🧮 How it Works
| Step       | Description                                                     |
| ---------- | --------------------------------------------------------------- |
| **Step 1** | Load and preprocess the uploaded checkerboard image             |
| **Step 2** | Detect corners using `cv2.findChessboardCorners()`              |
| **Step 3** | Estimate initial camera parameters with `cv2.calibrateCamera()` |
| **Step 4** | Perform RANSAC-based inlier selection                           |
| **Step 5** | Refine calibration using inliers only                           |
| **Step 6** | Undistort the input image using refined parameters              |
| **Step 7** | Compute reprojection residuals & summary metrics                |

---

## 🧩 Tech Stack

| Layer | Technologies |
|-------|---------------|
| **Frontend** | React + Vite + Tailwind CSS |
| **Backend** | Flask + OpenCV + NumPy |
| **Utilities** | html2pdf.js, Base64 encoding for images |
| **Deployment** | Vercel (Frontend) + Render (Backend) |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/RohithReddyGK/Radial-Distortion-Estimation.git
cd Radial-Distortion-Estimation

# Create virtual environment
python -m venv vision
vision\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

### FrontEnd
```bash
cd FrontEnd
npm install
npm run dev
```

---

## 🧾 PDF & Colab Integration

After successful processing:
- Click “**Download PDF Report**” to export a detailed summary
- Click “**Run in Google Colab**” to explore the notebook interactively

---

## Deployment

### Backend
```bash
Install Gunicorn: pip install gunicorn
Add to requirements.txt

Start command in Render:
gunicorn app:app --bind 0.0.0.0$PORT
```

### Frontend
```bash
Set VITE_BACKEND_URL in Vercel environment variables.
Deploy using npm run build or Vercel’s automatic deployment.
```

---

## 🙋‍♂️ Author

**Rohith Reddy.G.K**  
🔗 [GitHub Profile](https://github.com/RohithReddyGK)  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/rohithreddygk)

---

### 🌟 **If you like this project, give it a ⭐ **
