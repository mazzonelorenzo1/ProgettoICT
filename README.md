# 👁️ Smart Peephole - Intelligent Face Recognition System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![Dlib](https://img.shields.io/badge/Library-Dlib-orange)
![SphereFace](https://img.shields.io/badge/Model-SphereFace-red)

## 📑 Table of Contents
- [About the Project](#intro)
- [System Architecture & Logic](#architecture)
- [Installation](#install)
- [How to Run the System](#demo)
- [Performance & Benchmarks](#results)
- [Acknowledgments](#aknow)

---

<a id="intro"></a>
## 📖 About the Project
**Smart Peephole** is an end-to-end Face Recognition System designed for domestic use. The project ensures real-time access control by recognizing registered residents and immediately alerting them when unknown individuals are detected at the door.

Developed with privacy and edge-readiness in mind, the system operates entirely locally, storing biometric data on a local database without relying on external cloud infrastructure. 

**Key Features:**
*   **Real-world Interaction & Alerts:** Automatic Telegram Bot notifications with captured snapshots when an unknown user is detected.
*   **Web Deployment:** Lightweight Flask-based web application providing a user-friendly interface for continuous monitoring and new user enrollment.
*   **Secure Remote Access:** Uses `ngrok` to establish a secure HTTPS tunnel to the local server, allowing remote monitoring without exposing the local network.

---

<a id="architecture"></a>
## ⚙️ System Architecture & Logic

The recognition pipeline is decoupled into discrete stages to maximize performance on consumer hardware:

1.  **Face Detection & Alignment:** Utilizes **Dlib** to locate faces in the captured frames and perform geometric pixel-level alignment.
2.  **Feature Extraction:** A **SphereFace** neural network processes the aligned faces to extract high-dimensional, discriminative facial embeddings.
3.  **Identification & Thresholding:** The extracted embedding is compared against all locally stored user embeddings using a distance-based metric. If the similarity score exceeds a strict decision threshold, the identity is confirmed; otherwise, the subject is classified as *Unknown*.
4.  **Enrollment:** New users can be registered dynamically via the Web UI by capturing at least 4 photos to guarantee robustness against varying poses, illuminations, and facial expressions.

---

<a id="install"></a>
## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mazzonelorenzo1/Smart-Peephole.git
cd Smart-Peephole
```

2. **Create a virtual environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install the dependencies:**
Ensure you have CMake installed for building `dlib`.
```bash
pip install -r requirements.txt
```

---

<a id="demo"></a>
## 🖥️ How to Run the System

1. **Telegram & Ngrok Setup:**
   * Configure your Telegram Bot token in the application environment variables.
   * Start your `ngrok` agent to expose the local Flask port (default 5000):
     ```bash
     ngrok http 5000
     ```

2. **Start the Backend Server:**
   ```bash
   python app/app.py
   ```

3. **User Interaction (Client Side):**
   * Open the provided `ngrok` HTTPS link in any modern browser. No additional software is needed on the client device.
   * **Live Camera:** Click "Start camera" to capture frames. The system auto-identifies faces and updates the dashboard.
   * **Enrollment:** Use the UI to type a name, capture multiple photos, and register a new resident.

---

<a id="results"></a>
## 📊 Performance & Benchmarks

The system was rigorously benchmarked across consumer CPUs and GPUs to ensure edge-readiness.

*   **Benchmark 1 (End-to-End Latency):** From raw input to final identification (detection, alignment, embedding, verification).
    *   CPU processing (Intel Core Ultra 7 155H, Ryzen 5 9600x) is dominated by the Dlib face detection and alignment stages.
    *   GPU acceleration (RTX 5070) yields moderate end-to-end gains due to CPU-bound preprocessing steps.
*   **Benchmark 2 (SphereFace Forward Pass):** Isolating the neural network inference.
    *   The RTX 5070 GPU achieved a **4-5x speedup** compared to CPUs, dropping average latency to ~1-2 ms per embedding with high stability (low p50-p95 dispersion).
*   **Conclusion:** The pipeline is highly scalable and completely functional on CPU-only machines, perfect for single-image domestic authentication.

---

<a id="aknow"></a>
## 🏆 Acknowledgments
*   **Project By:** Chiara Curgu & Lorenzo Mazzone
*   **Course:** Intelligent Consumer Technologies (AA 2025/2026)
*   **Institution:** Università degli Studi di Milano-Bicocca
