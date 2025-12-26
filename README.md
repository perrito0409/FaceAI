# FaceAI - Hệ thống Nhận diện Khuôn mặt & Liveness Detection

Dự án Demo nhận diện khuôn mặt và kiểm tra thực thể sống (Liveness Detection) để xác minh danh tính, phát hiện giả mạo khuôn mặt qua video.

## 📂 Cấu trúc dự án
* **`face_engine.py`**: Module chính xử lý nhận diện khuôn mặt (so khớp vector 512 chiều).
* **`liveness.py`**: Module kiểm tra Liveness (phát hiện chớp mắt/cử động từ video).
* **`test_blink.mp4`**: Video mẫu dùng để demo tính năng Liveness.
* **`requirements.txt`**: Danh sách các thư viện cần thiết.

## ⚙️ Cài đặt môi trường

1. Yêu cầu đã cài đặt Python (3.8 trở lên).
2. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt