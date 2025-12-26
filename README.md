# FaceAI - Hệ thống Điểm danh & Chống Giả mạo (Liveness Detection)

Đồ án môn học: Xây dựng hệ thống nhận diện khuôn mặt có khả năng phát hiện người thật/giả.

## 🌟 Chức năng chính
1. **Face Recognition:** Nhận diện khuôn mặt và xác định danh tính.
2. **Liveness Detection:** Chống giả mạo bằng cách phát hiện nháy mắt (Blink Detection) qua Video.

## 🛠️ Cài đặt môi trường
Dự án chạy trên Python 3.12. Để cài đặt các thư viện cần thiết:

```bash
# 1. Tạo môi trường ảo (Khuyên dùng)
python3 -m venv venv
source venv/bin/activate  # Trên Linux/Mac
# venv\Scripts\activate   # Trên Windows

# 2. Cài đặt thư viện
pip install mediapipe opencv-python