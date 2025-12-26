import cv2
import time

# --- CẤU HÌNH ---
VIDEO_SOURCE = "test_blink.mp4" 

# Tải bộ não có sẵn của OpenCV (Không cần internet, không cần mediapipe)
# Đây là thuật toán kinh điển để tìm Mặt và Mắt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(VIDEO_SOURCE)

blink_counter = 0
eyes_closed_frames = 0
is_eyes_open = True
color = (255, 0, 255) # Tím

print("Hệ thống OpenCV Native đang chạy...")

while True:
    success, img = cap.read()
    if not success:
        print("Hết video.")
        break
    
    # Chuyển sang ảnh xám để nhận diện nhanh hơn
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Tìm khuôn mặt
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Vẽ khung mặt
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Chỉ tìm mắt trong vùng khuôn mặt (nửa trên)
        roi_gray = gray[y:y+int(h/1.5), x:x+w]
        roi_color = img[y:y+int(h/1.5), x:x+w]
        
        # 2. Tìm mắt
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        
        # LOGIC NHÁY MẮT (Đơn giản hóa)
        # Nếu tìm thấy >= 1 mắt -> Mắt mở
        # Nếu tìm thấy khuôn mặt mà KHÔNG thấy mắt -> Nhắm mắt
        
        if len(eyes) >= 1:
            if not is_eyes_open: # Nếu trước đó đang nhắm mà giờ mở -> Đếm 1 cái
                blink_counter += 1
                is_eyes_open = True
                color = (0, 255, 0) # Xanh lá
            
            # Vẽ vòng tròn quanh mắt
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        else:
            # Không thấy mắt đâu -> Đang nhắm
            if is_eyes_open:
                is_eyes_open = False
                color = (0, 0, 255) # Đỏ

    # HIỆN KẾT QUẢ
    cv2.putText(img, f'Nhay mat: {blink_counter}', (30, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    if blink_counter >= 1:
        cv2.putText(img, "REAL USER (NGUOI THAT)", (30, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    else:
        cv2.putText(img, "FAKE / DANG CHO...", (30, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow("OpenCV Blink Detection", img)
    
    # Chỉnh tốc độ (30ms)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()