# file: face_engine.py
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
import os

class FaceEngine:
    def __init__(self):
        print("⏳ Đang khởi động AI Engine...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Công cụ cắt mặt
        self.mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=self.device)
        # 2. Model FaceNet
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        print(f"✅ AI Engine đã sẵn sàng! (Chạy trên {self.device})")

    def image_to_vector(self, image_path):
        """Chuyển ảnh thành vector"""
        if not os.path.exists(image_path):
            print(f"❌ Không tìm thấy ảnh: {image_path}")
            return None
        
        try:
            img = Image.open(image_path)
            # Cắt mặt
            img_cropped = self.mtcnn(img)
            
            if img_cropped is not None:
                # Tạo vector
                img_embedding = self.resnet(img_cropped.unsqueeze(0).to(self.device))
                return img_embedding.detach().cpu().numpy()[0].tolist()
            else:
                print(f"⚠️ Không tìm thấy mặt trong: {image_path}")
                return None
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {image_path}: {e}")
            return None

    def is_match(self, vector1, vector2, threshold=0.8):
        """
        So sánh 2 vector.
        threshold: Ngưỡng giống nhau (càng nhỏ càng khắt khe). 
        Thường 0.8 - 1.0 là ổn.
        """
        if vector1 is None or vector2 is None:
            return False
            
        # Tính khoảng cách Euclidean
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        distance = np.linalg.norm(v1 - v2)
        
        print(f"🔍 Khoảng cách giữa 2 ảnh: {distance:.4f}")
        
        # Khoảng cách càng nhỏ nghĩa là càng giống nhau
        if distance < threshold:
            return True # Cùng 1 người
        else:
            return False # Khác người

# --- PHẦN CHẠY THỬ (MAIN) ---
if __name__ == "__main__":
    engine = FaceEngine()
    
    print("\n--- BẮT ĐẦU SO SÁNH ---")
    img1 = "test.jpg"
    img2 = "test2.jpg" # Nhớ phải có file này nhé!

    # 1. Lấy vector của cả 2 ảnh
    print(f"Đang xử lý {img1}...")
    vec1 = engine.image_to_vector(img1)
    
    print(f"Đang xử lý {img2}...")
    vec2 = engine.image_to_vector(img2)

    # 2. So sánh
    if vec1 is not None and vec2 is not None:
        ket_qua = engine.is_match(vec1, vec2)
        
        print("-" * 30)
        if ket_qua:
            print(f"✅ KẾT QUẢ: CÙNG MỘT NGƯỜI! (OpenCV đã nhận diện đúng)")
        else:
            print(f"❌ KẾT QUẢ: HAI NGƯỜI KHÁC NHAU.")
    else:
        print("⚠️ Không thể so sánh vì thiếu dữ liệu khuôn mặt.")