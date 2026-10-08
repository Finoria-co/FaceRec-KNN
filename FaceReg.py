import cv2
import numpy as np
import os

# 1. ฟังก์ชัน KNN เขียนเองสั้นๆ เทียบ Euclidean Distance ตรงๆ
def knn(X, y, z, k=1):
    # Cast เป็น float32 เพื่อไม่ให้ค่าติดลบแล้วล้น (Underflow)
    diff = X.astype(np.float32) - z.astype(np.float32)
    distances = np.sum(diff ** 2, axis=1) # Euclidean Distance
    
    # ดึง index ของรูปที่ใกล้เคียงที่สุด K อันดับแรก
    idx = np.argsort(distances)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]

# 2. อ่านรูปจากโฟลเดอร์เพื่อนๆ เข้ามาเก็บใน X และ y
X = []
y = []

for folder in os.listdir():
    if os.path.isdir(folder) and not folder.startswith("."):
        for file in os.listdir(folder):
            if file.endswith(".jpg"):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    X.append(img.flatten())
                    y.append(folder)

X = np.array(X)
y = np.array(y)

# 3. เปิดกล้อง Real-time
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # วาดกรอบสี่เหลี่ยมสีแดง
    cv2.rectangle(frame, (150, 50), (390, 450), (242, 143, 143), 2)

    # ตัดภาพในกรอบแปลงเป็น Grayscale
    face = cv2.cvtColor(frame[50:450, 150:390], cv2.COLOR_BGR2GRAY)
    z = face.flatten()

    # ทำการทำนายผลถ้ามีรูปในระบบ
    if len(X) > 0:
        name = knn(X, y, z, k=1)
        cv2.putText(frame, name, (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()