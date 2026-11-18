import cv2
import numpy as np
import os

# 1. ฟังก์ชัน KNN
def knn(X, y, z, k=1, threshold=3500):
    # Cast เป็น float32 เพื่อไม่ให้ค่าติดลบแล้วล้น (Underflow)
    diff = X.astype(np.float32) - z.astype(np.float32)
    # ใช้ np.mean แทน np.sum เพื่อหาค่าเฉลี่ยความต่างต่อ 1 พิกเซล (ทำให้ไม่ขึ้นกับขนาดรูปภาพ)
    distances = np.mean(diff ** 2, axis=1) 
    
    # ดึง index ของรูปที่ใกล้เคียงที่สุด K อันดับแรก
    idx = np.argsort(distances)[:k]
    
    # เช็คระยะห่าง ถ้าน้อยที่สุดยังมากกว่า threshold ให้ถือว่าไม่รู้จัก
    if distances[idx[0]] > threshold:
        return "Unknown"
        
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
                    img = cv2.resize(img, (250, 250)) # ปรับขนาดรูป
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

    frame = cv2.flip(frame, 1) # พลิกภาพให้เหมือนกระจกเงา

    cv2.rectangle(frame, (195, 125), (445, 375), (242, 143, 143), 2)

    # ตัดภาพในกรอบแปลงเป็น Grayscale
    face = cv2.cvtColor(frame[125:375, 195:445], cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (250, 250)) # ปรับขนาดรูป
    z = face.flatten()

    # ทำการทำนายผลถ้ามีรูปในระบบ
    if len(X) > 0:
        name = knn(X, y, z, k=1)
        # แสดงชื่ออยู่เหนือกรอบ (x=195 ตรงขอบซ้าย, y=110 เหนือขอบบน 125)
        cv2.putText(frame, name, (195, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # กดปุ่ม Esc เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()