import cv2
import os

# รับชื่อของเพื่อนที่จะบันทึกหน้า
name = input("ป้อนชื่อเพื่อน (ภาษาอังกฤษ): ").strip()

# สร้างโฟลเดอร์ถ้ายังไม่มีชื่อนี้
if not os.path.exists(name):
   os.mkdir(name)

# นับจำนวนรูปที่มีอยู่จะได้ไม่ทับไฟล์ที่มีอยู่
existing_files = [f for f in os.listdir(name) if f.endswith('.jpg')]
i = len(existing_files) + 1

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print(f"กำลังเปิดกล้อง... กด 's' เพื่อเซฟรูปหน้า {name} (กด 'q' เพื่อออก)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถเปิดกล้องได้")
        break

    cv2.rectangle(frame, (250,120), (390,300), (0,0,255), 2)
    face = cv2.cvtColor(frame[120:300, 250:390, :], cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    cv2.imshow('face', face)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f'{name}/{i}.jpg', face)
        print(f"บันทึกรูปที่ {i} ของ {name} เรียบร้อย")
        i += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()