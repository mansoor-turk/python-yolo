from ultralytics import YOLO
import cv2

# Load YOLO model for person detection
# yolo_model = YOLO("yolov8m.pt")
yolo_model = YOLO("yolov8l.pt")  # Larger model
# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load image
image_path = "C:/Users/m.mansoor/Downloads/30person.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 🔍 1. Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# 🔍 2. Detect persons using YOLO
results = yolo_model(image_path, conf=0.30)  # Slightly higher confidence
person_boxes = [box for box in results[0].boxes if int(box.cls[0]) == 0]

# 🎨 Draw face boxes
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.putText(image, "Face", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# 🎨 Draw person boxes
for box in person_boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0]) * 100
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"Person ({conf:.1f}%)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 💾 Save result
cv2.imwrite("both_person_face_detected.jpg", image)

# 👁 Show output
cv2.imshow("Body + Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 📊 Stats
print(f"✅ Persons detected: {len(person_boxes)}")
print(f"✅ Faces detected: {len(faces)}")
