from ultralytics import YOLO
md = YOLO("yolov8n.pt")
result = md("C:/Users/m.mansoor/Downloads/blur.png")

result[0].show()
# print(result)
print(result[0])