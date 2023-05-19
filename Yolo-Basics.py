from ultralytics import YOLO
import cv2

#call the YOLO and pass in the weights by yolo version 8 and weight with .pt extension
#n-nano ,m-medium , l-large
model = YOLO('../Yolo-Weights/yolov8m.pt')
results = model("1.jpg",show=True)
cv2.waitKey(0)