from ultralytics import YOLO
import cv2
import cvzone
import math
import time


cap = cv2.VideoCapture(0)
cap.set(3,1280) #Width
cap.set(4,720)#height
model = YOLO("../Yolo-Weights/yolov8n.pt")




classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boot",
              "traffic light", "fire hydrant", "stop sign", "parking neter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "best", "banana", "apple",
              "sandwich", "orange", 'broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','potttedplant','bed',
              'diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone',
              'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
              'teddy bear','hair drier','toothbrush','pen']

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success,img=cap.read()
    img = cv2.flip(img,1)
    results = model(img, stream=True)#for video stream=True & for image show=True
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1,y1,x2,y2=box.xyxy[0]#x1 y1 width height
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2-x1, y2-y1
            bbox = x1,y1,w,h
            #print(x1,y1,x2,y2)
            cvzone.cornerRect(img,bbox)
            #confidence
            conf = math.ceil((box.conf[0]*100))/100
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(30,y1)))

            #class name
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(30, y1)),scale=0.8,thickness=1)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image",img)
    if cv2.waitKey(25) & 0xff == ord("f"):
        break
cap.release()
cv2.destroyAllWindows()
