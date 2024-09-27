from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("video/car1.mp4")

model = YOLO("../yolo_weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
fixed_region_select = cv2.imread("fixed_region_select.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [185, 225, 550, 225]
totalCount = []

while True:
    success, img = cap.read()

    # Ensure both images have the same size sothat error doesnot occur
    fixed_region_select = cv2.resize(fixed_region_select, (img.shape[1], img.shape[0]))

    # Ensure both images have the same number of channels
    if len(fixed_region_select.shape) == 2:  # If fixed_region_select is grayscale
        fixed_region_select = cv2.cvtColor(fixed_region_select, cv2.COLOR_GRAY2BGR)

    # Ensure both images have the same data type
    fixed_region_select = fixed_region_select.astype(img.dtype)

    # Perform the bitwise_and operation
    imgRegion = cv2.bitwise_and(img,fixed_region_select)
    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    result = model(imgRegion, stream = True)

    detections = np.empty((0, 5))
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),3)
            w,h = x2-x1 , y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h),l=2)
            # confidence
            conf = math.ceil((box.conf[0] * 100))/100
            print(conf)
            # class

            cls = int(box.cls[0]) # class index number

            currentclass = classNames[cls]

            if currentclass == "car" or currentclass == "truck" or currentclass == "bus" or currentclass == "motorbike" and conf > 0.4:
                # cvzone.putTextRect(img, f"{currentclass} {conf}", (max(0, x1), max(30, y1)),
                #                    scale=1, thickness=1, colorB=(0, 255, 0), offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=10,rt = 5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2,colorR=(255,0.0))
        cvzone.putTextRect(img, f"{int(id)}", (max(0, x1), max(30, y1)),
                           scale=1, thickness=1, colorB=(0, 255, 0), offset=2)
        cx, cy = x1 + w // 2, y1 + h // 2 # to find the center
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if limits[0] < cx < limits[2] and limits[1] - 30 < cy < limits[1] + 30:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
    cv2.putText(img,str(len(totalCount)),(180,100),cv2.FONT_HERSHEY_PLAIN,3,(50,50,255),5)


    cv2.imshow("image",img)
    cv2.imshow("imgregion", imgRegion)
    cv2.waitKey(1)