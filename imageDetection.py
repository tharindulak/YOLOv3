import cv2
import argparse
import numpy as np
from util_objects import bounding_box as boundingBoxes
from detection_utils import lineDetection as lineDetection
from detection_utils import violationDet as vd

# handle command line arguments

image = cv2.imread('testImg/ca2.png')

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

# read class names from text file
classes = None
with open('new.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet('yolov3-tiny_obj_9000.weights', 'yolov3-tiny_obj.cfg')

# create input blob
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

# set input blob for the network
net.setInput(blob)


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# run inference through the network
# and gather predictions from output layers
outs = net.forward(get_output_layers(net))

# initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.2
nms_threshold = 0.4

# for each detetion from each output layer
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

# apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

info_boxes = []
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    label = str(classes[class_ids[i]])
    boxPredictor = boundingBoxes.Box_Predictor(label, confidences[i], y, x, round(y + h), round(x + w))
    info_boxes.append(boxPredictor)
    #draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

frontVehicles = []
backVehicles = []
selectedLineClasses = []

newInfoBoxes = info_boxes.copy()
for boxPrediction in newInfoBoxes:
    if ('Front' in boxPrediction.label):
        frontVehicles.append(boxPrediction)
    elif ('Back' in boxPrediction.label):
        backVehicles.append(boxPrediction)
    elif (('dash' in boxPrediction.label) or ('nDouble' in boxPrediction.label) or ('mDouble' in boxPrediction.label) or ('double' in boxPrediction.label)):
        selectedLineClasses.append(boxPrediction)

def getKey(vehicle):
    return vehicle.y2

# finding the second last front vehicle from the front view
secondFrontVehicle = None
if (frontVehicles.__len__() ==1):
    secondFrontVehicle = frontVehicles.pop()
else:
    # sort by y2 coordinate of the image
    sortedFrontVehicles = sorted(frontVehicles, key=getKey)
    if (sortedFrontVehicles.__len__() >= 2):
        sortedFrontVehicles.pop()
    if (sortedFrontVehicles.__len__() >= 1 ):
        secondFrontVehicle = sortedFrontVehicles.pop()

# finding the last back vehicle from the front view
nearestBackVehicle = None
if (backVehicles.__len__() == 1):
    nearestBackVehicle = backVehicles.pop()
else:
    sortedBackVehicles = sorted(backVehicles, key=getKey)
    if (sortedBackVehicles.__len__() > 1):
        # sort by y2 coordinate of the image
        nearestBackVehicle = sortedBackVehicles.pop()

# Detect the lines in the lane
lineDetectionResults = lineDetection.lineDetector(image, secondFrontVehicle, nearestBackVehicle)
image = lineDetectionResults.pop()

# Detect lines using model
imshape = image.shape
probableRoiX = imshape[1] / 2
isDashed = True
for selectedLineClass in selectedLineClasses:
    lineClassXMid = (selectedLineClass.x1 + selectedLineClass.x2)/2
    if lineClassXMid > probableRoiX:
        isDashed = False

violatedVehicle = []
for boxPrediction in info_boxes:
    isMiddleLineCrossed = False
    if ('Front' in boxPrediction.label) or ('Back' in boxPrediction.label):
        if lineDetectionResults.__len__() > 0:
            isMiddleLineCrossed = vd.is_violation(boxPrediction.x1, boxPrediction.x2, boxPrediction.y1, boxPrediction.y2, lineDetectionResults[2][0], lineDetectionResults[2][1], lineDetectionResults[2][2], lineDetectionResults[2][3], boxPrediction.label)
        else:
            isMiddleLineCrossed = False
    else:
        isMiddleLineCrossed = False
    violatedDet = (isMiddleLineCrossed, boxPrediction.label, not(isDashed))
    violatedVehicle.append(violatedDet)
    print(str(boxPrediction.x1) + "," + str(boxPrediction.x2) + ","+ str(boxPrediction.y1), str(boxPrediction.y2), str(boxPrediction.label) + "," + str(boxPrediction.score))
    #print(boxPrediction.label)

# go through the detections remaining
# after nms and draw bounding box
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

# display output image
cv2.imshow("object detection", image)
print(str(violatedVehicle) + "Violation")

# wait until any key is pressed
cv2.waitKey()

# save output image to disk
cv2.imwrite("results/yolov3_2.jpg", image)

# release resources
cv2.destroyAllWindows()