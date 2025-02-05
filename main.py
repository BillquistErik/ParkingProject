import cv2
import json
from ultralytics import YOLO

# Load marked spots from JSON
with open("parking_spots.json", "r") as f:
    regions = json.load(f)["regions"]

# Function to check if a detected object's bounding box is within a region from the JSON
def is_within_region(objectbox, region):
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    objx, objy, objw, objh = objectbox
    return (objx < x + w and objx + objw > x and objy < y + h and objy + objh > y)


cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error accessing webcam"


model = YOLO("best.pt")  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame)
    detected_regions = set() 

    for result in results:
        for objectbox in result.boxes:
            # Extract object bounding box coordinates and class label
            objectbox_values = objectbox.xywh[0].tolist()
            objx, objy, objw, objh = objectbox_values
            class_id = int(objectbox.cls[0]) 
            class_name = model.names[class_id]  

            # Filter
            # if class_name.lower() == "cell phone":
                # Check if the object is in a marked spot
            for region in regions:
                if is_within_region((objx, objy, objw, objh), region):
                    detected_regions.add(region["id"])  # Mark region as detected

                # Draw box for YOLO detection
                cv2.rectangle(frame, (int(objx), int(objy)), (int(objx + objw), int(objy + objh)), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    class_name, 
                    (int(objx), int(objy) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )

    # Loop to Draw JSON spots and update the UI based on detections
    for region in regions:
        x, y, w, h = region["x"], region["y"], region["width"], region["height"]
        region_id = region["id"]

        # Change color based on detection
        color = (0, 255, 0) if region_id in detected_regions else (0, 0, 255)
        status = "Occupied" if region_id in detected_regions else "Empty"

        # Draw spots from JSON
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{region_id}: {status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

 
    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
