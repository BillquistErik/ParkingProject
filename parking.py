import cv2
import numpy as np
import math
import json

def highlight_parking_spots(image_path, output_json="parking_spots.json"):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=30)
    parking_mask = np.zeros_like(image)

    vertical_lines = []
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 10:  # Horizontal lines
                horizontal_lines.append((x1, y1, x2, y2))
                cv2.line(parking_mask, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for horizontal lines
            elif 80 < abs(angle) < 100:  # Vertical or slightly tilted lines
                vertical_lines.append((x1, y1, x2, y2))
                cv2.line(parking_mask, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for detected lines

    vertical_lines.sort(key=lambda l: min(l[0], l[2]))  # Sort by x-position
    horizontal_lines.sort(key=lambda l: min(l[1], l[3]))  # Sort by y-position

    mid_y = image.shape[0] // 2
    if horizontal_lines:
        mid_y = horizontal_lines[len(horizontal_lines) // 2][1]

    parking_spots = []
    spot_id = 1
    for i in range(len(vertical_lines) - 1):
        x1_left, y1_left, x2_left, y2_left = map(int, vertical_lines[i])
        x1_right, y1_right, x2_right, y2_right = map(int, vertical_lines[i + 1])
        if x1_right - x1_left > 30:
            top_spot = {"id": f"spot{spot_id}", "x": int(x1_left), "y": int(50), "width": int(x1_right - x1_left), "height": int(mid_y - 55)}
            bottom_spot = {"id": f"spot{spot_id + 1}", "x": int(x1_left), "y": int(mid_y + 5), "width": int(x1_right - x1_left), "height": int(image.shape[0] - mid_y - 55)}
            parking_spots.append(top_spot)
            parking_spots.append(bottom_spot)
            spot_id += 2
            cv2.rectangle(parking_mask, (x1_left, 50), (x1_right, mid_y - 5), (0, 0, 255), 2)
            cv2.rectangle(parking_mask, (x1_left, mid_y + 5), (x1_right, image.shape[0] - 50), (0, 0, 255), 2)

    with open(output_json, "w") as json_file:
        json.dump({"regions": parking_spots}, json_file, indent=2)

    parking_highlighted = cv2.addWeighted(image, 0.8, parking_mask, 0.5, 0)
    cv2.imshow("Parking Spot Detection", parking_highlighted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

highlight_parking_spots("parkinglot3.jpg")



