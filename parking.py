import cv2
import numpy as np

def highlight_parking_spots(image_path):
    # Load image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Transform across the entire image
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=30)

    parking_mask = np.zeros_like(image)  # Create mask for parking spots

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(parking_mask, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw green lines

    # Find contours to detect rectangular parking spots
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Assuming parking spots are rectangular
            cv2.drawContours(parking_mask, [approx], 0, (0, 0, 255), 2)  # Draw red rectangles

    # Blend the parking mask with the original image
    parking_highlighted = cv2.addWeighted(image, 0.8, parking_mask, 0.5, 0)

    # Show the result
    cv2.imshow("Parking Spot Detection", parking_highlighted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run function with an example image
highlight_parking_spots("parkinglot4.jpg")  # Replace with your parking lot image
