
import cv2
import numpy as np

def show(image):
    cv2.imshow("Laser Scan Image", image)
    cv2.waitKey(2000)


class Test:
    def detect_curved_areas(self, image):
        # Convert the image to grayscale
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        

        # Apply Gaussian blur
        kernel = np.ones((3, 3), np.float32)
        img = cv2.dilate(img, kernel, iterations=3)
        show(img)
        
        # # img = cv2.bilateralFilter(img, 9, 4, 20)
        # # show(img)
        for i in range(10):
            img = cv2.GaussianBlur(img, (3, 3), 2)
        show(img)

        # # Apply Hough Circle Transform to detect circular shapes
        # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=2, param2=15, minRadius=20, maxRadius=40)
        
        # img = cv2.blur(img, (3, 3))
        # show(img)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1 = 10, param2 = 10, minRadius = 1, maxRadius = 40)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
                # Draw the circle
                cv2.circle(image, (x, y), r, (0, 0, 255), 2)

                # Draw the center of the circle
                cv2.circle(image, (x, y), 1, (255, 0, 0), 2)

                # Print the location of the curved area
                print(f"Curved area detected at ({x}, {y}) {r}")
        return image


img = cv2.imread('/home/andrew/Downloads/laser10.jpg')
t = Test()
im = t.detect_curved_areas(img)
show(im)