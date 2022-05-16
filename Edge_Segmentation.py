import cv2
import numpy as np

IMG_PATH = './images/chopped-vegetables.jpg'

image_segmentation_names = ['Original Image', 'Sobel X', 'Laplacian of Gaussian',
                            'Sobel Y', 'Sobel', 'Prewitt X', 'Prewitt Y', 'Prewitt']

img = cv2.imread(IMG_PATH)
print(img.shape)


if(img.shape[0] > 3000):
    img = cv2.resize(img, dsize=(0, 0), fx=0.1, fy=0.1)
print(img.shape)

for name in image_segmentation_names:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, height=img.shape[0], width=img.shape[1])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

# sobel
img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_8U, 1, 0, ksize=3)
img_sobely = cv2.Sobel(img_gaussian, cv2.CV_8U, 0, 1, ksize=3)
img_sobel = img_sobelx + img_sobely


# prewitt
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

# laplacian
img_laplace = cv2.convertScaleAbs(
    cv2.Laplacian(img_gaussian, cv2.CV_16S, ksize=1))

cv2.imshow("Original Image", img)
cv2.imshow("Sobel X", img_sobelx)
cv2.imshow("Sobel Y", img_sobely)
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt", img_prewitt)
cv2.imshow("Laplacian of Gaussian", img_laplace)

cv2.waitKey(0)
cv2.destroyAllWindows()
