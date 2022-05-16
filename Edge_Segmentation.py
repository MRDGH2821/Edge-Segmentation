import cv2
import numpy as np

image_segmentation_names = ['Original Image', 'Sobel X', 'Laplacian of Gaussian',
                            'Zero-crossing', 'Sobel Y', 'Sobel', 'Prewitt X',
                            'Prewitt Y', 'Prewitt']


def Zero_crossing(image):
    z_c_image = np.zeros(image.shape)

    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i + 1, j - 1], image[i + 1, j], image[i + 1, j + 1],
                         image[i, j - 1], image[i, j + 1], image[i - 1, j - 1],
                         image[i - 1, j], image[i - 1, j + 1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h > 0:
                    positive_count += 1
                elif h < 0:
                    negative_count += 1

            # If both negative and positive values exist in
            # the pixel neighborhood, then that pixel is a
            # potential zero crossing

            z_c = ((negative_count > 0) and (positive_count > 0))

            # Change the pixel value with the maximum neighborhood
            # difference with the pixel

            if z_c:
                if image[i, j] > 0:
                    z_c_image[i, j] = image[i, j] + np.abs(e)
                elif image[i, j] < 0:
                    z_c_image[i, j] = np.abs(image[i, j]) + d

    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image / z_c_image.max() * 255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image


img = cv2.imread('./simple-image-wide.png')
print(img.shape)

for name in image_segmentation_names:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, img.shape[1], img.shape[0])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

# sobel
img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_8U, 1, 0, ksize=5)
img_sobely = cv2.Sobel(img_gaussian, cv2.CV_8U, 0, 1, ksize=5)
img_sobel = img_sobelx + img_sobely


# prewitt
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

# laplacian
img_laplace = cv2.convertScaleAbs(
    cv2.Laplacian(img_gaussian, cv2.CV_16S, ksize=1))

# zero cross
LoG = cv2.Laplacian(img_gaussian, cv2.CV_16S, ksize=1)
zeroCross = Zero_crossing(LoG)

minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3, 3)))
maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3, 3)))
zeroCross2 = np.logical_or(np.logical_and(
    minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))

cv2.imshow("Original Image", img)
cv2.imshow("Sobel X", img_sobelx)
cv2.imshow("Sobel Y", img_sobely)
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)
cv2.imshow("Laplacian of Gaussian", img_laplace)
# cv2.imshow('Zero-crossing', zeroCross)
cv2.imshow('Zero-crossing', zeroCross2)

cv2.waitKey(0)
cv2.destroyAllWindows()
