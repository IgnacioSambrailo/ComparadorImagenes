import cv2
import numpy as np
import imutils

# Cargar las dos imágenes
img1 = cv2.imread("imagen1.jpg")
img2 = cv2.imread("imagen3.jpg")

# Hago resize de las imagenes
scale_percent = 20 # Porcentaje del resize
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)

img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

# Calcular la diferencia absoluta entre las dos imágenes
diff = cv2.absdiff(img1, img2)

# Aplicar threshold
thresh = cv2.threshold(diff, 20, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

# Aplicar dilatación
kernel = np.ones((2,2), np.uint8)
dilate = cv2.dilate(thresh[1], kernel, iterations=1)
cv2.imshow("Muestra", dilate)

cv2.waitKey(0)
cv2.destroyAllWindows()
