import cv2
import numpy as np

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

dimensions = img1.shape
print(dimensions)


# Calcular los histogramas de colores de cada imagen

hist1 = cv2.calcHist([img1], [0, 1], None, [180, 256], [0, 180, 0, 256])
hist2 = cv2.calcHist([img2], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Normalizar los histogramas
cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Comparar los histogramas y obtener el valor de similitud
similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Si la similitud es menor que un cierto umbral, dibujar rectángulos alrededor de las diferencias
if similarity < 0.1:
    diff = cv2.absdiff(img1, img2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Mostrar la imagen con los rectángulos dibujados
cv2.imshow("Diferencias", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
