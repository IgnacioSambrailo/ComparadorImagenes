import cv2
import numpy as np
import imutils

# Carga las imágenes que quieres comparar
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

# Convierte las imágenes a escala de grises
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calcula la diferencia entre las dos imágenes
diff = cv2.absdiff(gray1, gray2)

# Aplico threshold
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Dilatacion de diferencias
kernel = np.ones((5,5), np.uint8)
dilate = cv2.dilate(thresh, kernel, iterations = 2)

# Encuentro los contornos

contornos = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos = imutils.grab_contours(contornos)

# Loopeamos los contornos
for contorno in contornos:
    if cv2.contourArea(contorno) > 100:
        # Calculamos el rectangulo
        x, y, w, h = cv2.boundingRect(contorno)
        # Dibujamos el rectangulo
        cv2.rectangle(img1, (x,y), (x+w,y+h), (0,0,255),2)
        cv2.rectangle(img2, (x,y), (x+w,y+h), (0,0,255),2)


# Comparar histogramas de color de las imágenes
hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

# Normalizar los histogramas
cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Calcular la diferencia entre los histogramas
diferencia = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

# Encontrar los píxeles en donde la diferencia sea mayor que un umbral
umbral = 0.3
diff_color = cv2.absdiff(img1, img2)
mask = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)
mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
mask = cv2.dilate(mask, kernel, iterations=2)

# Señalizar los lugares donde el color es distinto en la imagen 1
for i, c in enumerate(contornos):
    # Calcular el área y el perímetro del contorno
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    
    # Si el contorno es lo suficientemente grande
    if area > 500 and perimeter > 100:
        # Calcular las coordenadas del rectángulo que encierra el contorno
        x, y, w, h = cv2.boundingRect(c)
        
        # Dibujar el rectángulo en la imagen 1
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Mostrar imagenes finales con resultado
x = np.zeros((480,10,3), np.uint8)
result = np.hstack((img1, x ,img2))
cv2.imshow("Diferencias", result)





cv2.waitKey(0)
cv2.destroyAllWindows()


