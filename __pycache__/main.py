from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
import sys
import cv2
import numpy as np
import os
import base64
import imutils


class ImageCompare(QWidget):
    def __init__(self):
        super().__init__()

        # Inicializar variables
        self.image_paths1 = []
        self.image_paths2 = []
        self.output_dir = ""

        # Configurar ventana
        self.setWindowTitle("Comparación de imágenes")
        self.setGeometry(100, 100, 800, 600)

        # Configurar widgets
        self.input1_label = QLabel("Seleccione los diseños originales")
        self.input2_label = QLabel("Seleccione las capturas a comparar")
        self.input1_button = QPushButton("Seleccionar imágenes")
        self.input2_button = QPushButton("Seleccionar imágenes")
        self.dir_label = QLabel("Directorio de salida")
        self.dir_button = QPushButton("Seleccionar directorio")
        self.start_button = QPushButton("Comenzar análisis")
        self.result_label = QLabel("Resultados:")

        # Conectar botones a funciones
        self.input1_button.clicked.connect(self.select_image1)
        self.input2_button.clicked.connect(self.select_image2)
        self.dir_button.clicked.connect(self.select_output_dir)
        self.start_button.clicked.connect(self.start_analysis)

        # Configurar layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.input1_label)
        self.layout.addWidget(self.input1_button)
        self.layout.addWidget(self.input2_label)
        self.layout.addWidget(self.input2_button)
        self.layout.addWidget(self.dir_label)
        self.layout.addWidget(self.dir_button)
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.result_label)
        self.setLayout(self.layout)
        
    def image_to_base64(self, image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            encoded = base64.b64encode(image_bytes)
            return encoded.decode(errors="replace")



    def select_image1(self):
        self.image_paths1, _ = QFileDialog.getOpenFileNames(self, "Seleccionar imágenes", "", "Images (*.png *.xpm *.jpg *.bmp)")

    def select_image2(self):
        self.image_paths2, _ = QFileDialog.getOpenFileNames(self, "Seleccionar imágenes", "", "Images (*.png *.xpm *.jpg *.bmp)")

    def select_output_dir(self):
        self.output_dir = str(QFileDialog.getExistingDirectory(self, "Seleccionar directorio"))

    def start_analysis(self):
        if not self.image_paths1 or not self.image_paths2:
            self.result_label.setText("Por favor seleccione al menos una imagen en cada input")
            return

        if not self.output_dir:
            self.result_label.setText("Por favor seleccione un directorio de salida")
            return

        # Crear archivo HTML para resultados
        result_html = "<html><head><title>Resultados de la comparación de imágenes</title></head><body>"
        result_html += "<h1>Diseño vs Captura // Identificación general + Colores</h1>"

        for i, (path1, path2) in enumerate(zip(self.image_paths1, self.image_paths2)):
            # Leer imágenes
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)

            # Hago resize de las imagenes
            scale_percent = 20 # Porcentaje del resize
            width = int(img1.shape[1] * scale_percent / 100)
            height = int(img1.shape[0] * scale_percent / 100)
            dim = (width, height)

            img1 = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
            img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

            # Convierte las imágenes a escala de grises
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Calcula la diferencia entre las dos imágenes
            diff = cv2.absdiff(gray1, gray2)

            # Aplico threshold
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # Dilatacion de diferencias
            kernel = np.ones((5,5), np.uint8)
            dilate = cv2.dilate(thresh, kernel, iterations = 3)

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

            # Mostrar imagenes finales con resultado
            x = np.zeros((480,10,3), np.uint8)
            result = np.hstack((img1, x ,img2))
            cv2.imwrite('result.jpg', result)

            # Identificacion colores

            # Calcular la diferencia absoluta entre las dos imágenes
            diff = cv2.absdiff(img1, img2)

            # Aplicar threshold
            thresh = cv2.threshold(diff, 20, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

            # Aplicar dilatación
            kernel = np.ones((2,2), np.uint8)
            filtrado = cv2.dilate(thresh[1], kernel, iterations=1)


            # Convertir imagen a formato HTML
            retval, buffer = cv2.imencode('.jpg', result)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            retval, buffer2 = cv2.imencode('.jpg', filtrado)
            jpg_as_text2 = base64.b64encode(buffer2).decode('utf-8')

            #Aplico nombre descriptivo
            
            titulo = path1.split('/')
            titulo = titulo[-1]

            result_html += f"<h2>Par de imágenes {titulo}:</h2>"
            result_html += '<img src="data:image/jpeg;base64,{}">'.format(jpg_as_text)
            result_html += '<img src="data:image/jpeg;base64,{}">'.format(jpg_as_text2)
            

        # Cerrar archivo HTML
        result_html += "</body></html>"

        # Guardar archivo HTML
        result_path = os.path.join(self.output_dir, "results.html")
        with open(result_path, "w") as f:
            f.write(result_html)

        # Mostrar mensaje de éxito
        self.result_label.setText(f"Análisis completado. Resultados guardados en {result_path}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = ImageCompare()
    gui.show()
    sys.exit(app.exec_())