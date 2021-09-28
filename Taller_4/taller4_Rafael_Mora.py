# Created by Christian Rafael Mora Parga
# TALLER 4, miércoles 29 de septiembrede 2021, 11am
import cv2
import numpy as np
import os
import sys
from enum import Enum

from hough import Hough
from orientation_methods import gradient_map


class Quadrilateral:

    def __init__(self, N, img):

        if (N % 2) == 0:
            self.N_size_img = N
        else:
            print('El tamaño debe ser un entero par, se le restará una unidad a su entrada')
            self.N_size_img = N - 1

        self.imagen = img
        #self.back_img = None

    # 1) Implemente una clase en Python llamada Quadrilateral, la cual genera cuadriláteros de color
    # magenta sobre un fondo de color cian, en una imagen de tamaño N x N.
    def generate(self):
        # Se genera el fondo Cian:
        h, w = self.N_size_img, self.N_size_img  # ancho y alto

        cian_back = np.zeros((h, w, 3), np.uint8)
        cian_back[:, :] = (255, 255, 0)  # (B, G, R) para el color cyan

        # Se generan las cordenadas del cuadrilatero
        # aquí es [columnas, filas]
        # Se definen las 4 coordenadas a partir de los límites en pixeles de los 4 cuadrantes
        a_h, a_w = int(np.random.uniform(0, int(h/2) - 2, 1)), int(np.random.uniform(0, int(w/2) - 2, 1))
        d_h, d_w = int(np.random.uniform(0, int(h/2) - 2, 1)), int(np.random.uniform(int(w/2) - 1, w - 1, 1))
        b_h, b_w = int(np.random.uniform(int(h/2) - 1, h - 1, 1)), int(np.random.uniform(0, int(w/2) - 2, 1))
        c_h, c_w = int(np.random.uniform(int(h/2) - 1, h - 1, 1)), int(np.random.uniform(int(w/2) - 1, w - 1, 1))

        # cuadrantes: 1=a, 2=b, 3=c, 4=d
        a, b, c, d = [a_w, a_h], [b_w, b_h], [c_w, c_h], [d_w, d_h]
        puntos = np.array([a, b, c, d], dtype=np.int32)
        puntos.reshape((-1, 1, 2))
        # Se dibujan las líneas y se llena ese espacio de color magenta:
        cv2.polylines(cian_back, [puntos], True, (229, 9, 127), 3)
        cv2.fillPoly(cian_back, [puntos], (229, 9, 127))

       #self.back_img = cian_back
        return cian_back
        # cv2.imshow("Image", cian_back)

    # 2) Implemente una método en Python llamado DetectCorners, el cual recibe la imagen de entrada
    # en RGB de un polígono y detecta sus esquinas (máximo polígonos de 10 lados) .
    def DetectCorners(self):
        class Methods(Enum):
            Standard = 1
            Direct = 2

        method = Methods.Standard
        high_thresh = 300
        bw_edges = cv2.Canny(self.imagen, high_thresh * 0.3, high_thresh, L2gradient=True)

        hough = Hough(bw_edges)
        if method == Methods.Standard:
            accumulator = hough.standard_transform()
        elif method == Methods.Direct:
            image_gray = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
            theta, _ = gradient_map(image_gray)
            accumulator = hough.direct_transform(theta)
        else:
            sys.exit()

        #VALORES RECOMENDADOS PARA LA IMAGEN GENERADA POR EL METODO generate():
        #70 votos a cada recta, y 10 picos (líneas) máximo
        acc_thresh, N_peaks = 70, 10
        nhood = [25, 9]
        # Se utiliza el algoritmo de HOUGH para hallar las rectas presentes en la imagen
        peaks = hough.find_peaks(accumulator, nhood, acc_thresh, N_peaks)

        _, cols = self.imagen.shape[:2]
        image_draw_rgb = np.copy(self.imagen)
        lista_interpts = []
        for peak in peaks:
            rho, theta_ = peak[0], hough.theta[peak[1]]
            theta_pi, theta_ = np.pi * theta_ / 180, theta_ - 180
            a, b = np.cos(theta_pi), np.sin(theta_pi)
            x0, y0 = a * rho + hough.center_x, b * rho + hough.center_y

            x1, y1 = int(round(x0 + cols * (-b))), int(round(y0 + cols * a))
            x2, y2 = int(round(x0 - cols * (-b))), int(round(y0 - cols * a))

            vals = {'point1': (x1, y1), 'point2': (x2, y2)}
            lista_interpts.append(vals)

        # se hallan las intersecciones entre las líneas no paralelas encontradas
        intersec = []
        for line1 in lista_interpts:
            x1, y1, x2, y2 = line1['point1'][0], line1['point1'][1], line1['point2'][0], line1['point2'][1]
            A1, B1 = (y2 - y1), (x1 - x2)
            C1 = A1 * x1 + B1 * y1

            for line2 in lista_interpts:
                x1, y1, x2, y2 = line2['point1'][0], line2['point1'][1], line2['point2'][0], line2['point2'][1]
                A2, B2 = y2 - y1, x1 - x2
                C2 = A2*x1 + B2*y1
                det = A1*B2 - A2*B1  # determinante del sistema
                det += np.finfo(float).eps  # se añade un pequeño valor para que no sea igual a 0
                # soluciones al sistema de ecuaciones:
                intx, inty = int((B2 * C1 - B1 * C2) / det), int((A1 * C2 - A2 * C1) / det)

                # los valores de las soluciones están acotados entre 0 y el tamaño N
                if ((0 <= intx <= self.N_size_img - 1) and (0 <= inty <= self.N_size_img - 1)):
                    intersec.append([intx, inty])
                    # se dibujan las líneas y los circulos de las intersecciones
                    # cv2.line(image_draw, line1['point1'], line1['point2'], (0, 0, 255), 1)
                    # cv2.line(image_draw, line2['point1'], line2['point2'], (0, 0, 255), 1)
                    cv2.circle(image_draw_rgb, (intx, inty), 5, (0, 255, 255), 2)

        return image_draw_rgb, np.unique(intersec, axis=0)


if __name__ == '__main__':

    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)

    # Punto 1:
    #Tamano N que debe ser par:
    N = 500
    obj = Quadrilateral(N, image)

    quadri_img = obj.generate()
    #se guarda la imagen generada
    cv2.imwrite('imagen_quadri.jpg', quadri_img)

    # Punto 2:
    rgb_img, pos_esq = obj.DetectCorners()

    cv2.imshow("lines", rgb_img)
    cv2.waitKey(0)
    print('Coordenadas (posiciones) esquínas', pos_esq)

