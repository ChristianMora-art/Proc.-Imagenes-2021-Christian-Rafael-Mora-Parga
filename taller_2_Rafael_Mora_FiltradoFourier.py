# Created by Christian Rafael Mora Parga
import cv2
import numpy as np
import os
import sys


class thetaFilter:
    def __init__(self, img_gris):
        self.img_gris = img_gris #recibe la imagen en grises
        self.theta = None
        self.delta_theta = None

    #1. b) Un método set_thetapara recibir los parámetros theta y delta_theta que definen la respuesta del filtro.
    def set_theta(self, theta, delta_theta):
        self.theta = theta
        self.delta_theta = delta_theta

    #1. c)Un método filtering que implementa un filtrado vía FFT que permite solo el paso de las componentes de
    # frecuencia orientadas entre 𝜃−Δ𝜃 y 𝜃-Δ𝜃 (anula las demás).
    def filtering(self):
        # Se realiza la FFT con su respectivo reordenamiento de frecuencias (shift)
        image_gray_fft = np.fft.fft2(self.img_gris)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        # Se calcula magnitud y fase (orientaciones)
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_gray_fft_phase = np.angle(image_gray_fft_shift, deg=True)

        num_rows, num_cols = (image_gray_fft_phase.shape[0], image_gray_fft_phase.shape[1])

        #se calcula el rango de angulos de frecuencias de paso en lim con base a "theta":
        #si Δ𝜃 ∈ ℝ+: 𝜃 < frecs. de paso < 𝜃 - Δ𝜃
        #si Δ𝜃 ∈ ℝ-: 𝜃 - Δ𝜃 < frecs. de paso < 𝜃
        lim = self.theta - self.delta_theta

        # Métodos propuestos (que entregan mismo resultado):
        # 1) función de transferencia según los ángulos que pertenecen al rango angular de paso:
        # se multiplica ambas transformadas espectralmente:
        phase_filter_mask = np.zeros_like(image_gray)
        # 2) Se reasignan directamente las fases pertenecientes a transformada de la señal en grises
        # que están en el rango de direcciones del filtro (se usa la función polar2Cart(A, phi)):
        filt_gray_fft_shift = np.zeros_like(image_gray_fft_shift)

        def polar2Cart(A, phi):
            return A * (np.cos(phi) + np.sin(phi) * 1j)

        if self.delta_theta > 0:
            for row in range(num_rows):
                for col in range(num_cols):
                    if image_gray_fft_phase[row][col] < self.theta and image_gray_fft_phase[row][col] > lim:
                        phase_filter_mask[row][col] = 1
                        filt_gray_fft_shift[row][col] = polar2Cart(image_gray_fft_mag[row][col], np.pi*image_gray_fft_phase[row][col]/180)
                    else:
                        phase_filter_mask[row][col] = 0
                        filt_gray_fft_shift[row][col] = polar2Cart(0.0, 0.0)
        else:
            for row in range(num_rows):
                for col in range(num_cols):
                    if image_gray_fft_phase[row][col] < lim and image_gray_fft_phase[row][col] > self.theta:
                        phase_filter_mask[row][col] = 1
                        filt_gray_fft_shift[row][col] = polar2Cart(image_gray_fft_mag[row][col], np.pi*image_gray_fft_phase[row][col]/180)
                    else:
                        phase_filter_mask[row][col] = 0
                        filt_gray_fft_shift[row][col] = polar2Cart(0.0, 0.0)

        # Método 1):
        fft_filtered = image_gray_fft_shift * phase_filter_mask
        # Método 2):
        #fft_filtered = filt_gray_fft_shift

        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        #Se solicitó filtrar por la fase:
        image_filtered = np.angle(image_filtered)
        maxi = np.max(image_filtered)
        if np.max(image_filtered) == 0:
            print('No hay componentes de frecuencia en este rango de orientaciones angulares')
            maxi = 1
        image_filtered /= maxi
        #Visualización de las imagenes del punto 1
        cv2.imshow("1) Imagen Original", self.img_gris); cv2.waitKey(0)
        cv2.imshow("1) Imagen Filtrada", image_filtered);cv2.waitKey(0)
        return image_filtered

    #Punto 2:
    # Implemente un banco de 4 filtros direccionales utilizando la clase theta Filter
    # para las orientaciones 0°, 45°, 90°y 135°, y con Δ𝜃=5°, sobre una imagen de entrada.
    def fourFilter_banc(self):
        # Construcción del filtro de la misma forma que en el punto anterior:
        # Retorna una función de transferencia de unos y ceros (máscara espectral)
        def filterConstruct(theta, num_rows, num_cols, image_gray_fft_phase):
            delta_theta = 5
            lim = theta - delta_theta #se calcula el rango de orientaciones requerido según theta
            phase_filter_mask = np.zeros_like(image_gray)
            for row in range(num_rows):
                for col in range(num_cols):
                    if not image_gray_fft_phase[row][col] < theta and image_gray_fft_phase[row][col] > lim:
                        phase_filter_mask[row][col] = 1
                    else:
                        phase_filter_mask[row][col] = 0
            return phase_filter_mask
        # Filtrado realizado de la misma forma que en el punto anterior:
        # Retorna la imagen en el dominio original ya filtrada
        def convolvFreq(sig, mask):
            fft_filtered = sig * mask
            image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
            image_filtered = np.angle(image_filtered)
            maxi = np.max(image_filtered)
            if np.max(image_filtered) == 0:
                print('En este filtro no hay componentes de frecuencia en este rango de orientaciones angulares')
                maxi = 1
            image_filtered /= maxi
            return image_filtered

        #Se realiza la transformada de la imagen en grises una sola vez:
        image_gray_fft = np.fft.fft2(self.img_gris)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        image_gray_fft_phase = np.angle(image_gray_fft_shift, deg=True)
        num_rows, num_cols = (image_gray_fft_phase.shape[0], image_gray_fft_phase.shape[1])

        # Se envía la representación espectral de la imagen 4 veces con sus ángulos de filtrado correspondientes:
        phase_filter_mask_1 = filterConstruct(0, num_rows, num_cols, image_gray_fft_phase)
        phase_filter_mask_2 = filterConstruct(45, num_rows, num_cols, image_gray_fft_phase)
        phase_filter_mask_3 = filterConstruct(90, num_rows, num_cols, image_gray_fft_phase)
        phase_filter_mask_4 = filterConstruct(135, num_rows, num_cols, image_gray_fft_phase)

        #Se realiza el filtrado en el dominio de la frecuencia, como se hizo para el primer punto
        img_filt1, img_filt2 = convolvFreq(image_gray_fft_shift,phase_filter_mask_1), convolvFreq(image_gray_fft_shift,phase_filter_mask_2)
        img_filt3, img_filt4 = convolvFreq(image_gray_fft_shift,phase_filter_mask_3), convolvFreq(image_gray_fft_shift,phase_filter_mask_4)

        # 2. c) Sintetice una nueva imagen promediandolas imágenes resultantes después del banco de filtro y
        #visualice la imagen resultante.
        # Se calcula el promedio pixel a pixel sobre las 4 imágenes:
        imag_prom = np.mean(np.array([img_filt1, img_filt2, img_filt3, img_filt4]), axis=0)

        # 2. b) Visualicecada una de las imágenes con su respectivo titulo
        #se muestran todas las 4 imagenes filtradas, y la sintética
        cv2.imshow("2) Filtered image 0 grados", img_filt1);cv2.waitKey(0)
        cv2.imshow("2) Filtered image 45 grados", img_filt2); cv2.waitKey(0)
        cv2.imshow("2) Filtered image 90 grados", img_filt3); cv2.waitKey(0)
        cv2.imshow("2) Filtered image 135 grados", img_filt4); cv2.waitKey(0)
        cv2.imshow("2) Promedio Filtered image", imag_prom); cv2.waitKey(0)


if __name__ == '__main__':

    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    #se pasa la imagen a grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. a) La clase recibe en su constructor la imagen (en grises) a filtrar:
    ob = thetaFilter(image_gray)
    #primer argumento de entrada theta, segundo delta theta:
    ob.set_theta(0,180)

    # 2. Segundo punto, banco de 4 filtros:
    ob.filtering()
    ob.fourFilter_banc()
