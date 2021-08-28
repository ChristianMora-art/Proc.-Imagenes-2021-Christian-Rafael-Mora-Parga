# Created by Christian Rafael Mora Parga
import cv2
import numpy as np
import os
import sys


class thetaFilter:
    def __init__(self, img_gris):
        self.img_gris = img_gris  # recibe la imagen en grises
        self.theta = None
        self.delta_theta = None

    #  1. b) Un método set_thetapara recibir los parámetros theta y delta_theta que definen la respuesta del filtro.
    def set_theta(self, theta, delta_theta):
        if theta < 0:
            theta = -1 * theta + 180
        # else:
        #   theta += 90
        self.theta = theta
        self.delta_theta = delta_theta

    #  Método usado en ambos puntos para crear la mascara para realizar la convolución en frecuencia (producto):
    def angularSpectral_mask(self):
        # pre-computations
        num_rows, num_cols = (self.img_gris.shape[0], self.img_gris.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2  # el centro de la imagen

        # 2) Se crea una matriz de ángulos respecto al centro (origen) según las dimensiones de la imagen original:
        filter_space = 180 + 180 * np.arctan2(row_iter - half_size, col_iter - half_size) / np.pi

        # se calcula el rango de angulos de frecuencias de paso en lim con base a "theta":
        # lim_dif = 𝜃 - Δ𝜃, lim_sum = 𝜃 + Δ𝜃 y en lim_dif_180 y lim_sum_180 se añaden 180 grados para construir el
        # otro rango simétrico de fases.
        # 𝜃 - Δ𝜃 < filter_space < 𝜃 + Δ𝜃 donde se fuerza que estén entre [0, 360] usando mod(.,360)
        lim_dif, lim_sum = np.divmod(self.theta - self.delta_theta, 360), np.divmod(self.theta + self.delta_theta, 360)
        lim_dif_180, lim_sum_180 = np.divmod(180 + self.theta - self.delta_theta, 360), np.divmod(
            180 + self.theta + self.delta_theta, 360)

        idx_1, idx_2 = filter_space < lim_sum[1], filter_space > lim_dif[1]
        idx1_180, idx2_180 = filter_space < lim_sum_180[1], filter_space > lim_dif_180[1]

        # Se hace la operación AND para obtener los valores True en las intersecciones True de ambas matrices:
        idx_phases1, idx_phases2 = np.bitwise_and(idx_1, idx_2), np.bitwise_and(idx1_180, idx2_180)
        # Se hace la operación OR para superponer los valores de ambas matrices:
        idx_phases = np.bitwise_or(idx_phases1, idx_phases2)
        phase_filter_mask = idx_phases.astype(np.uint8)
        # Se asegura que esté en True el punto central de la imagen en el dom espectral:
        phase_filter_mask[int(half_size), int(half_size)] = 1

        # intervalos angulares problemáticos:
        a, aa, aaa = 180 - 1 + self.delta_theta, 0 - 1 + self.delta_theta, 360 - 1 + self.delta_theta,
        b, bb, bbb = 180 - self.delta_theta, 0 - self.delta_theta, 260 - self.delta_theta
        # se construye la otra parte del espectro angular reflejando sobre el eje x & y, la parte bien construida
        # si este se encuentra en los intervalos a, aa, aaa, b, bb, bbb, que son todos iguales a 0, 180 y 360 grados
        if self.theta in range(b, a + 1) or self.theta in range(bb, aa + 1) or self.theta in range(bbb, aaa + 1):
            mask1, mask2 = np.flip(phase_filter_mask, 0), np.flip(phase_filter_mask)
            mask2 = np.flip(mask2, 0)
            phase_filter_mask = np.bitwise_or(mask1, mask2)
            phase_filter_mask = np.flip(phase_filter_mask, 0)

        # cv2.imshow("1) Imagen fil1", 255*idx_1.astype(np.uint8))
        # cv2.imshow("1) Imagen Fil2", 255*idx_2.astype(np.uint8))

        return phase_filter_mask

    #  1. c)Un método filtering que implementa un filtrado vía FFT que permite solo el paso de las componentes de
    #  frecuencia orientadas entre 𝜃−Δ𝜃 y 𝜃-Δ𝜃 (anula las demás).
    def filtering(self):
        # Se realiza la FFT con su respectivo reordenamiento de frecuencias (shift)
        image_gray_fft_shift = np.fft.fftshift(np.fft.fft2(self.img_gris))

        phase_filter_mask = self.angularSpectral_mask()

        fft_filtered = image_gray_fft_shift * phase_filter_mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        maxi = np.max(image_filtered)
        if np.max(image_filtered) == 0:
            print('No hay componentes de frecuencia en este rango de orientaciones angulares')
            maxi = 1
        image_filtered /= maxi

        return image_filtered, phase_filter_mask

    # Punto 2:
    # Implemente un banco de 4 filtros direccionales utilizando la clase theta Filter
    # para las orientaciones 0°, 45°, 90°y 135°, y con Δ𝜃=5°, sobre una imagen de entrada.

    def fourFilter_banc(self, masks):
        # Filtrado realizado de la misma forma que en el punto anterior:
        # Retorna la imagen en el dominio original ya filtrada
        def convolvFreq(sig, mask):
            fft_filtered = sig * mask
            image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
            image_filtered = np.absolute(image_filtered)
            maxi = np.max(image_filtered)
            if np.max(image_filtered) == 0:
                print('No hay componentes de frecuencia en este rango de orientaciones angulares')
                maxi = 1
            image_filtered /= maxi

            return image_filtered

        # Se realiza la transformada de la imagen en grises una sola vez:
        image_gray_fft_shift = np.fft.fftshift(np.fft.fft2(self.img_gris))

        filterImages = []
        for i in masks:
            # Se realiza el filtrado en el dominio de la frecuencia, como se hizo para el primer punto
            img_filt = convolvFreq(image_gray_fft_shift, i)
            filterImages.append(img_filt)

        # 2. c) Sintetice una nueva imagen promediandolas imágenes resultantes después del banco de filtro y
        # visualice la imagen resultante.
        # Se calcula el promedio pixel a pixel sobre las 4 imágenes:
        imag_prom = np.mean(filterImages, axis=0)

        # 2. b) Visualice cada una de las imágenes con su respectivo titulo
        # se muestran todas las 4 imagenes filtradas, y la sintética
        cv2.imshow("2) Filtered image 0 grados", filterImages[0])
        cv2.imshow("2) Filtered image 45 grados", filterImages[1])
        cv2.imshow("2) Filtered image 90 grados", filterImages[2])
        cv2.imshow("2) Filtered image 135 grados", filterImages[3])
        cv2.imshow("2) Promedio Filtered image", imag_prom)
        cv2.waitKey(0)

        return imag_prom, filterImages

    # Punto 3:
    def std_N(self, imagen, N, thresh):
        imagen_n, imagen_n2 = cv2.blur(imagen, (N, N)), cv2.blur(np.power(imagen, 2), (N, N))
        imagen_std = np.sqrt(imagen_n2 - np.power(imagen_n, 2))
        imagen_std /= np.max(imagen_std)
        mask_std = imagen_std > thresh
        mask_std = mask_std.astype(float) #usar np.float en vez de float por si acaso
        return mask_std

    def std_plus_filter(self, filter_images, std_images):
        fin_image = 0
        for i, j in zip(filter_images, std_images):
            fin_image += i*j

        return fin_image


if __name__ == '__main__':

    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    # se pasa la imagen a grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. a) La clase recibe en su constructor la imagen (en grises) a filtrar:
    ob = thetaFilter(image_gray)
    # primer argumento de entrada theta, segundo delta theta:
    theta = 50
    delta_theta = 30
    ob.set_theta(theta, delta_theta)
    image_filtered, phase_filter_mask = ob.filtering()

    # Visualización de las imagenes del punto 1
    cv2.imshow("1) Imagen Original", image_gray)
    cv2.imshow("1) Filtro", phase_filter_mask * 255)
    cv2.imshow("1) Imagen Filtrada", image_filtered)
    cv2.waitKey(0)

    # 2. Segundo punto, banco de 4 filtros:
    thetas, delta_thetas = [0, 45, 90, 135], [20, 20, 20, 20]
    masks = []
    for i, j in zip(thetas, delta_thetas):
        ob.set_theta(i, j)
        masks.append(ob.angularSpectral_mask())

    ob.fourFilter_banc(masks)

    # 3. Tercer Punto opcional, banco de 4 filtros, sintesis usando desviación estándar:
    '''
    crear un método que permita sintetizar la imagen de la huella utilizando una máscara basada en la desviación 
    estándar local, para determinar que pixeles de la imágenes filtradas se incluyen en la imagen sintetizada. 
    Para la desviación estándar local defina un ventana rectangular y calcula la desviación estándar sobre la 
    imagen resultante del filtro de orientación. Utilizando un umbral predeterminado defina la mascara de 
    desviación estándar como los valores mayores al umbral. Utilice las distintas máscaras para ponderas las 
    imágenes filtradas de las diferentes orientaciones.
    '''

    thetas, delta_thetas = [0, 45, 90, 135], [20, 20, 20, 20]
    masks = []
    for i, j in zip(thetas, delta_thetas):
        ob.set_theta(i, j)
        masks.append(ob.angularSpectral_mask())

    #N: tamaño de la ventana cuadrada, thresh: umbral para obtener valores de desviación estándar superiores a éste
    N, thresh = 21, 0.5
    prom_img, filterImages = ob.fourFilter_banc(masks)
    std_image = []

    for i in filterImages:
        std_image.append(ob.std_N(i, N, thresh))

    fin_image = ob.std_plus_filter(filterImages, std_image)
    cv2.imshow("3) Sintesis STD", fin_image)
    cv2.waitKey(0)

