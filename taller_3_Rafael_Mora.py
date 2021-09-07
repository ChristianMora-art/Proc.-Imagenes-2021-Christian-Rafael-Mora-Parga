# Created by Christian Rafael Mora Parga
import cv2
import numpy as np
import os
import sys

class taller_III():
    def __init__(self, D, I, img_gris):
        self.img_gris = img_gris  # recibe la imagen en grises
        self.D_factor = D
        self.I_factor = I
        self.mask = None

    #Para el diezmado y la interpolación se utiliza un filtro LOW PASS
    def low_pass_mask(self, DiezInter, imag):
        # pre-computations
        num_rows, num_cols = (imag.shape[0], imag.shape[1])
        enum_rows, enum_cols = np.linspace(0, num_rows - 1, num_rows), np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_size = num_rows / 2 - 1
        # low pass filter mask
        low_pass_mask = np.zeros_like(imag)
        if DiezInter:
            freq_cut_off = 1 / self.D_factor
        else:
            freq_cut_off = 1 / self.I_factor

        radius_cut_off = int(freq_cut_off * half_size)
        idx_lp = np.sqrt((col_iter - half_size) ** 2 + (row_iter - half_size) ** 2) < radius_cut_off
        low_pass_mask[idx_lp] = 1

        return low_pass_mask

    # 1) Implemente un método de diezmado por un factor D utilizando la FFT:
    def diezmar(self, imagen):
        #Primero se filtra luego se diezma
        #Se genera la máscara del LP según el factor de diezmado D "self.D_factor"
        low_pass_mask = self.low_pass_mask(True, imagen)
        #Filtrado en el dominio espectral
        image_gray_fft_shift = np.fft.fftshift(np.fft.fft2(imagen))
        fft_filtered = image_gray_fft_shift * low_pass_mask
        #Transformada inversa:
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        #Diezmado:
        imagen_decimada = image_filtered[::self.D_factor, ::self.D_factor]
        #Para comparar qué sucede sin el filtrado:
        #imagen_decimada = self.img_gris[::self.D_factor, ::self.D_factor]; cv2.imshow("FFT", 255*low_pass_mask)
        #cv2.imshow("imagen_decimada", imagen_decimada); cv2.waitKey(0)

        return imagen_decimada

    # 2) Implemente un método de interpolación por un factor I utilizando la FFT:
    def interpolar(self, imagen):
        #Primero se interpola, luego se filtra
        filas, colms = imagen.shape
        num_ceros = self.I_factor - 1
        imagen_ceros = np.zeros((num_ceros*filas, num_ceros*colms), dtype=imagen.dtype)
        #Inteporlación:
        imagen_ceros[::num_ceros, ::num_ceros] = imagen

        # Se genera la máscara del LP según el factor de interpolación I "self.I_factor"
        low_pass_mask = self.low_pass_mask(False, imagen_ceros)
        #Filtrado en el dominio espectral
        image_gray_fft_shift = np.fft.fftshift(np.fft.fft2(imagen_ceros))
        fft_filtered = image_gray_fft_shift * low_pass_mask
        #Transformada inversa:
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_interpolada = image_filtered / np.max(image_filtered)

        return image_interpolada

    # 3 y 4) Implemente un método de descomposición utilizando un banco de filtros_
    def descompo(self, N):
        #Definición de los kernels/filtros:
        H = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        D = np.array([[2, -1, -2], [-1, 4, -1], [-2, -1, 2]])
        L = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])

        # Primer descomposición:
        image_conv_H, image_conv_V  = cv2.filter2D(self.img_gris, -1, H), cv2.filter2D(self.img_gris, -1, V)
        image_conv_D, image_conv_L = cv2.filter2D(self.img_gris, -1, D), cv2.filter2D(self.img_gris, -1, L)
        #Se diezma por un factor de 2
        self.D_factor = 2
        image_conv_H, image_conv_V = self.diezmar(image_conv_H), self.diezmar(image_conv_V)
        image_conv_D, image_conv_L = self.diezmar(image_conv_D), self.diezmar(image_conv_L)

        # Se realizan las siguientes N-1 Descomposiciones de forma recursiva:
        def banco_descompo(imag, orden, list_images):
            image_conv_H, image_conv_V = cv2.filter2D(imag, -1, H), cv2.filter2D(imag, -1, V)
            image_conv_D, image_conv_L = cv2.filter2D(imag, -1, D), cv2.filter2D(imag, -1, L)

            image_conv_H, image_conv_V = self.diezmar(image_conv_H), self.diezmar(image_conv_V)
            image_conv_D, image_conv_L = self.diezmar(image_conv_D), self.diezmar(image_conv_L)
            level = [image_conv_H, image_conv_V, image_conv_D, image_conv_L]
            # se anexan las nuevas imagenes filtradas apartir de la nueva ingresada en cada iteración recursiva:
            list_images.append(level)
            if (orden > 0):
                #en el punto 3 y 4 se pide hacer la decomposición a partir del filtro L:
                return banco_descompo(image_conv_L, orden - 1, list_images)

        #Se ingresa la primera imagen, el orden de descomposición y una lista que será llenada de imagenes de descomposición
        #Cada fila (N) de "imagenes_decomp" es un nivel de descomposición, y cada columna (4) es un filtro H,V,D y L en
        # dicho nivel N de descomposición.
        imagenes_descomp = list()
        imagenes_descomp.append([image_conv_H, image_conv_V, image_conv_D])
        banco_descompo(image_conv_L, N - 2, imagenes_descomp)
        '''cv2.imshow("conv L", imagenes_descomp[0][0])
                cv2.imshow("conv LL", imagenes_descomp[1][0])
                cv2.imshow("conv LLL", imagenes_descomp[2][0]) 
        #  ETC...
                cv2.waitKey(0)'''
        return imagenes_descomp

if __name__ == '__main__':

    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     # se pasa la imagen a grises

    #Factores de Diezmado (D_factor) e interpolación (I_factor):
    D_factor = 4
    I_factor = 5
    ob = taller_III(D_factor, I_factor, image_gray)

    # Punto 1
    diezmado = ob.diezmar(image_gray)
    cv2.imshow('Punto 1, imagen diezmada', diezmado)
    cv2.waitKey(0)

    # Punto 2
    interpolado = ob.interpolar(image_gray)
    cv2.imshow('Punto 2, imagen interpolada', interpolado)
    cv2.waitKey(0)

    # Punto 3
    orden_decomp = 3
    banco_filtros = ob.descompo(orden_decomp)
    #Observar internamente a banco_filtros

    # Punto 4
    orden_decomp = 2
    banco_filtros = ob.descompo(orden_decomp)

    ob.I_factor = 4 #interpolar a un factor de 4
    i = 0
    for lvl_descomp in banco_filtros:
        i += 1
        j = 0
        for filter_in_lvl in lvl_descomp:
            j += 1
            title = 'nivel descomp ' + str(i) + ', filter ' + str(j)
            cv2.imshow(title, ob.interpolar(filter_in_lvl))

    cv2.waitKey(0)



