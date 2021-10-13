# Created by Christian Rafael Mora Parga
# TALLER 5, miércoles 13 de Octubre de 2021, 11am
import cv2
import os
import sys
import numpy as np

from enum import Enum


if __name__ == '__main__':

    path = sys.argv[1]

    # a)
    images_N = os.listdir(path)
    N_images = len(images_N)
    print('El número de imagenes recibidas es: ', str(N_images))

    # b)
    n_ref = input('¿Cuál de las ' + str(N_images) + ' imágenes desea que sea de referencia?: ')
    n_ref = int(n_ref)
    assert (N_images > n_ref > 1), 'La imagen de referencia debe estar entre la primera y la imagen número ' + str(N_images)
    print(n_ref)

    #n_ref = 2
    # c) Hallan los puntos para las proyecciones de las homografías

    def get_points_(image_draw, points, color):
        point_counter = 0
        point_m = list()
        while True:
            cv2.imshow("Image", image_draw)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("x"):
                point_m = points.copy()
                points = list()
                break
            if len(points) > point_counter:
                point_counter = len(points)
                cv2.circle(image_draw, (points[-1][0], points[-1][1]), 3, color, -1)
        return points, point_m

    points = list()

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    # a: imagen izquierda, "points_a_imgs"
    # b: imagen derecha, "points_b_imgs"
    points_a_imgs, points_b_imgs = list(), list()

    for i in range(N_images - 1):
        #print(i, i + 1)
        image_a = cv2.imread(os.path.join(path, images_N[i]))
        image_b = cv2.imread(os.path.join(path, images_N[i+1]))

        height_b, width_b, channels_b = image_b.shape
        #Unión de las imágenes izquierda y derecha
        im_h = cv2.hconcat([image_a, image_b])

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", click)

        #Color de los puntos
        points, points_a = get_points_(im_h, points, [0, 0, 255])
        points, points_b = get_points_(im_h, points, [255, 0, 0])

        N = min(len(points_a), len(points_b))
        assert N >= 4, 'Se requieren al menos 4 puntos por imagen (4 rojos, 4 azules)'
        #print('los de b', points_b)
        # Corrección de las coordenadas de la imagen izquierda, se le resta width_b:
        for j in range(len(points_b)):
            points_b[j] = (points_b[j][0] - width_b, points_b[j][1])
        #print('los de b AHORA ', points_b)

        points_a, points_b = np.array(points_a[:N]), np.array(points_b[:N])

        points_a_imgs.append(points_a)
        points_b_imgs.append(points_b)


    # d) Hallar homografías de imágenes contiguas
    H_con = list()
    counter = 0
    for i, j in zip(points_a_imgs, points_b_imgs):
        # i y j son las imagenes contiguas de la forma, para 4 imágenes: 1 2, 2 3, 3 4
        H, mask = cv2.findHomography(i, j, cv2.RANSAC)

        #Aquí los puntos de la imagen de la derecha son llevados a la de la izquierda:
        if counter >= (n_ref - 1):
            H, mask = cv2.findHomography(j, i, cv2.RANSAC)

        H_con.append(H)
        counter += 1
        print('Homografia_', str(counter-1), '', str(counter), H)

    cv2.destroyAllWindows()

    # e) Hallar homografías particulares según la imagen de referencia
    # Si se usan mas de 3 imágenes:
    if N_images > 3:
        indi_H_nm, indi_H_mk = list(), list()
        i_end = len(H_con[0:n_ref - 1])
        print(i_end, ', ', H_con)
        for i in range(i_end - 1):
            partial_H = np.identity(3)
            for j in range(i, i_end):
                # Producto matricial para hallar las homografías compuestas o indirectas de la derecha:
                partial_H = np.matmul(partial_H, H_con[j])
            indi_H_nm.append(partial_H)

        print('indi_H_nm = ', indi_H_nm)
        H_m_1m = H_con[n_ref - 2]
        indi_H_nm.append(H_m_1m)

        H_mm_1 = np.linalg.inv(H_con[n_ref])
        indi_H_mk.append(H_mm_1)
        for i in range(1, len(H_con) - i_end):
            partial_H = np.identity(3)
            for j in range(i_end - 1, i_end + i):
                # Producto matricial para hallar las homografías compuestas o indirectas de la izquierda:
                partial_H = np.matmul(partial_H, np.linalg.inv(H_con[j + 1]))
            indi_H_mk.append(partial_H)
        print('indi_H_mk = ', indi_H_mk)

        #image = cv2.imread(os.path.join(path, images_N[0]))
        #image_h = cv2.warpPerspective(image, indi_H_nm[0], (image.shape[1], image.shape[0]))

        # Producto entre las homografías y las imagenes originales
        # X'= H*X, X': x_primes
        x_primes = list()
        for i in range(len(indi_H_nm)):
            image = cv2.imread(os.path.join(path, images_N[i]))
            image_h = cv2.warpPerspective(image, indi_H_mk[i], (image.shape[1], image.shape[0]))
            cv2.imshow("Image h_" + str(i), image_h)
            x_primes.append(image_h)
        # append de la imagen de referencia:
        x_primes.append(cv2.imread(os.path.join(path, images_N[n_ref - 1])))

        for i in range(n_ref, n_ref + len(indi_H_mk)):
            image = cv2.imread(os.path.join(path, images_N[i]))
            image_h = cv2.warpPerspective(image, indi_H_mk[i - len(indi_H_nm) - 1], (image.shape[1], image.shape[0]))
            cv2.imshow("Image h_" + str(i), image_h)
            x_primes.append(image_h)
        cv2.waitKey(0)

    else: #cuando son 3 imágenes:
        x_primes = list()
        image = cv2.imread(os.path.join(path, images_N[0]))
        image_h = cv2.warpPerspective(image, H_con[0], (image.shape[1], image.shape[0]))

        x_primes.append(image_h)
        x_primes.append(cv2.imread(os.path.join(path, images_N[1])))

        image2 = cv2.imread(os.path.join(path, images_N[2]))
        image_h2 = cv2.warpPerspective(image2, np.linalg.inv(H_con[1]), (image.shape[1], image.shape[0]))
        x_primes.append(image_h2)
        cv2.imshow("Image h_" + str(0), image_h)
        cv2.imshow("Image h_" + str(1), cv2.imread(os.path.join(path, images_N[1])))
        cv2.imshow("Image h_" + str(2), image_h2)
        cv2.waitKey(0)

    #UNION DE LAS IMAGENES, promediado, etc
    #ref_dim = np.zeros(0)
    x_primes_copy = x_primes
    desplaza_x = 0
    for i in range(N_images - 1):
        px_s, px_sprima = list(), list()
        for ii, jj in zip(points_a_imgs[i], points_b_imgs[i]):
            px_s.append(ii[0])
            px_sprima.append(jj[0])
        # se superponen las siguientes dos componentes de las imagenes izquierda y derecha
        # es decir, la imagen izquierda queda encima de la derecha, utilizando como referencia x1 y xi':
        x1 = min(px_s) #componente x horizontal mínimo entre todos los puntos elegidos para la imagen izquierda
        x1_prima = min(px_sprima) #componente x horizontal mínimo entre todos los puntos elegidos para la imagen derecha

        if i == 0:
            x_primes_copy[i] = cv2.cvtColor(x_primes_copy[i], cv2.COLOR_BGR2HSV)
        # HSV de la matriz izquierda
        h, s, v = cv2.split(x_primes_copy[i])

        x_primes_copy[i + 1] = cv2.cvtColor(x_primes_copy[i + 1], cv2.COLOR_BGR2HSV)
        # HSV de la matriz derecha
        h2, s2, v2 = cv2.split(x_primes_copy[i + 1])

        end_in = len(x_primes[i + 1][0][:])
        # HSV de la matriz resultante. Se concatenan las dimensiones de las matrices izquierda y derecha teniendo en
        # cuenta la superposición entre ambas (lugar donde se intersectan y se hace el promedio entre ambas)
        hm = np.zeros((x_primes[i].shape[0], desplaza_x + x1 + len(x_primes[i + 1][0][x1_prima:end_in])))
        sm, vm = np.zeros_like(hm), np.zeros_like(hm)
        #hm = np.concatenate((hm, hm_zero), axis=1); sm = np.concatenate((sm, sm_zero), axis=1);vm = np.concatenate((vm, vm_zero), axis=1)

        #for row in range(i*h2.shape[0], hm.shape[0]):
        for row in range(hm.shape[0]):
            for col in range(hm.shape[1]):
                cond = col - x1 + x1_prima - desplaza_x
                if col <= x1 - (x1_prima + 1) + desplaza_x:
                    #c[row][col] = x_primes[i][row][col]
                    hm[row][col] = h[row][col]
                    sm[row][col] = s[row][col]
                    vm[row][col] = v[row][col]
                else:
                    if col <= x_primes_copy[i].shape[1] - 1:  # sección de superposición de a y b
                        # CÁLCULO DEL PROMEDIO:
                        # en caso de que existan ceros (IMAGEN NEGRA) se halla simplemente el mayor entre ambos arreglos:
                        if (h[row][col] == 0) or (s[row][col] == 0) or (s[row][col] == 0) or (h2[row][cond] == 0) or (s2[row][cond] == 0) or (s2[row][cond] == 0):
                            hm[row][col] = np.max([h[row][col], h2[row][cond]])
                            sm[row][col] = np.max([s[row][col], s2[row][cond]])
                            vm[row][col] = np.max([v[row][col], v2[row][cond]])
                        else:
                            hm[row][col] = np.mean([h[row][col], h2[row][cond]], dtype=np.uint8)
                            sm[row][col] = np.mean([s[row][col], s2[row][cond]], dtype=np.uint8)
                            vm[row][col] = np.mean([v[row][col], v2[row][cond]], dtype=np.uint8)
                    else:
                        # poner los elementos restantes de la matriz izquierda
                        #c[row][col] = x_primes[i+1][row][col - (x1 - x1_prima)]
                        hm[row][col] = h2[row][cond]
                        sm[row][col] = s2[row][cond]
                        vm[row][col] = v2[row][cond]
        # Se pasa de HSV a BGR
        hm, sm, vm = hm.astype(dtype=np.uint8), sm.astype(dtype=np.uint8), vm.astype(dtype=np.uint8)
        im3 = cv2.merge((hm, sm, vm))
        image_hue_bgr = cv2.cvtColor(im3, cv2.COLOR_HSV2BGR)
        #ref_dim = hm
        desplaza_x += x1 - (x1_prima - 1)
        x_primes_copy[i+1] = image_hue_bgr # se asigna la matriz resultante a la matriz en el índice actual de la próxima iteración

    cv2.imshow("Image Final", image_hue_bgr)
    cv2.waitKey(0)
