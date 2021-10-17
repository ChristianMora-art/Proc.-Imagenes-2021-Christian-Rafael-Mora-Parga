# Created by Christian Rafael Mora Parga
# TALLER 5, miércoles 19 de Octubre de 2021, 8pm
import cv2
import os
import sys
import numpy as np
from enum import Enum


class Methods(Enum):
    SIFT = 1
    ORB = 2


if __name__ == '__main__':

    path = sys.argv[1]

    # a) Recibir directorio de referencia
    images_N = os.listdir(path)
    print(images_N)
    N_images = len(images_N)
    print('El número de imagenes recibidas es: ', str(N_images))

    # b) Selección imagen de referencia, debe estar entre la primera y la cantidad total de imágenes
    if N_images > 3:
        n_ref = input('¿Cuál de las ' + str(N_images) + ' imágenes desea que sea de referencia?: ')
        n_ref = int(n_ref)
        assert (N_images > n_ref > 1), 'La imagen de referencia debe estar entre la primera y la imagen número ' + str(
            N_images)
    else:  # si son 3 imágenes de entrada, la imagen de referencia es la segunda
        n_ref = 2
        print('por lo tanto la imagen central es la 2')

    # c) Elección método de búsqueda puntos de interés y búsqueda de puntos de interés según método
    pin_mod = input('Elija el método de búsqueda de puntos de interés escribiendo: "1": SIFT o cualquier otra entrada de teclado: ORB: ')
    method = Methods.SIFT if pin_mod == '1' else Methods.ORB

    # a: imagen izquierda, "points_a_imgs"
    # b: imagen derecha, "points_b_imgs"
    points_a_imgs, points_b_imgs = list(), list()

    for i in range(N_images - 1):
        img_a, image_b = cv2.imread(os.path.join(path, images_N[i])), cv2.imread(os.path.join(path, images_N[i + 1]))
        img_gray_a, image_gray_b = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

        # sift/orb interest points and descriptors
        if method == Methods.SIFT:
            sift = cv2.SIFT_create(nfeatures=100)  # shift invariant feature transform
            keypoints_1, descriptors_1 = sift.detectAndCompute(img_gray_a, None)
            keypoints_2, descriptors_2 = sift.detectAndCompute(image_gray_b, None)
        else:
            orb = cv2.ORB_create(nfeatures=100)  # oriented FAST and Rotated BRIEF
            keypoints_1, descriptors_1 = orb.detectAndCompute(img_gray_a, None)
            keypoints_2, descriptors_2 = orb.detectAndCompute(image_gray_b, None)

        image_draw_1 = cv2.drawKeypoints(img_gray_a, keypoints_1, None)
        image_draw_2 = cv2.drawKeypoints(image_gray_b, keypoints_2, None)

        # Interest points matching
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descriptors_1, descriptors_2, k=1)
        image_matching = cv2.drawMatchesKnn(img_a, keypoints_1, image_b, keypoints_2, matches, None)

        # Retrieve matched points
        points_a, points_b = list(), list()
        for idx, match in enumerate(matches):
            idx2 = match[0].trainIdx
            points_a.append(keypoints_1[idx].pt)
            points_b.append(keypoints_2[idx2].pt)

        N = min(len(points_a), len(points_b))
        assert N >= 4, 'Se requieren al menos 4 puntos por imagen (4 rojos, 4 azules)'

        points_a, points_b = np.array(points_a[:N]), np.array(points_b[:N])

        points_a_imgs.append(points_a)
        points_b_imgs.append(points_b)

    # d) Hallar homografías de imágenes utilizando RANSAC
    H_con = list()
    counter = 0
    for i, j in zip(points_a_imgs, points_b_imgs):
        # i y j son las imágenes contiguas de la forma ej. para 4 imágenes: 1 2, 2 3, 3 4
        # Aquí los puntos de la imagen de la derecha son llevados a la de la izquierda:
        if counter >= (n_ref - 1): #H_mk
            H, mask = cv2.findHomography(j, i, cv2.RANSAC)
            #print('RL Homografía_', str(counter), '', str(counter - 1), H)
        else: #H_nm
            H, mask = cv2.findHomography(i, j, cv2.RANSAC)
            #print('LR Homografía_', str(counter - 1), '', str(counter), H)
        H_con.append(H)
        counter += 1

    # e) Hallar homografías particulares según la imagen de referencia

    indi_H_nm, indi_H_mk = list(), list()
    i_end = len(H_con[0:n_ref - 1])
    print('TODAS LAS HOMOGRAFÍAS', i_end, ', ', H_con)

    for i in range(i_end):
        partial_H = np.identity(3)
        for j in range(i, i_end):
            # Producto matricial para hallar las homografías compuestas (indirectas) de izquierda a derecha:
            partial_H = np.matmul(partial_H, H_con[j])
        indi_H_nm.append(partial_H)
        # print('Izquierda a derecha:', i, ', ', indi_H_nm)

    print('H_nm = ', indi_H_nm)

    # H_mm_1 = np.linalg.inv(H_con[n_ref])
    H_mm_1 = H_con[n_ref - 1]
    indi_H_mk.append(H_mm_1)
    for i in range(1, len(H_con) - i_end):
        partial_H = np.identity(3)
        for j in range(i_end - 1, i_end + i):
            # Producto matricial para hallar las homografías compuestas (indirectas) de derecha a izquierda:
            # partial_H = np.matmul(partial_H, np.linalg.inv(H_con[j + 1]))
            partial_H = np.matmul(partial_H, H_con[j + 1])
        indi_H_mk.append(partial_H)
        # print('Derecha a izquierda:', i, ', ', indi_H_mk)
    print('H_mk = ', indi_H_mk)

    # WARP:
    x_primes = list()
    for i in range(len(indi_H_nm)):
        image = cv2.imread(os.path.join(path, images_N[i]))
        image_h = cv2.warpPerspective(image, indi_H_nm[i], (image.shape[1], image.shape[0]))
        cv2.imshow("Image h_" + str(i + 1) + '' + str(n_ref), image_h)
        x_primes.append(image_h)
    # append de la imagen de referencia:
    cv2.imshow("Image h_" + str(n_ref), cv2.imread(os.path.join(path, images_N[n_ref - 1])))

    x_primes.append(cv2.imread(os.path.join(path, images_N[n_ref - 1])))

    for i in range(n_ref, n_ref + len(indi_H_mk)):
        image = cv2.imread(os.path.join(path, images_N[i]))
        image_h = cv2.warpPerspective(image, indi_H_mk[i - len(indi_H_nm) - 1], (image.shape[1], image.shape[0]))
        cv2.imshow("Image h_" + '' + str(n_ref) + str(i + 1), image_h)
        x_primes.append(image_h)
    cv2.waitKey(0)

    # PROMEDIO DE LAS IMÁGENES
    images_parc = list()
    for image in x_primes:
        image = image.astype(np.float64) / 255
        images_parc.append(image)

    # se añade constante np.finfo(float).eps para evitar división por 0
    deno = np.count_nonzero(np.array(images_parc), axis=0) + np.finfo(float).eps
    image_final = np.sum(np.array(images_parc), axis=0) / deno

    image_final = (255 * image_final).astype(np.uint8)

    cv2.imshow("Image Final", image_final)
    cv2.waitKey(0)
