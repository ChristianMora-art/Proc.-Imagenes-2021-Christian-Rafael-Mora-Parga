import cv2
import numpy as np


class basicColor:
    def __init__(self, ruta_img, imagen=None):
        self.path = ruta_img

        if imagen == None:
            self.image = None

    def displayProperties(self):

        # Se lee la imagen
        self.image = cv2.imread(self.path)
        # Se revisa que sea válida la imagen
        assert self.image is not None, "There is no image at {}".format(self.path)

        dimensions = self.image.shape
        tam_mega_pixels = (dimensions[0]*dimensions[1])/10E6
        print('Tamano:', dimensions[0], 'x', dimensions[1], '=', tam_mega_pixels, 'MP')
        print('Canales:', dimensions[2])
        return dimensions

    def makeBW(self):
        image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Hubral global de Otsu
        ret, Ibw_otsu = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imshow("Image", Ibw_otsu); cv2.waitKey(0)
        #Retorna la imagen bin:
        return Ibw_otsu

    def colorize(self, hue):
        # hue debe estar entre 0 y 179
        if hue > 179:
            hue = 179
        elif hue < 0:
            hue = 0
        print('hue =', hue)
        # Pasa de BGR a HSV
        image_hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image_hsv)

        # lo pasa a un vector de unos, y multiplica cada pixel por hue
        h = hue*np.ones_like(h)
        image_hue = cv2.merge((h, s, v))
        #Pasa de HSV a BGR
        image_hue_bgr = cv2.cvtColor(image_hue, cv2.COLOR_HSV2BGR)
        #cv2.imshow("Image", image_hue_bgr); cv2.waitKey(0)
        #Retorna la imagen con hue modificado
        return image_hue_bgr
