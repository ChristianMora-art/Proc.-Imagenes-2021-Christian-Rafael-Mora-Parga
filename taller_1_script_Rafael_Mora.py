import cv2
from taller_1_Rafael_Mora import basicColor

if __name__ == '__main__':

    ruta = input('Ingrese la ruta sin comillas de la imagen: ')
    objeto_clas = basicColor(ruta)
    objeto_clas.displayProperties()

    img_blanc = objeto_clas.makeBW()
    cv2.imshow("Imagen en blanco", img_blanc)
    cv2.waitKey(0)

    hue = input('Ingrese un valor de Hue entre 0 y 179: ')
    img_pros_hsv = objeto_clas.colorize(179)
    cv2.imshow("Imagen Hue mod", img_pros_hsv)
    cv2.waitKey(0)

#'/Users/mac/PycharmProjects/pythonProject/Images/cat.png'