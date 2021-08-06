# Created by Christian Rafael Mora Parga
import cv2
from taller_1_Rafael_Mora import basicColor

if __name__ == '__main__':

    #se pide al usuario la ruta de la imagen (absolura, o relativa si se halla en el mismo lugar del script)
    ruta = input('Ingrese la ruta sin comillas de la imagen: ')
    #se crea un objeto de la clase basicColor, y se ingresa la ruta
    objeto_clas = basicColor(ruta)

    #llamado del primer método
    objeto_clas.displayProperties()

    # llamado del segundo método que retorma su self para mostrar aquí la imagen
    img_blanc = objeto_clas.makeBW()
    cv2.imshow("Imagen en blanco", img_blanc)
    cv2.waitKey(0)

    # llamado del tercer método y solicitud de ingreso del parámetro hue
    # que retorma su self para mostrar aquí la imagen
    hue = int(input('Ingrese un valor de Hue entre 0 y 179: '))
    img_pros_hsv = objeto_clas.colorize(hue)
    cv2.imshow("Imagen Hue mod", img_pros_hsv)
    cv2.waitKey(0)

#'/Users/mac/PycharmProjects/pythonProject/Images/cat.png'