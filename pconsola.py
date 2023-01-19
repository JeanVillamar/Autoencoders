
import numpy as np
import cargar_datos as cd
import entrenamiento as en
import modelo as mo
import time
#PROGRAMA PRINCIPAL

snn_report = classification_report(np.argmax(data[0], axis=1), model)  
print(snn_report)
ruta_archivo=input("Ingrese la ruta de la imágen:")
ase = cd.load_image(ruta_archivo)
im = mo.make_image(ase[0],ase[1],ase[2])
im.show()
im2_pred = en.predict(en.modelo_entrenado(), np.array(ase[0]))
#Normalizar si algo se pasa de 255 o de 0
im2_pred = np.clip(im2_pred, 0,255)
time.sleep(3)
print("...mejorando imágen")
time.sleep(3)
#Imagen aclarada
are = en.numpy_to_list(im2_pred)
a=mo.make_image(are , ase[1], ase[2])
a.show()
nombre_imagen=ruta_archivo.split("/")[-1]
print("Su imagen {} ha sido aclarada con éxito".format(nombre_imagen))
nuevo_nombre=input("Ingrese el nuevo nombre de su imágen mejorada")
a.save(nuevo_nombre)
print("Imágen guardada")