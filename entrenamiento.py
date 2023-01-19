
import keras
import numpy as np
import cargar_datos as cd
import entrenamiento as en
import modelo as mo
from sklearn.metrics import confusion_matrix, classification_report 


def train_model(train_data, valid_data, model):
    model.fit(x=train_data, 
              y=valid_data, 
              epochs=2, 
              validation_split=0.8, 
              batch_size=32,
              
              callbacks=[keras.callbacks.EarlyStopping(patience=5),
                       keras.callbacks.TensorBoard(log_dir='tf_logs', 
                                                   histogram_freq=0, 
                                                   batch_size=32, 
                                                   write_graph=True, 
                                                   write_images=True)]
    )

def predict(model, array_boxes):
    return model.predict(array_boxes)

def numpy_to_list(tensor_list):
    lista = []
    for tensor in tensor_list:
        lista.append(tensor)
    return lista

def modelo_entrenado():
    return model



model = mo.build_model()
data = cd.load_dataset()

en.train_model(np.array(data[1]), np.array(data[0]), model)
model.save_weights('v31-percent.h5')
model.load_weights('v31-percent.h5')

snn_report = classification_report(np.argmax(data[0], axis=1), model)  
print(snn_report)




