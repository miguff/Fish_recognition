#Importáljuk a könyvtárakat
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image


from tensorflow import keras
import pathlib


def main():
    
    #-----------------------------------------------------------------------------------------------------
    #modell betöltése
    #----------------------------------------------------------------------------------------------------
    
    TF_MODEL_FILE_PATH = "fish_recognition_V1.tflite"
    interpreter = tf.lite.Interpreter(model_path = TF_MODEL_FILE_PATH)
    print(interpreter.get_signature_list())
    classify_lite = interpreter.get_signature_runner('serving_default')

    #---------------------------------------------------------------------------------
    #Képek beállítása
    #---------------------------------------------------------------------------------
    img_height = 180
    img_widht = 180
   
    class_names = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 'Fourfinger Threadfin', 'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 'Gourami', 'Grass Carp', 'Green Spotted Puffer', 'Indian Carp', 'Indo-Pacific Tarpon', 'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish', 'Mosquito Fish', 'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp', 'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']
    #--------------------------------------------------------------------------------
    #Test kép tesztelése
    #--------------------------------------------------------------------------------
    im_path = r"E:\PROGRAMOZÁS\FISH_RECOGNITION\FishImgDataset\test\Indian Carp\Catla_2_jpg.rf.f7ab40aea15b16f83e938dc692b236fb.jpg"
    #Kép betöltése
    img = tf.keras.utils.load_img(
    im_path, target_size=(img_height, img_widht)
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)    

    #Itt megézzük, hogy a modell minek gondolja
    predictions_lite = classify_lite(rescaling_1_input=img_array)['dense_1']
    score_lite = tf.nn.softmax(predictions_lite)

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
    )


if __name__ == "__main__":
    main()

