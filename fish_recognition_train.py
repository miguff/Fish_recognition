
#Importáljuk a könyvtárakat
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
#import matplotlib.pyplot as plt
import sys


def main():
    
    #Könyvtárak beállítása
    #--------------------------------------------------------------------------------
    data_dir_train = r"E:\PROGRAMOZÁS\FISH_RECOGNITION\FishImgDataset\train"
    data_dir_train = pathlib.Path(data_dir_train)

    data_dir_test = r"E:\PROGRAMOZÁS\FISH_RECOGNITION\FishImgDataset\test"
    data_dir_test = pathlib.Path(data_dir_test)

    data_dir_val = r"E:\PROGRAMOZÁS\FISH_RECOGNITION\FishImgDataset\val"
    data_dir_val = pathlib.Path(data_dir_val)
    #---------------------------------------------------------------------------------
    #Képek beállítása
    #---------------------------------------------------------------------------------
    batch_size = 16
    img_height = 180
    img_widht = 180
    #----------------------------------------------------------------------------------
    #Képek beállítása, hogy lehessen trainelni
    #----------------------------------------------------------------------------------
    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_train,
    image_size=(img_height, img_widht),
    batch_size=batch_size,
    seed=123)

    test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_test,
    image_size=(img_height, img_widht),
    batch_size=batch_size,
    seed=123)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_val,
    image_size=(img_height, img_widht),
    batch_size=batch_size,
    seed=123)

    class_names = train_ds.class_names
    print(class_names)
    sys.exit()
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    #-----------------------------------------------------------------------------------
    #Modell felállítása
    #-----------------------------------------------------------------------------------
    num_classes = len(class_names)

    model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_widht, 3)),
    layers.Conv2D(16, 3, padding ="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes)
    ])

    #-------------------------------------------------------------------------------------
    #Modell complie-ing
    #-------------------------------------------------------------------------------------
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.summary()
    #--------------------------------------------------------------------------------------
    #Modell trainelése
    #--------------------------------------------------------------------------------------
    epochs = 10
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
    )

    #--------------------------------------------------------------------------------------
    #Vizualizáció
    #--------------------------------------------------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    #--------------------------------------------------------------------------
    #Modell átalakítása Lite objektummá, hogy bárhol feltudjam használni
    #--------------------------------------------------------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(r"E:\PROGRAMOZÁS\FISH_RECOGNITION\FishImgDataset\fish_recognition_V1.tflite", "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    main()