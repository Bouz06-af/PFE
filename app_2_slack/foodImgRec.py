import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle de classification d'images
model_path = 'best_model_101class.hdf5'
model = load_model(model_path, compile=False)

class_N = {}
N_class = {}
with open('classes.txt', 'r') as txt:
    classes = [i.strip() for i in txt.readlines()]
    class_N = dict(zip(classes, range(len(classes))))
    N_class = dict(zip(range(len(classes)), classes))
    class_N = {i: j for j, i in N_class.items()}

# Charger les données nutritionnelles
nutrition_data = pd.read_csv('nutrition101.csv')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (200, 200))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_label = N_class[class_index]
    return class_label

print(predict("test.jpg"))
predicted_class=predict("test.jpg").replace('_',' ')
nutrition_info = nutrition_data[nutrition_data['name'] == predicted_class]
print(nutrition_info)
if nutrition_info['sugar'].values[0] > 10:
    print("Compatibilité avec le diabète : Unsafe")
else:
    print("Compatibilité avec le diabète : Safe")