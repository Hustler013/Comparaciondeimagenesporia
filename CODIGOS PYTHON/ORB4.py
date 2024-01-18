# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:47:28 2023

@author: Isaac
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_descriptors, plot_matches
import glob

import time

# Código que quieres medir
start_time = time.time()

# Rutas de las carpetas de imágenes en español e inglés
carpeta_imagenes_espanol = 'C:/Users/Isaac/OneDrive - Universidad Politecnica Salesiana/Castro - Morales/FOTOS/BING/ESP/*.jpg'
carpeta_imagenes_ingles = 'C:/Users/Isaac/OneDrive - Universidad Politecnica Salesiana/Castro - Morales/FOTOS/BING/ENG/*.jpg'

# Lista para almacenar las imágenes
img_espanol = []
img_ingles = []

# Obtener la lista de nombres de archivo de las imágenes en las carpetas
archivos_imagenes_espanol = glob.glob(carpeta_imagenes_espanol)
archivos_imagenes_ingles = glob.glob(carpeta_imagenes_ingles)

# Leer las imágenes de la carpeta en español
for nombre_archivo in archivos_imagenes_espanol:
    imagen = cv2.imread(nombre_archivo, cv2.IMREAD_GRAYSCALE)
    if imagen is not None:
        img_espanol.append(imagen)

# Leer las imágenes de la carpeta en inglés
for nombre_archivo in archivos_imagenes_ingles:
    imagen = cv2.imread(nombre_archivo, cv2.IMREAD_GRAYSCALE)
    if imagen is not None:
        img_ingles.append(imagen)

# Crear el detector ORB
orb = cv2.ORB_create()

# Lista para almacenar keypoints y descriptores de las imágenes
keypoints_espanol = []
descriptores_espanol = []
keypoints_ingles = []
descriptores_ingles = []

# Procesar las imágenes en español para obtener keypoints y descriptores
for img in img_espanol:
    kp, des = orb.detectAndCompute(img, None)
    keypoints_espanol.append(kp)
    descriptores_espanol.append(des)

# Procesar las imágenes en inglés para obtener keypoints y descriptores
for img in img_ingles:
    kp, des = orb.detectAndCompute(img, None)
    keypoints_ingles.append(kp)
    descriptores_ingles.append(des)
    

# Comparación de descriptores de imágenes en español e inglés
for i in range(min(len(descriptores_espanol), len(descriptores_ingles))):
    matches = match_descriptors(descriptores_espanol[i], descriptores_ingles[i], cross_check=True)
    
    distancia_descriptores = []
    for match in matches:
        idx_espanol, idx_ingles = match  # Índices de los descriptores en español e inglés
        distancia = np.linalg.norm(descriptores_espanol[i][idx_espanol] - descriptores_ingles[i][idx_ingles])
        distancia_descriptores.append(distancia)
    
    distancia_media = np.mean(distancia_descriptores)
    print(f"Distancia media de descriptores entre imágenes {i+1}: {distancia_media}")
    
    
# Comparar los descriptores de las imágenes en español con las imágenes en inglés
for i in range(len(img_espanol)):
    matches = match_descriptors(descriptores_espanol[i], descriptores_ingles[i], cross_check=True)
       
    # Visualizar los emparejamientos
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 8))
    plt.gray()

    keypoints_np_espanol = np.array([kp.pt for kp in keypoints_espanol[i]])
    keypoints_np_ingles = np.array([kp.pt for kp in keypoints_ingles[i]])

    plot_matches(ax[0], img_espanol[i], img_ingles[i], keypoints_np_espanol, keypoints_np_ingles, matches)
    ax[0].axis('off')
    ax[0].set_title("Imagen Español vs. Imagen Inglés\n(todos los keypoints y coincidencias)")

    plot_matches(ax[1], img_espanol[i], img_ingles[i], keypoints_np_espanol, keypoints_np_ingles, matches[::15], only_matches=True)
    ax[1].axis('off')
    ax[1].set_title("Imagen Español vs. Imagen Inglés\n(subconjunto de coincidencias para visibilidad)")

    plt.tight_layout()
    plt.show()
    
end_time = time.time()
# Tiempo transcurrido en segundos
elapsed_time = end_time - start_time

# Convertir a minutos y segundos
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"El tiempo transcurrido es: {minutes} minutos y {seconds} segundos")