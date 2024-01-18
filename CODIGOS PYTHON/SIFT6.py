# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:16:50 2023

@author: Isaac
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from skimage.feature import match_descriptors, plot_matches, SIFT

import cv2
import glob

import time
    
# Código que quieres medir
start_time = time.time()

# Ruta donde se encuentran las imágenes (cambia por tu ruta)
carpeta_imagenes_español = 'C:/Users/Isaac/OneDrive - Universidad Politecnica Salesiana/Castro - Morales/FOTOS/DALLE/ESP1/*.jpg'
carpeta_imagenes_ingles = 'C:/Users/Isaac/OneDrive - Universidad Politecnica Salesiana/Castro - Morales/FOTOS/DALLE/ENG1/*.jpg'

# Lista para almacenar las imágenes
img1 = []
img2 = []

# Obtener la lista de nombres de archivo de las imágenes en la carpeta
archivos_imagenes1 = glob.glob(carpeta_imagenes_español)
archivos_imagenes2 = glob.glob(carpeta_imagenes_ingles)

# Recorrer la lista de nombres de archivo y leer las imágenes
for nombre_archivo in archivos_imagenes1:
    # Imprimir el nombre de la imagen
    #print(f"Leyendo: {nombre_archivo}")
    # Leer la imagen y añadirla a la lista
    imagen = cv2.imread(nombre_archivo, cv2.IMREAD_GRAYSCALE)
    #imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    if imagen is not None:
        img1.append(imagen)
        
for nombre_archivo in archivos_imagenes2:
    # Imprimir el nombre de la imagen
    #print(f"Leyendo: {nombre_archivo}")
    # Leer la imagen y añadirla a la lista
    imagen = cv2.imread(nombre_archivo, cv2.IMREAD_GRAYSCALE)
    #imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    if imagen is not None:
        img2.append(imagen)

tform = transform.AffineTransform(scale=(1.3, 1.1), rotation=0.5, translation=(0, -200))

# Definición de extractor SIFT
descriptor_extractor = SIFT()

# Variables para keypoints y descriptores
keypoints1 = []
keypoints2 = []
descriptors1 = []
descriptors2 = []

# Procesar las imágenes para obtener keypoints y descriptores de las imágenes en español
for img in img1:
    descriptor_extractor.detect_and_extract(img)
    keypoints1.append(descriptor_extractor.keypoints)
    descriptors1.append(descriptor_extractor.descriptors)

# Procesar las imágenes para obtener    keypoints y descriptores de las imágenes en inglés
for img in img2:
    descriptor_extractor.detect_and_extract(img)
    keypoints2.append(descriptor_extractor.keypoints)
    descriptors2.append(descriptor_extractor.descriptors)

#Comparación Descriptores Imagen 1 con Imagen 2
for i in range(len(img1)):  
    matches = match_descriptors(descriptors1[i], descriptors2[i], max_ratio=0.6, cross_check=True)
    #print(descriptors1[i])
    n=510 #510 numero de descriptores
    d=np.zeros((n,1))
    for j, match in enumerate(matches):
        idx1, idx2 = match  # Índices del descriptor en la primera y segunda imagen
        if idx1 < len(descriptors1[i]) and idx2 < len(descriptors2[i]):
            d[j] = np.linalg.norm(descriptors1[i][idx1] - descriptors2[i][idx2])
        else:
            print("Index out of bounds or descriptors not found")

    D = np.mean(d)
    #print(D)
    print(f"Distancia del emparejamiento {i+1}: {D}")

# Comparación Descriptores Imagen 1 con Imagen 2
for i in range(len(img1)):
    matches12 = match_descriptors(descriptors1[i], descriptors2[i], max_ratio=0.6, cross_check=True)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 8))
    plt.gray()

    plot_matches(ax[0], img1[i], img2[i], keypoints1[i], keypoints2[i], matches12)
    ax[0].axis('off')
    ax[0].set_title("Imagen Español vs. Imagen Ingles\n"
                   "(todos los keypoints y coincidencias)")

    plot_matches(ax[1], img1[i], img2[i], keypoints1[i], keypoints2[i], matches12[::15], only_matches=True)
    ax[1].axis('off')
    ax[1].set_title("Imagen Español vs. Imagen Ingles\n"
                   "(subconjunto de coincidencias para visibilidad)")

    plt.tight_layout()
    plt.show()

end_time = time.time()
# Tiempo transcurrido en segundos
elapsed_time = end_time - start_time

# Convertir a minutos y segundos
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print(f"El tiempo transcurrido es: {minutes} minutos y {seconds} segundos")