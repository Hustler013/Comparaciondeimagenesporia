# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:37:12 2023

@author: Isaac
"""

import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f as f_distribution

# Leer datos desde el archivo Excel
Leonardo = pd.read_excel('SIFT1.xlsx', sheet_name='Hoja1') #1
Bing = pd.read_excel('SIFT1.xlsx', sheet_name='Hoja2')  #2
Dalle = pd.read_excel('SIFT1.xlsx', sheet_name='Hoja3') #3

# Realizar ANOVA
f_stat, p_value = f_oneway(Leonardo['Grupo1'], Bing['Grupo2'], Dalle['Grupo3'])

# Imprimir resultados
print("Estadística F:", f_stat)
print("Valor p:", p_value)

# Comprobar si se acepta o rechaza la hipótesis nula
alpha = 0.05  # Nivel de significancia

# Grados de libertad entre grupos y dentro de los grupos
df_between = 2  # Grados de libertad entre grupos
df_within = 297  # Grados de libertad dentro de los grupos
# Calcular el valor crítico de F
critical_value = f_distribution.ppf(1 - alpha, df_between, df_within)

print("Valor crítico para F:", critical_value)

if p_value < alpha:
    print("Se rechaza la hipótesis nula: Hay diferencias significativas entre los grupos.")
else:
    print("No se rechaza la hipótesis nula: No hay suficiente evidencia para concluir diferencias significativas entre los grupos.")

# Gráfico de caja y bigotes para cada grupo
plt.boxplot([Leonardo['Grupo1'], Bing['Grupo2'], Dalle['Grupo3']], labels=['Leonardo', 'Bing', 'Dalle'])
plt.xlabel('Grupos')
plt.ylabel('Valores')
plt.title('Diagrama de caja y bigotes para SIFT')
plt.show()

"""
# Gráfico de densidad para cada grupo
sns.kdeplot(df1['Grupo1'], fill=True, label='Grupo 1')
sns.kdeplot(df2['Grupo2'], fill=True, label='Grupo 2')
#sns.kdeplot(df3['Grupo3'], fill=True, label='Grupo 3')
"""

#SIFT
#HO= (LEONARDO Y BING) (LEONARDO Y DALLE)
#H1= (LEONARDO Y BING Y DALLE) (BING Y DALLE)

#ORB
#HO= (LEONARDO Y BING Y DALLE) (LEONARDO Y DALLE)
#H1= (LEONARDO y BING) (BING Y DALLE)


