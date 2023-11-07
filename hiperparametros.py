#se importan las dependencias
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

#se abre el archivo que se va a manejar (horse-colic.csv)
file = pd.read_csv('horse-colic.csv', names=[
    'surgery'	,
    'Age'	,
    'Hospital Number'	,
    'rectal temperature'	,
    'pulse' 	,
    'respiratory rate',
    'temperature of extremities',
    'peripheral pulse',
    'mucous membranes',
    'capillary refill time',
    'pain',
    'peristalsis',
    'abdominal distension',
    'nasogastric tube',
    'nasogastric reflux',
    'nasogastric reflux PH',
    'rectal examination',
    'abdomen',
    'packed cell volume',
    'total protein',
    'abdominocentesis appearance',
    'abdomcentesis total protein',
    'outcome',
    'surgical lesion',
    'cp_data'
])

#se crea el dataframe
df = pd.DataFrame(file)

#se reemplazan los signos de interrogacion de los datos faltantes por None(null)
df = df.replace("?",np.NaN)

df[[
    'surgery'	,
    'Age'	,
    'Hospital Number'	,
    'rectal temperature'	,
    'pulse' 	,
    'respiratory rate',
    'temperature of extremities',
    'peripheral pulse',
    'mucous membranes',
    'capillary refill time',
    'pain',
    'peristalsis',
    'abdominal distension',
    'nasogastric tube',
    'nasogastric reflux',
    'nasogastric reflux PH',
    'rectal examination',
    'abdomen',
    'packed cell volume',
    'total protein',
    'abdominocentesis appearance',
    'abdomcentesis total protein',
    'outcome',
    'surgical lesion',
    'cp_data',
]] = df[[
    'surgery'	,
    'Age'	,
    'Hospital Number'	,
    'rectal temperature'	,
    'pulse' 	,
    'respiratory rate',
    'temperature of extremities',
    'peripheral pulse',
    'mucous membranes',
    'capillary refill time',
    'pain',
    'peristalsis',
    'abdominal distension',
    'nasogastric tube',
    'nasogastric reflux',
    'nasogastric reflux PH',
    'rectal examination',
    'abdomen',
    'packed cell volume',
    'total protein',
    'abdominocentesis appearance',
    'abdomcentesis total protein',
    'outcome',
    'surgical lesion',
    'cp_data',
]].astype(float)

df = df.drop(columns=['nasogastric tube', 'nasogastric reflux', 'nasogastric reflux PH', 'rectal examination', 'abdomen', 'abdominocentesis appearance', 'abdomcentesis total protein'])

## media si es valor (temperatura, cntidad de proteina), moda si es categorica (estado civil, etc)
moda = pd.Series(df[["surgery"]].values.flatten()).mode()[0]
df["surgery"] = df['surgery'].fillna(moda)

moda1 = pd.Series(df[["temperature of extremities"]].values.flatten()).mode()[0]
df["temperature of extremities"] = df['temperature of extremities'].fillna(moda1)

moda2 = pd.Series(df[["peripheral pulse"]].values.flatten()).mode()[0]
df["peripheral pulse"] = df['peripheral pulse'].fillna(moda2)

moda3 = pd.Series(df[["mucous membranes"]].values.flatten()).mode()[0]
df["mucous membranes"] = df['mucous membranes'].fillna(moda3)

moda4 = pd.Series(df[["capillary refill time"]].values.flatten()).mode()[0]
df["capillary refill time"] = df['capillary refill time'].fillna(moda4)

moda5 = pd.Series(df[["pain"]].values.flatten()).mode()[0]
df["pain"] = df['pain'].fillna(moda5)

moda6 = pd.Series(df[["peristalsis"]].values.flatten()).mode()[0]
df["peristalsis"] = df['peristalsis'].fillna(moda6)

moda7 = pd.Series(df[["abdominal distension"]].values.flatten()).mode()[0]
df["abdominal distension"] = df['abdominal distension'].fillna(moda7)

moda8 = pd.Series(df[["outcome"]].values.flatten()).mode()[0]
df["outcome"] = df['outcome'].fillna(moda8)

media = pd.Series(df[["rectal temperature"]].values.flatten()).mean()
df["rectal temperature"] = df['rectal temperature'].fillna(media)

media1 = int(pd.Series(df[['pulse']].values.flatten()).mean())
df['pulse'] = df['pulse'].fillna(media1)

media2 = int(pd.Series(df[['respiratory rate']].values.flatten()).mean())
df['respiratory rate'] = df['respiratory rate'].fillna(media2)

media3 = int(pd.Series(df[['packed cell volume']].values.flatten()).mean())
df['packed cell volume'] = df['packed cell volume'].fillna(media3)

media4 = int(pd.Series(df[['total protein']].values.flatten()).mean())
df['total protein'] = df['total protein'].fillna(media4)

#se calcula el porcentaje de datos faltantes por columna en el dataframe
na_ratio = ((df.isnull().sum() / len(df))*100)

def atipicos(columna,ubicacion,valor,mda):
    val = int(df[columna+""].size)

    i = 0
    while i < val:
        if ubicacion == "arriba":
            if df.loc[i,columna+""] > valor:
                df.loc[i,columna+""] = mda
        if ubicacion == "abajo":
            if df.loc[i,columna+""] < valor:
                df.loc[i,columna+""] = mda
        i += 1

df = df.drop(columns=['cp_data','Hospital Number'])

#dataframe con las variables cuantitativas para determinar el coeficiente de pearson
dfc = df.drop(columns=['surgery','Age','temperature of extremities','peripheral pulse','mucous membranes','capillary refill time',
    'pain','peristalsis','abdominal distension','outcome','surgical lesion'])

atipicos("rectal temperature","arriba",41,media)
atipicos("rectal temperature","abajo",37,media)
atipicos("pulse","arriba",120,media1)
atipicos("packed cell volume","abajo",30,media3)

print(df["outcome"].value_counts())
dfsvm = df
dfsvm["outcome"] = dfsvm['outcome'].replace(1,0)
dfsvm["outcome"] = dfsvm['outcome'].replace(2,1)
dfsvm["outcome"] = dfsvm['outcome'].replace(3,1)
print(dfsvm["outcome"].value_counts())

x = dfsvm.drop(columns=['outcome'])
y = dfsvm['outcome']
y = y.astype(int)  # Asegúrate de que sea una variable numérica

# se aplica sleckbest de forma que se seleccione las 5 mejores variables
best = SelectKBest( k = 5 )
features = array(x.columns)
X_new = best.fit_transform(x, y)
filter = best.get_support()
x_new2 = pd.DataFrame(X_new, columns = features[filter])

def crearBarras (etiquetas, y):
  count_classes = pd.value_counts(y, sort = True)
  count_classes.plot(kind = 'bar', rot=0)
  plt.xticks(range(2), etiquetas)
  plt.title("Frecuencia de acuerdo al número de observaciones")
  plt.xlabel("Outcome")
  plt.ylabel("cantidad de observaciones")

etiquetas= ['0','1']
crearBarras ( etiquetas, y)

"""
División de entrenamiento y prueba:
Divide tus datos en conjuntos de entrenamiento y prueba usando train_test_split:
"""

X_train, X_test, y_train, y_test = train_test_split(x_new2, y, train_size=0.7, random_state = 1)

param_grid = {'C': np.logspace(-5, 7, 20)}

# Búsqueda por validación cruzada
# ==============================================================================
grid_search = GridSearchCV(
#'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        estimator  = SVC(kernel= 'linear',
        gamma='scale'),
        param_grid = param_grid,
        scoring    = 'accuracy',
        n_jobs     = -1,
        cv         = 3,
        verbose    = 0,
        return_train_score = True
      )

res = grid_search.fit(X_train, y_train)
print (res)
print (grid_search.best_params_)
svm_model = SVC(kernel='linear', C=grid_search.best_params_['C'])
svm_model.fit(X_train, y_train)
# Imprimir la precisión del modelo en el conjunto de datos de prueba
y_pred = svm_model.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))