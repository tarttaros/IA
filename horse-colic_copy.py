#se importan las dependencias
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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

#se calcula el porcentaje de datos faltantes por columna en el dataframe
na_ratio = ((df.isnull().sum() / len(df))*100)

print("porcentaje de nulls antes de la eliminacion de las columnas con mas del 30% de valores nullos ----------------------------------------------------")

#se imprime el dataframe
print(na_ratio)

#print(df.isnull())

print("graficas antes de rellenar las columnas ----------------------------------------------------")

#se realizan las graficas de cajas y bigotes con sus datos atipicos
"""
category_orders = {"outcome": ["1", "2", "3"]}

fig = px.box(df, x="outcome", y="rectal temperature", color='outcome', points = 'all', title='Boxplot variable rectal temperature',category_orders=category_orders)
fig.show()

fig1 = px.box(df, x='outcome', y='pulse', color='outcome', points = 'all', title='Boxplot variable pulse',category_orders=category_orders)
fig1.show()
"""
df = df.drop(columns=['nasogastric tube', 'nasogastric reflux', 'nasogastric reflux PH', 'rectal examination', 'abdomen', 'abdominocentesis appearance', 'abdomcentesis total protein'])

#se calcula el porcentaje de datos faltantes por columna en el dataframe
na_ratio = ((df.isnull().sum() / len(df))*100)

print("porcentaje de nulls despues de la eliminacion de las columnas ----------------------------------------------------")

#se imprime el dataframe
print(na_ratio)

print("modas ----------------------------------------------------")
## media si es valor (temperatura, cntidad de proteina), moda si es categorica (estado civil, etc)
moda = pd.Series(df[["surgery"]].values.flatten()).mode()[0]
df["surgery"] = df['surgery'].fillna(moda)
print("moda surgery: "+repr(moda))

moda1 = pd.Series(df[["temperature of extremities"]].values.flatten()).mode()[0]
df["temperature of extremities"] = df['temperature of extremities'].fillna(moda1)
print("moda temperature of extremities: "+repr(moda1))

moda2 = pd.Series(df[["peripheral pulse"]].values.flatten()).mode()[0]
df["peripheral pulse"] = df['peripheral pulse'].fillna(moda2)
print("moda peripheral pulse: "+repr(moda2))

moda3 = pd.Series(df[["mucous membranes"]].values.flatten()).mode()[0]
df["mucous membranes"] = df['mucous membranes'].fillna(moda3)
print("moda mucous membranes: "+repr(moda3))

moda4 = pd.Series(df[["capillary refill time"]].values.flatten()).mode()[0]
df["capillary refill time"] = df['capillary refill time'].fillna(moda4)
print("moda capillary refill time: "+repr(moda4))

moda5 = pd.Series(df[["pain"]].values.flatten()).mode()[0]
df["pain"] = df['pain'].fillna(moda5)
print("moda pain: "+repr(moda5))

moda6 = pd.Series(df[["peristalsis"]].values.flatten()).mode()[0]
df["peristalsis"] = df['peristalsis'].fillna(moda6)
print("moda peristalsis: "+repr(moda6))

moda7 = pd.Series(df[["abdominal distension"]].values.flatten()).mode()[0]
df["abdominal distension"] = df['abdominal distension'].fillna(moda7)
print("moda abdominal distension: "+repr(moda7))

moda8 = pd.Series(df[["outcome"]].values.flatten()).mode()[0]
df["outcome"] = df['outcome'].fillna(moda8)
print("moda outcome: "+repr(moda8))

print("medias ----------------------------------------------------")

media = pd.Series(df[["rectal temperature"]].values.flatten()).mean()
df["rectal temperature"] = df['rectal temperature'].fillna(media)
print("media rectal temperature: "+repr(media))

media1 = int(pd.Series(df[['pulse']].values.flatten()).mean())
df['pulse'] = df['pulse'].fillna(media1)
print("media pulse: "+repr(media1))

media2 = int(pd.Series(df[['respiratory rate']].values.flatten()).mean())
df['respiratory rate'] = df['respiratory rate'].fillna(media2)
print("media respiratory rate: "+repr(media2))

media3 = int(pd.Series(df[['packed cell volume']].values.flatten()).mean())
df['packed cell volume'] = df['packed cell volume'].fillna(media3)
print("media packed cell volume: "+repr(media3))

media4 = int(pd.Series(df[['total protein']].values.flatten()).mean())
df['total protein'] = df['total protein'].fillna(media4)
print("media total protein: "+repr(media4))

#se calcula el porcentaje de datos faltantes por columna en el dataframe
na_ratio = ((df.isnull().sum() / len(df))*100)

print("porcentaje de nulls despues de llenar las columnas ----------------------------------------------------")

#se imprime el dataframe
print(na_ratio)

print("graficas despues de rellenar las columnas ----------------------------------------------------")


"""
fig1 = px.box(df, x='outcome', y='pulse', color='outcome', points = 'all', title='Boxplot variable pulse',category_orders=category_orders)
fig1.show()
"""

print("se eliminan datos atipicos ----------------------------------------------------")

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

print(df)

df = df.drop(columns=['cp_data','Hospital Number'])

#dataframe con las variables cuantitativas para determinar el coeficiente de pearson
dfc = df.drop(columns=['surgery','Age','temperature of extremities','peripheral pulse','mucous membranes','capillary refill time',
    'pain','peristalsis','abdominal distension','outcome','surgical lesion'])

print(df)

matriz_correlacion = df.corr(method='pearson')

print(matriz_correlacion)

atipicos("rectal temperature","arriba",41,media)
atipicos("rectal temperature","abajo",37,media)
atipicos("pulse","arriba",120,media1)
atipicos("packed cell volume","abajo",30,media3)

"""
category_orders = {"outcome": ["1", "2", "3"]}
fig = px.box(df, x="outcome", y="rectal temperature", color='outcome', points = 'all', title='Boxplot variable rectal temperature',category_orders=category_orders)
#fig.show()

category_orders = {"outcome": ["1", "2", "3"]}
fig1 = px.box(df, x='outcome', y='packed cell volume', color='outcome', points = 'all', title='Boxplot variable packed cell volume',category_orders=category_orders)
#fig1.show()
"""

def ponerNumerosBarra (ax):
  for bar in ax.patches:
        height = bar.get_height()
        width = bar.get_width()
        x = bar.get_x()
        y = bar.get_y()

        label_text = height
        label_x = x + width / 2
        label_y = y + height / 2


        ax.text(label_x, label_y, '{:,.1f}'.format(label_text), ha='center',
                va='center')


"""
sns.countplot(data=df, x="mucous membranes", hue = 'outcome')
plt.show()


sns.countplot(data=df, x='pain', hue = 'outcome')
sns.countplot(data=df, x='capillary refill time', hue = 'outcome')
sns.countplot(data=df, x='peristalsis', hue = 'outcome')
sns.countplot(data=df, x='abdominal distension', hue = 'outcome')
sns.countplot(data=df, x='Age', hue = 'outcome')



sns.stripplot(x="Age", y="pulse", data=df)
ax=df.groupby(['surgery','outcome']).size().groupby(level=0).apply(lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True)
ponerNumerosBarra(ax)

sns.boxplot(data=df, x="Age", y="rectal temperature")
plt.title("Distribución de edades y temperaturas rectales")
plt.xlabel("Edad")
plt.ylabel("Temperatura rectal (°C)")
plt.show()


sns.countplot(data=df, x="pain")
plt.title("Distribución de niveles de dolor")
plt.xlabel("Nivel de dolor")
plt.ylabel("Cantidad de casos")
plt.show()

outcome_surgery = df.groupby(['outcome', 'surgical lesion']).size().unstack()
outcome_surgery.plot(kind='bar', stacked=True)
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Stacked Bar Plot: Outcome vs. Surgical Lesion')
plt.show()

val = int(df["pulse"].size)

i_pulse = 0
while i_pulse < val:
    if df.loc[i_pulse, "pulse"] > 120:
        df.loc[i_pulse, "pulse"] = media1
    i_pulse += 1



val1 = int(df["packed cell volume"].size)
i_pcv = 0
while i_pcv < val1:
    if df.loc[i_pcv, "cc"] > 120:
        df.loc[i_pcv, "packed cell volume"] = media3
    i_pcv += 1

sns.stripplot(x="Age", y="pulse", data=df,jitter=True,hue='outcome',palette='Set1',dodge=True)
plt.show()


val = int(df["rectal temperature"].size)

i = 0
while i < val:
    if df.loc[i, "rectal temperature"] < 36 or df.loc[i, "rectal temperature"] > 39.5:
        df.loc[i, "rectal temperature"] = media
    i += 1


# Calcular los outliers
outliers = st.outliers(df["rectal temperature"])

# Reemplazar los outliers con la media
for outlier in outliers:
  df["rectal temperature"] = df["rectal temperature"].replace(outlier, media)


  for outlier in outliers:
    Tabla.loc[outlier, 'Temperatura rectal'] = np.mean(Tabla[Temperatura rectal])

category_orders = {"outcome": ["1", "2", "3"]}
figTest = px.box(df, x="outcome", y="rectal temperature", color='outcome', points = 'all', title='Boxplot variable rectal temperature',category_orders=category_orders)
figTest.show()
"""