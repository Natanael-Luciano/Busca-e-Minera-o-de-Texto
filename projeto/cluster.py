import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
import numpy as np
import re
from ast import literal_eval
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from plt_cm import *
from joblib import dump, load

def cat_change(cat):
    if cat == 4:
        return 2
    if cat == 5:
        return 3
    if cat == 7:
        return 4
    return cat

def cat_original(cat):
    if cat == 2:
        return 4
    if cat == 3:
        return 5
    if cat == 4:
        return 7
    return cat

np.random.seed(42)

df = pd.read_csv('vec_colação.csv')
df_data = pd.read_hdf('vec_colação.h5', 'vetores')

df['categoria_efetiva'] = df['categoria_efetiva'].apply(cat_change)
print(df_data.head())
mask = df['categoria_efetiva'] != 0

df_data = df_data[mask]
df_cat = df.categoria_efetiva[mask]
df_cat = df_cat.to_frame(name = 'categoria_efetiva')

matriz_de_vetores = np.stack(df_data.values)

print(df_cat.head())


pca = PCA(n_components=50)
pca.fit(matriz_de_vetores)
dump(pca, 'pca.joblib')


reduced_data = pca.fit_transform(matriz_de_vetores)

# Clusterização


print(df_cat.head())



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Suponha que 'X' são suas características e 'y' são os rótulos verdadeiros
X_train, X_test, y_train, y_test = train_test_split(reduced_data, df_cat['categoria_efetiva'], test_size=0.3, random_state=42)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_train)

y_pred = kmeans.predict(X_test)

rand_index = adjusted_rand_score(y_test, y_pred)
print("Índice Rand Ajustado:", rand_index)

plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=True, target_names=['IC', 'Equivalências/Isenções','Trancar/Destrancar matricula', 'Alteração/Inclusão/Exclusão de grau'],
                      title='Matriz de Confusão K-Means para T2')
print("Matriz de Confusão:\n")

# ######################################################################################################


from sklearn.svm import SVC


model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))

# Matriz de Confusão
plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=True, target_names=['IC', 'Equivalências/Isenções','Trancar/Destrancar matricula', 'Alteração/Inclusão/Exclusão de grau'],
                      title='Matriz de Confusão SVM para T2')
print("Matriz de Confusão:\n")

dump(model, 'SVC_T1.joblib')

# ######################################################################################################


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Instanciando e treinando o modelo
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predições
y_pred = rf.predict(X_test)

# Acurácia
print("Acurácia:", accuracy_score(y_test, y_pred))

# Relatório de Classificação
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de Confusão
plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=True, target_names=['IC', 'Equivalências/Isenções','Trancar/Destrancar matricula', 'Alteração/Inclusão/Exclusão de grau'],
                      title='Matriz de Confusão RFC para T2')
print("Matriz de Confusão:\n")

# ######################################################################################################


from sklearn.ensemble import GradientBoostingClassifier

print("#"*30)

print("\n"*2)

# Dividindo os dados

# Instanciando e treinando o modelo
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbm.fit(X_train, y_train)

# Avaliando o modelo
y_pred = gbm.predict(X_test)
print("Acurácia de GBM:", accuracy_score(y_test, y_pred))

# Relatório de Classificação
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de Confusão
plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=True, target_names=['IC', 'Equivalências/Isenções','Trancar/Destrancar matricula', 'Alteração/Inclusão/Exclusão de grau'],
                      title='Matriz de Confusão GradBoosting para T2')
print("Matriz de Confusão:\n")
# ######################################################################################################

print("#"*30)

print("\n"*2)

from sklearn.neural_network import MLPClassifier


# Instanciando e treinando o modelo
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001,
                    solver='adam', verbose=10, random_state=42, learning_rate_init=0.001)
mlp.fit(X_train, y_train)

# Avaliando o modelo
y_pred = mlp.predict(X_test)
print("Acurácia de Redes Neurais:", accuracy_score(y_test, y_pred))

# Matriz de Confusão
plot_confusion_matrix(confusion_matrix(y_test, y_pred), normalize=True, target_names=['IC', 'Equivalências/Isenções','Trancar/Destrancar matricula', 'Alteração/Inclusão/Exclusão de grau'],
                      title='Matriz de Confusão MLP para T2')
print("Matriz de Confusão:\n")

