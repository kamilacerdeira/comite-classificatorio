import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('census_data.csv')
data = data.dropna(axis=0)# Remove as linhas com valores ausentes

# Definição das colunas de entrada e do target
X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

# Divide a base em 80% dos dados para treinamento e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pre-processamento dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instancia do classificador KNN
k = 5 # número de vizinhos a considerar
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)

predictions = knn.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print('Acurácia:', accuracy)

