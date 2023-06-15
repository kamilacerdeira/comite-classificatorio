import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import pydotplus
from PIL import Image

data = pd.read_csv('census_data.csv')
data = data.dropna(axis=0) # Remove as linhas com valores ausentes

# Definição das colunas de entrada e do target
X = data.drop('TenYearCHD', axis=1)
y = data['TenYearCHD']

# Divide a base em 80% dos dados para treinamento e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print('Acurácia:', accuracy)

# Geração da imagem da árvore de decisão treinada
plt.figure(figsize=(10, 10))

dot_data = export_graphviz(model, out_file=None, feature_names=X.columns.astype(str), class_names=model.classes_.astype(str), filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('decision_tree.png')

# Exibir a imagem da árvore de decisão
img = Image.open("decision_tree.png")
img.show()
