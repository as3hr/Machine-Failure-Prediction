import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from modelling import Modelling
from utils import predict_failure

data = pd.read_csv('machine_failure_prediction.csv')
data = data.drop(["UDI", 'Product ID'], axis=1)
data['nf'] = data['Tool wear [min]'] * data['Torque [Nm]']

# plt.figure(figsize=(10, 5))
# sns.countplot(data=data[data['Target'] == 1], x="Failure Type")
# plt.show()
# plt.figure(figsize=(10, 8))
# sns.countplot(data=data, x="Target")
# plt.show()

label_encoder = LabelEncoder()
label_encoder.fit(data['Type'])
data['Type'] = label_encoder.transform(data['Type'])

X = data.drop(['Failure Type', 'Target'], axis=1)
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=1)

classification = Modelling(X_train, y_train, X_test, y_test, [knn])
classification.fit()
classification.results()

print('Accuracy of model:', classification.best_model_accuracy())
print('Training Runtime in seconds', classification.best_model_runtime())

predict_failure(knn, label_encoder, X, y)