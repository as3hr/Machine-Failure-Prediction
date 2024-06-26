import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

data = pd.read_csv('machine_failure_prediction.csv')
data.head()
data = data.drop(["UDI", 'Product ID'], axis=1)
data['nf'] = data['Tool wear [min]'] * data['Torque [Nm]']

label_encoder = LabelEncoder()
label_encoder.fit(data['Type'])
data['Type'] = label_encoder.transform(data['Type'])

X = data.drop(['Failure Type', 'Target'], axis=1)
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = []
imported_as = []
knn = KNeighborsClassifier(n_neighbors=1)
classifier.append('k Nearest Neighbours')
imported_as.append('knn')
X_train.info()

class Modelling:
    def __init__(self, X_train, Y_train, X_test, Y_test, models):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.models = models

    def fit(self):
        model_acc = []
        model_time = []
        for i in self.models:
            start = time.time()
            if i == 'knn':
                accuracy = []
                for j in range(1, 200):
                    kn = KNeighborsClassifier(n_neighbors=j)
                    kn.fit(self.X_train, self.Y_train)
                    predK = kn.predict(self.X_test)
                    accuracy.append([accuracy_score(self.Y_test, predK), j])
                temp = accuracy[0]
                for m in accuracy:
                    if temp[0] < m[0]:
                        temp = m
                i = KNeighborsClassifier(n_neighbors=temp[1])
            i.fit(self.X_train, self.Y_train)
            model_acc.append(accuracy_score(self.Y_test, i.predict(self.X_test)))
            stop = time.time()
            model_time.append((stop - start))
            print(i, 'has been fit')
        self.models_output = pd.DataFrame({'Models': self.models, 'Accuracy': model_acc, 'Runtime (s)': model_time})

    def results(self):
        models = self.models_output
        self.best = models['Models'][0]
        self.models_output_cleaned = models
        return models

    def best_model_accuracy(self):
        return self.models_output_cleaned['Accuracy'][0]

    def best_model_runtime(self):
        return round(self.models_output_cleaned['Runtime (s)'][0], 3)

classification = Modelling(X_train, y_train, X_test, y_test, [knn])
classification.fit()
classification.results()

print('Accuracy of model:', classification.best_model_accuracy())
print('Training Runtime in seconds', classification.best_model_runtime())

def get_test_size():
    while True:
        try:
            test_size = float(input("Enter the percentage of data to use for testing (e.g., 10 for 10%): ")) / 100
            if 0 < test_size < 1 < 100:
                return test_size
            else:
                print("Please enter a valid percentage between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")

def predict_failure(classifier, label_encoder):
    print("\nAll Models have trained.\nPress 1 to predict new machine failure type or\nPress 2 to test the certain percentage of data.\nEnter any other key to exit the menu.")
    choice = input("Enter your choice: ").strip()
    
    if choice == "1":
        print("\nEnter the details of the new machine to predict its failure type:")
        air_temp = float(input("Air temperature [K]: "))
        process_temp = float(input("Process temperature [K]: "))
        rotational_speed = float(input("Rotational speed [rpm]: "))
        torque = float(input("Torque [Nm]: "))
        tool_wear = float(input("Tool wear [min]: "))
        machine_type = input("Machine Type (L, M, H): ")

        new_data = pd.DataFrame({
            'Type': [machine_type],
            'Air temperature [K]': [air_temp],
            'Process temperature [K]': [process_temp],
            'Rotational speed [rpm]': [rotational_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear],
        })

        new_data['Type'] = label_encoder.transform(new_data['Type'])

        new_data['nf'] = new_data['Tool wear [min]'] * new_data['Torque [Nm]']

        prediction = classifier.predict(new_data)

        if prediction[0] == 1:
            print("\nPredicted Failure Type: Failure")
        else:
            print("\nPredicted Failure Type: Non-failure")
    elif choice == "2":
        test_size = get_test_size()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        classification = Modelling(X_train, y_train, X_test, y_test, [knn])
        classification.fit()
        classification.results()
        print('Accuracy of model:', classification.best_model_accuracy())
        print('Training Runtime in seconds', classification.best_model_runtime())
    else: 
        print("Exiting without testing a new machine.")

predict_failure(knn, label_encoder)