import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
sns.set_style("darkgrid")

data = pd.read_csv('machine_failure_prediction.csv')
data.head()
data.info()
data = data.drop(["UDI", 'Product ID'], axis=1)
plt.figure(figsize=(10, 8))
sns.countplot(data=data, x="Target")
plt.figure(figsize=(10, 5))
sns.countplot(data=data[data['Target'] == 1], x="Failure Type")

data['nf'] = data['Tool wear [min]'] * data['Torque [Nm]']