import joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
dataset = pd.read_csv(r'C:\Users\acer\OneDrive\Desktop\ML-PROJECTS\SONAR- rock vs mine prediction\sonar data.csv', header=None)
dataset.head()
dataset.shape
dataset.describe()
dataset[60].value_counts()
dataset.groupby(60).mean()
x = dataset.drop(columns=60, axis=1)
y = dataset[60]
print(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)
print(x.shape,x_train.shape,x_test.shape)
model = LogisticRegression()
model.fit(x_train, y_train)
x_train_predictions  = model.predict(x_train)
training_dataset_accuracy = accuracy_score(x_train_predictions,  y_train)
print("training dataset accuracy is : ",training_dataset_accuracy)
x_test_predictions = model.predict(x_test)
testing_dataset_accuracy = accuracy_score(x_test_predictions, y_test)
print("testing dataset accuracy is : ",testing_dataset_accuracy)
input=(0.0126,0.0519,0.0621,0.0518,0.1072,0.2587,0.2304,0.2067,0.3416,0.4284,0.3015,0.1207,0.3299,0.5707,0.6962,0.9751,1.0000,0.9293,0.6210,0.4586,0.5001,0.5032,0.7082,0.8420,0.8109,0.7690,0.8105,0.6203,0.2356,0.2595,0.6299,0.6762,0.2903,0.4393,0.8529,0.7180,0.4801,0.5856,0.4993,0.2866,0.0601,0.1167,0.2737,0.2812,0.2078,0.0660,0.0491,0.0345,0.0172,0.0287,0.0027,0.0208,0.0048,0.0199,0.0126,0.0022,0.0037,0.0034,0.0114,0.0077)
input_as_numpy_array = np.asarray(input)
input_reshape = input_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_reshape)
print(prediction)
if(prediction[0]=='R'):
    print("Object is Rock")
else:
    print("Object is Mine")


joblib.dump(model, 'rock_mine_model.sav')
