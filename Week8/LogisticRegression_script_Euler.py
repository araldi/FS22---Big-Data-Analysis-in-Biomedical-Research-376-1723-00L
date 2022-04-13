"""# Prepare a script to run the logistic regression model on Euler

1.   Comment well what you are doing to be able to easily retrieve the errors
2.   Change to the exact directories on the cluster for loading/saving data
3.   Load all the libraries upfront
4.   Test the script first (resetting the notebook) to avoid undefined variables


"""

# load all the libraries you need upfront: If there are mistakes in loading, 
# you will not discover in the middle of the run

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score,precision_score
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import json

# load the data
df= pd.read_csv('https://github.com/rashida048/Datasets/raw/master/mushrooms.csv')

# prepare the data for ML
to_binary = [] 

for col in df.columns:

  # remove "?"
  if "?" in df[col].value_counts().index:
    print('convert ? values in column ', col)
    conditions = [df[col] == '?']
    choices = ['x']
    df[col] = np.select(conditions, choices, default = df[col])

  # reassign binary columns to 0,1
  elif len(df[col].value_counts()) == 2:
    print('convert this column into binary:', col)
    to_binary.append(col)
    options = list(df[col].value_counts().index)
    conditions = [df[col] == options[0], df[col] == options[1]]
    choices = [0,1]
    df[col] = np.select(conditions, choices)
    
  # drop columns with identical valuse
  elif len(df[col].value_counts()) < 2:
    print('drop this column:', col)
    df = df.drop(columns = [col])

# convert categorical into its corresponding numerical
conditions = [df['ring-number'] == 'n', df['ring-number'] == 'o', df['ring-number'] == 't']
choices = [0,1,2]
df['ring-number'] = np.select(conditions, choices)

# select what is not binary or numerical
not_to_encode = to_binary + ['ring-number']
to_encode = [i for i in df.columns if i not in not_to_encode]

# create a new dataframe for ML where the categorical columns are one-hot-encoded
df_ML = df[not_to_encode]

for col in to_encode:
  enc = preprocessing.OneHotEncoder()
  enc.fit(df[[col]])
  one_hot = enc.transform(df[[col]]).toarray()
  new_cols = [col + "_"+ i for i in enc.categories_[0]]
  df_ML[new_cols] = one_hot

# quickly visualize data with TSNE
#create model
model = TSNE(learning_rate = 100)

#fit model
x = df_ML[[i for i in df_ML.columns if 'class' not in i]]
y = df_ML['class']
transformed = model.fit_transform(x.values)
xs = transformed[:,0]
ys = transformed[:,1]
df_trans = pd.DataFrame({'xs':xs, 'ys':ys})

#create plots
plt.scatter(df_trans.loc[y==0]['xs'], df_trans.loc[y ==0]['ys'], c= 'tab:green')
plt.scatter(df_trans.loc[y ==1]['xs'], df_trans.loc[y ==1]['ys'], c= 'tab:blue')
plt.legend(loc ='lower left', labels = ['p', 'e'])

#edit this!!

# plt.savefig('/cluster/home/username/exercise_dir/TSNE_mushrooms.svg')
# plt.savefig('/cluster/home/username/exercise_dir/TSNE_mushrooms.png')

#creating test/train split
#scaling not necessary
y=df_ML['class']
X=df_ML.drop(['class'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2022)
df_train, df_test = train_test_split(df_ML, test_size = 0.20, random_state=2022)

# train logreg model
model = LogisticRegression().fit(X_train,y_train)
y_pred = model.predict(X_train)

print('------ TRAINING DATASET ------')
print('Accuracy Score : ' + str(accuracy_score(y_train,y_pred)))
print('Precision Score : ' + str(precision_score(y_train,y_pred)))

#Logistic Regression Classifier Confusion matrix
print('Confusion Matrix : \n' + str(confusion_matrix(y_train,y_pred)))

# test model

descriptors = []
y_pred = model.predict(X_test)

print('------ TEST DATASET ------')
accuracy = 'Accuracy Score : ' + str(accuracy_score(y_test,y_pred))
precision ='Precision Score : ' + str(precision_score(y_test,y_pred))
print(accuracy)
print(precision)


#Logistic Regression Classifier Confusion matrix
cm = 'Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred))
print(cm)

# save the descriptors in a list
descriptors = [accuracy, precision, cm]

# write the output somewhere
# with open("/cluster/home/username/exercise_dir/descriptors_logistic_regression_predictions.txt", "w") as file:
#     file_lines = "\n".join(descriptors)
#     file.write(file_lines)

plot_confusion_matrix(model,X_test,y_test,cmap='inferno',values_format='g')
# plt.savefig('/cluster/home/username/exercise_dir/confusion_matrix_test_dataset_predictions.svg')

# save the coefficients
logistic_coef = LogisticRegression().fit(x,y).coef_.flatten()
coeff = [i for i in df_ML.columns if 'class' not in i]

dict_coef = dict(zip(coeff, logistic_coef))

# save data into file:
# json.dump( dict_coef, open( "/cluster/home/username/exercise_dir/logistic_regression_coefficients.json", 'w' ) )
