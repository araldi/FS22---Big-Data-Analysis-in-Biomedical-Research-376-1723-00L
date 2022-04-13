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
from sklearn.manifold import TSNE

# load the data
df_ML= pd.read_csv('https://github.com/araldi/FS22---Big-Data-Analysis-in-Biomedical-Research-376-1723-00L/raw/main/Week8/ML_ready_mushroom.csv')

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

plt.savefig('/cluster/home/username/exercise_dir/TSNE_mushrooms.svg')
plt.savefig('/cluster/home/username/exercise_dir/TSNE_mushrooms.png')
