import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier as kNN

data = pd.read_csv('voice_extracted_features.csv', sep=', ')

def qualitative_to_0_1(data,column,value_to_be_1):
    mask=data[column].values==value_to_be_1
    data[column][mask]=1
    data[column][~mask]=0
    return data

data=qualitative_to_0_1(data,'label','female')
features = list(data.column)
vals = data.values
X=vals[:,:-1]
y=vals[:,-1]
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, shuffle=False)

X_paced=PCA(2).fit_transform(X_train) #skalowanie 
fig,ax=plt.subplots(1,1)
females=y_train==1
ax.scatter(X_paced[females,0],X_paced[females,1])
ax.scatter(X_paced[~females,0],X_paced[~females,1])
ax.legend()
#3

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
pca_transform = PCA()
pca_transform.fit(X_train)
variances = pca_transform.explained_variance_ratio_
cumulated_variances = variances.cumsum()
plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
plt.yticks(np.arange(0, 1.1, 0.1))
PC_num = (cumulated_variances<0.95).sum()

#4
pipe =Pipeline([['transformer', PCA(9)],['scaler', StandardScaler()],['classifier', kNN(weights='distance')]])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

