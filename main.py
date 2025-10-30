import pandas as pd 
from matplotlib import pyplot as plt 
from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA 
from sklearn.preprocessing import MinMaxScaler 

data = pd.read_csv('creditcard.csv')
#print(data.head(10)) 

df = data.drop(['V28','V27','V26' ,'V25' ,'V24' ,'V23' ,'V22' ,'V21' ,'V20' ,'V19' ,'V18' ,'V17' ,'V16' ,'V15' ,'V14' ,'V13' ,'V12' ,'V11' ,'V10' ,'V9' ,'V8'] , axis = 1)

print(df.head(10))

#Data Preprocessing and Feature scaling 
#Applying Normalization om the dataset, since the datasets lie within the range of -1 to 0 

scaler = MinMaxScaler() 
scaled_data = scaler.fit_transform(df)  
#print(scaled_data)

X = df[['V1' , 'V2' , 'V3' , 'V4' , 'V5' , 'V6' , 'V7']] 
y = df['Amount'] 

#then we do dimensionality reduction upon the preprocessed dataset 
#i.e. Principal Component Analysis 

pca = PCA(0.25) 

X_pca = pca.fit_transform(X) 

print(X_pca.shape) 

print(pca.explained_variance_ratio_) 

#Exploratory Data Analysis 
plt.figure(figsize = (8,6)) 
plt.scatter(X_pca[: , 0] , X_pca[: , 1] , c = y, cmap = 'rainbow' , edgecolor = 'k') 
plt.xlabel('PC1')
plt.ylabel('PC2') 
plt.title('PCA Projection of creditcard dataset') 
plt.show() 

