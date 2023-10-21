import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv('diagnosis.data',sep='\t',encoding='utf-16',header=None,names=["Temperature","Nausea","Lumbar Pain",
                                                                                 "Urine Pushing","Micturition pains","Burning","Inflammation","Nephritis"])

def temperature(temperature):
    if ',' in temperature:
        temperature=temperature.replace(',','.')
    return temperature

data["Temperature"]=data.apply(lambda row:temperature(row["Temperature"]),axis=1)
data['Temperature']=data['Temperature'].astype('float')
print(data)

def boolean_conv(info):
    if info=='no':
        info=0
    else:
        info=1
    return info

data["Nausea"]=data.apply(lambda row:boolean_conv(row["Nausea"]),axis=1)
data["Lumbar Pain"]=data.apply(lambda row:boolean_conv(row["Lumbar Pain"]),axis=1)
data["Urine Pushing"]=data.apply(lambda row:boolean_conv(row["Urine Pushing"]),axis=1)
data["Micturition pains"]=data.apply(lambda row:boolean_conv(row["Micturition pains"]),axis=1)
data["Burning"]=data.apply(lambda row:boolean_conv(row["Burning"]),axis=1)
data["Inflammation"]=data.apply(lambda row:boolean_conv(row["Inflammation"]),axis=1)
data["Nephritis"]=data.apply(lambda row:boolean_conv(row["Nephritis"]),axis=1)

print(data)

X = data[['Temperature', 'Nausea', 'Lumbar Pain', 'Urine Pushing', 'Micturition pains', 'Burning','Nephritis']]
Y = data[['Inflammation']]
corr_matrix = X.corr()
print(corr_matrix)
corr_with_y = X.corrwith(Y)
corr_with_y = corr_with_y.dropna()
print(corr_with_y)

ca=sns.heatmap(corr_matrix, annot=True)
plt.xticks(rotation=45)
plt.show()
print(ca)


X = data[['Temperature', 'Nausea', 'Lumbar Pain', 'Urine Pushing', 'Micturition pains', 'Burning','Inflammation']]
Y = data[['Nephritis']]
corr_matrix = X.corr()
print(corr_matrix)
corr_with_y = X.corrwith(Y)
corr_with_y = corr_with_y.dropna()
print(corr_with_y)
ca=sns.heatmap(corr_matrix, annot=True)
plt.xticks(rotation=45)
plt.show()
print(ca)


X = data[['Temperature', 'Nausea', 'Lumbar Pain', 'Urine Pushing', 'Micturition pains', 'Burning']]
Y = data[['Inflammation','Nephritis']]
corr_matrix = X.corr()
print(corr_matrix)
corr_with_y = X.corrwith(Y)
corr_with_y = corr_with_y.dropna()
print(corr_with_y)
ca=sns.heatmap(corr_matrix, annot=True)
plt.xticks(rotation=45)
plt.show()
print(ca)


sns.pairplot(data, x_vars=['Temperature', 'Nausea', 'Lumbar Pain', 'Urine Pushing', 'Micturition pains', 'Burning'], y_vars=['Inflammation'])
plt.show()

sns.pairplot(data, x_vars=['Temperature', 'Nausea', 'Lumbar Pain', 'Urine Pushing', 'Micturition pains', 'Burning'], y_vars=['Nephritis'])
plt.show()

# create a linear regression model and fit the data
model = LinearRegression()
model.fit(X, Y)

# print the coefficients of the input variables
print(model.coef_)


model = RandomForestRegressor()
model.fit(X, Y)

print(model.feature_importances_)




