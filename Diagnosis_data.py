

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

data = data.head()
 
df = pd.DataFrame(data, columns=["Temperature", "Inflammation", "Nephritis"])
 
# plot the dataframe
df.plot(x="Temperature", y=["Inflammation", "Nephritis"], kind="bar", figsize=(9, 8))
 
# print bar graph
plt.show()





