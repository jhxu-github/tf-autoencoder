#import lib and chinese language
import  numpy as np
import  pandas as pd
import  tensorflow as tf
import  matplotlib.pyplot as plt
import  matplotlib.font_manager as fm
font_path = "/root/anaconda3/envs/tf/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/SIMKAI.TTF"
myfont = fm.FontProperties(fname=font_path)
#check for load csv file
creditcard_path = "/root/PycharmProjects/creditcard/creditcard.csv"

try:
    data = pd.read_csv(creditcard_path)
except Exception as e:
    data = pd.read_csv('./creditcard_path')
#check data-batch shape
#print(data.shape)    #correct 284807,31
#check any null value
data.isnull().values.any()
#show label-data distribution
count_calss  = pd.value_counts(data['Class'],sort= True).sort_index()
count_calss.plot(kind= 'bar')
plt.title("Cheat histogram")
plt.xlabel("Category")
plt.ylabel("Frequentness")
plt.show()

#data preproce
from sklearn.preprocessing  import  StandardScaler

data['normAount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time','Amount'],axis=1)
data.head()
#test for data-table,just output 20 row and all col
#df = pd.DataFrame(data.values[0:20,0:30],columns=list(range(1,31)))
#print(df)





