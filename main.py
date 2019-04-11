#import lib and chinese language
import  numpy as np
import  pandas as pd
import  tensorflow as tf
import  matplotlib.pyplot as plt
import  matplotlib.font_manager as fm
#import Class
from  autoencoder import  Autoencoder


font_path = "/root/anaconda3/envs/tf/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/SIMKAI.TTF"
myfont = fm.FontProperties(fname=font_path)
#check for load csv file
creditcard_path = "/root/PycharmProjects/datasoure/creditcard.csv"

try:
    data = pd.read_csv(creditcard_path)
except Exception as e:
    print("CSV file can not open,please check!")
#check data-batch shape
#print(data.shape)    #correct 284807,31
#check any null value
data.isnull().values.any()
##show label-data distribution,by your needs
#count_calss  = pd.value_counts(data['Class'],sort= True).sort_index()
#count_calss.plot(kind= 'bar')
#plt.title("Cheat histogram")
#plt.xlabel("Category")
#plt.ylabel("Frequentness")
#plt.show()

#data preproce
from sklearn.preprocessing  import  StandardScaler

data['normAount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Time','Amount'],axis=1)
data.head()
#test for data-table,just output 20 row and all col
#df = pd.DataFrame(data.values[0:20,0:30],columns=list(range(1,31)))
#print(df)

#divied databatch ,for train an test
from sklearn.model_selection import train_test_split
good_data = data[data['Class'] == 0]
bad_data = data[data['Class'] == 1]
#train:80% and test:20%
X_train, X_test = train_test_split(data,test_size=0.2,random_state=42)

X_train = X_train[X_train['Class'] == 0]
X_train = X_train.drop(['Class'],axis = 1)

y_test = X_test['Class']
X_test = X_test.drop(['Class'],axis=1)

X_train = X_train.values   #this is train data
X_test =    X_test.values  #this ia test data

#define label
X_good = good_data.loc[:,good_data.columns != 'Class']
y_good = good_data.loc[:,good_data.columns == 'Class']

X_bad = bad_data.loc[:,bad_data.columns != 'Class']
y_bad = bad_data.loc[:,bad_data.columns == 'Class']

# train model

model = Autoencoder(n_hidden_1=15,n_hidden_2=3,n_input=X_train.shape[1],learning_rate = 0.01)
#define batch_size,epochs ...
train_epochs = 100  #all training times
batch_size = 256    #one training data sizes
display_step = 20   #display step
record_step = 10    #record step
#start training
total_batch = int(X_train.shape[0] / batch_size)    #one training use samples at every epochs
cost_summary = [] #temporary list

for epoch in range(train_epochs):              # epochs iterator
    cost = None
    for i in range(total_batch):                #one epoch iterate
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch = X_train[batch_start:batch_end,:]
        cost = model.partial_fit(batch)
    if epoch % display_step == 0 or epoch % record_step == 0:
        total_cost = model.calc_total_cost(X_train)
        if epoch % record_step == 0:
            cost_summary.append({'epoch':epoch+1,'cost':total_cost})
        if epoch % display_step == 0:
            print("Epoch:{},cost={:.9f}".format(epoch+1,total_cost))


##get result: weights and biases,this is finally destination
result_weights = model.weights
result_biases = model.biases
sess1 = tf.Session()  #tf sesion
init1 = tf.global_variables_initializer() #init op
sess1.run(init1)  #run init op
##weights:
encoder_h1 = sess1.run(result_weights['encoder_h1'])
encoder_h2 = sess1.run(result_weights['encoder_h2'])
decoder_h1 = sess1.run(result_weights['decoder_h1'])
decoder_h2 = sess1.run(result_weights['decoder_h2'])
##biases:
encoder_b1 = sess1.run(result_biases['encoder_b1'])
encoder_b2 = sess1.run(result_biases['encoder_b2'])
decoder_b1 = sess1.run(result_biases['decoder_b1'])
decoder_b2 = sess1.run(result_biases['decoder_b2'])
#show result matrix shape:
print("weihths matrix shape are :{},{},{},{}".format(encoder_h1.shape,encoder_h2.shape,decoder_h1.shape,decoder_h2.shape))
print("biases matrix shape are :{},{},{},{}".format(encoder_b1.shape,encoder_b2.shape,decoder_b1.shape,decoder_b2.shape))
## show relationship ,by  loss and iterate-step;according to your needs ,use this  code
f, ax1 = plt.subplots(1,1,figsize=(10,4))
ax1.plot(list(map(lambda x: x['epoch'],cost_summary)),list(map(lambda x:x['cost'],cost_summary)))
ax1.set_title("loss_values")
plt.xlabel('iterations')
plt.show()

##test model  ,according to your needs ,use this test code
encode_decode = None
total_batch = int(X_test.shape[0] / batch_size) + 1
for i in range(total_batch):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    batch = X_test[batch_start:batch_end,:]
    batch_res = model.reconstruct(batch)
    if encode_decode is None:
        encode_decode = batch_res
    else:
        encode_decode = np.vstack((encode_decode,batch_res))


def get_df(orig,ed,_y):
    rmse = np.mean(np.power(orig - ed,2),axis=1)
    return  pd.DataFrame({'rmse':rmse,'target':_y})
df = get_df(X_test,encode_decode,y_test)

##show test imformation
show_df = df.describe()
print(show_df)