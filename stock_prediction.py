import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore") 

#--------------------------------rISHAB bAIRI----------------------b21117--------------------


dat = pd.read_excel('Apple_stock_data.xlsx', index_col=None)   
h=["Date","Close/Last","Volume","Open","High","Low"]
dat["Date"] = pd.to_datetime(dat["Date"])  

#------------------------------plot the various variables---------------------------------------------------
for i in range(1,len(h)):
    dat[h[i]] = pd.to_numeric(dat[h[i]].replace('[^0-9\.-]','',regex=True))

plt.plot(dat[h[1]],alpha=1)  
plt.title(h[1])
plt.show()
plt.plot(dat[h[2]],alpha=1)  
plt.title(h[2])
plt.show()
plt.plot(dat[h[3]],alpha=1)  
plt.title(h[3])
plt.show()
plt.plot(dat[h[4]],alpha=1)  
plt.title(h[4])
plt.show()
plt.plot(dat[h[5]],alpha=1)  
plt.title(h[5])
plt.show()
def coeff_reg(x,y):
    parameters = np.linalg.inv(x.T @ x) @ x.T @ y    
    return parameters

dat=dat[::-1]



#--------------------------------------------------------------------------------------------------
time = '2023'
x_train = dat[dat[h[0]] < time]
x_test = dat[dat[h[0]] >= time] 

#---------------------------------------------------------------------------------------------------
def lag_time(df,colum,lag):
    x=pd.DataFrame()
    x[colum]=df[colum]
    for i in range(1,lag+1):
        x[colum+"Lag"+str(i)]=x[colum].shift(i)
    return x[lag:]  



def polynomial_func(x,poly):
    colname = list(x.columns)
    for k in colname:
        for i in range(2,poly+1):
            x[k+"Poly"+str(i)] = x[k]**i
    return x


#Note I have solved for high attribute for majority of accuracy questions because I was told this when I asked this doubt
#------------------------------------Auto regression for different p and k----------------------------------------------------


x_new= polynomial_func( lag_time(x_train,"High",5) , 5)
x_new = np.array(x_new[:-1])
y = x_train["High"].shift(-6)[:-6]
y=np.array(y)

coeff = coeff_reg(x_new,y)

x = polynomial_func( lag_time(x_test,"High",5) , 5) 

x = np.array(x[:-1])
y = x_test["High"].shift(-6)[:-6]
y=np.array(y) 

y_pred = x @ coeff.T # Predicted Value

er0rr = 0
for i in range(len(y_pred)):
    er0rr+=(y_pred[i]-y[i])**2
print((er0rr**0.5)/len(y_pred)) 


#-----------------------------Plot for every degree and time lag--------------------------------------------------
n = len(dat)

POLY_LIS = [1,2,3]
LAG_LIS = [5,10,15]


for degreee in POLY_LIS:
    
    for lag in LAG_LIS:
        x = polynomial_func(lag_time(x_train,"High",lag) , degreee)  #polynomial fitting of train
        x = np.array(x[:-1])
        y_orig = x_train["High"].shift(-lag-1)[:-lag-1] #introducing lag
        y_orig=np.array(y_orig)
        coeff = coeff_reg(x,y_orig)#coefficient calculation
        x = polynomial_func( lag_time(x_test,"High",lag) , degreee)  #polynomial fitting of test
        x = np.array(x[:-1])
        y_orig = x_test["High"].shift(-lag-1)[:-lag-1]
        y_orig=np.array(y_orig)
        y_pred = x @ coeff.T

        
        plt.plot([i for i in range(len(y_pred))],y_pred,linestyle = '-.',alpha=0.7) 
        
        plt.plot([i for i in range(len(y_pred))],y_orig,alpha = 0.4)
        plt.title("High Attribute"+" degree = "+str(degreee)+" Lag = "+str(lag))  
        plt.show()
        terror = 0
        for index in range(len(y_pred)):
            terror+=(y_pred[index]-y_orig[index])**2
        print("High Column"+" degree is equal = "+str(degreee)+"Time Lag = "+str(lag)+" Error is equal to",(terror**0.5)/len(y_pred))
        
            



#-----------------------------------pLOT degree vs RMSE-------------------------------------------------------
error_list=[]
i=0
j=0
lag = 3
for degreee in range(30):
    
    x = polynomial_func(lag_time(x_train,"High",lag) , degreee)  #polynomial fitting of train
    x = np.array(x[:-1])
    y_orig = x_train["High"].shift(-lag-1)[:-lag-1] #introducing lag
    y_orig=np.array(y_orig)

    coeff = coeff_reg(x,y_orig) #coefficient calculation
    
    x = polynomial_func( lag_time(x_test,"High",lag) , degreee)  #polynomial fitting of test
    x = np.array(x[:-1])
    y_orig = x_test["High"].shift(-lag-1)[:-lag-1]
    y_orig=np.array(y_orig)
    y_pred = x @ coeff.T


    terror = 0
    for index in range(len(y_pred)):
        terror+=(y_pred[index]-y_orig[index])**2
    error_list.append((terror**0.5)/len(y_pred))
plt.plot([i for i in range(30)],error_list)
plt.xlabel("Degree")
plt.ylabel("RMSE")
plt.title("High Attribute degree vs rmse")
plt.show()

#---------------------------------pLOT LAG vs RMSE-----------------------------------------------------------

# NOTE IT MIGHT BE SLOW IN THE LAG PART DUE TO HIGH COMUTATION FOR 6 LAGS
degree=3
error_list =[]
for lag in range(6):  
    x = polynomial_func(lag_time(x_train,"High",lag) , degreee)  #polynomial fitting of train
    x = np.array(x[:-1])
    y_orig = x_train["High"].shift(-lag-1)[:-lag-1] #introducing lag
    y_orig=np.array(y_orig)
    coeff = coeff_reg(x,y_orig) #coefficient calculation
    
    x = polynomial_func( lag_time(x_test,"High",lag) , degreee)  #polynomial fitting of test
    x = np.array(x[:-1])
    y_orig = x_test["High"].shift(-lag-1)[:-lag-1]
    y_orig=np.array(y_orig)
    y_pred = x @ coeff.T


    terror = 0
    for index in range(len(y_pred)):
        terror+=(y_pred[index]-y_orig[index])**2
    error_list.append((terror**0.5)/len(y_pred))

plt.plot([i for i in range(6)],error_list)
plt.xlabel("Time lag")
plt.ylabel("RMSE")
plt.title("High Attribute lag vs degree")
plt.show()
    
#---------------------------Using the three dimensions to plot variability--------------------------------           

LAG_LIS = [0,1,2,3,4,5,6,7,8,9]
POLY_LIS = [0,1,2,3,4,5,6,7,8,9]

ERROR_LIS = []
for degree in POLY_LIS:        
    for lag in LAG_LIS:
        x = polynomial_func(lag_time(dat,"High",lag) , degree)
        x = np.array(x[:-1])
        y = dat["High"].shift(-lag-1)[:-lag-1]
        y_org=np.array(y)

        coeff = coeff_reg(x,y)
        
        x = polynomial_func( lag_time(x_test,"High",lag) , degree)
        x = np.array(x[:-1])
        y = x_test["High"].shift(-lag-1)[:-lag-1]
        y_org=np.array(y_org)

        y_pred = x @ coeff.T

        Terror = 0
        for temp in range(len(y_pred)):
            Terror+=(y_pred[temp]-y_org[temp])**2
        Terror=Terror/len(y_pred)
        ERROR_LIS.append(Terror.round(3))

height_values = ERROR_LIS  
ERROR_LIS = np.array(ERROR_LIS)
z=np.resize(ERROR_LIS,(10,10))
lag_lis = np.array(LAG_LIS)
poly_lis = np.array(POLY_LIS)
x,y = np.meshgrid(lag_lis,poly_lis)#Grid for lanes of 3 axis
ax = plt.axes(projection="3d") 
ax.plot_surface(x,y,z) #plotting 3d surface
ax.set_xlabel("Time Lag")
ax.set_ylabel("Degree of polynomial")
ax.set_zlabel("RMSE")


