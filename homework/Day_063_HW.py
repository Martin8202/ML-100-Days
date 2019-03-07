
# coding: utf-8

# # Neural_Network Homework

# In[ ]:


# 載入套件
import numpy as np

#Input array
X = np.array([[1, 0, 1, 0], [1 ,0 ,1 ,1 ],[ 0 , 1 , 0 , 1 ]])

#Output
y = np.array([[1], [1], [0]])

# In[ ]:


#Sigmoid Function
def sigmoid(x): 
    return  1/(1+np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x): 
    return x*(1-x)


# In[ ]:


#Variable initialization
epoch = 5000 #Setting training iterations
lr = 0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set 
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer


#weight and bias initialization
wh = np.random.uniform( size = ( inputlayer_neurons , hiddenlayer_neurons ) ) 
bh = np.random.uniform( size = ( 1 , hiddenlayer_neurons ) ) 
wout = np.random.uniform( size = ( hiddenlayer_neurons , output_neurons ) ) 
bout = np.random.uniform( size = ( 1 , output_neurons ) )

# # 作業
# * 請參閱範例中的 hidden Layer 寫法, 完成 output Layer 的程式

# In[ ]:

for i in  range ( epoch ) :
    #Forward Propogation
    hidden_layer_input1 = np.dot(X, wh) 
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations =  sigmoid( hidden_layer_input ) 
    
    #寫出output Layer

    output_layer_input1 = np.dot(hiddenlayer_activations, wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)
    
print ( "output of Forward Propogation:\n{}" .format(output)) 
print ( "wout,bout of Backpropagation:\n{},\n{}" .format(wout, bout ))