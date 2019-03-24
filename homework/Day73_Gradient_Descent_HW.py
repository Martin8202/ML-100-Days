#!/usr/bin/env python
# coding: utf-8

# # Find the local minima of the function y=(x+5)² starting from the point x=3

# 
# Step 1 : Initialize x =3. Then, find the gradient of the function, dy/dx = 2*(x+5).
# 
# Step 2 : Move in the direction of the negative of the gradient. We use a learning rate. Let us assume the learning rate → 0.01
# 
# Step 3 : Let’s perform 2 iterations of gradient descent
# 
# Step 4 : We can observe that the X value is slowly decreasing and should converge to -5 (the local minima). However, how many iterations should we perform?

# # 作業:
#     請嘗試使用不同的組合驗證 learning rate 對所需 iteration 的影響
#     lr = [0.1, 0.0001]
#     主要驗證 Lr 對於grandient 收斂的速度
#     

# In[ ]:


cur_x = 3 # The algorithm starts at x=3
precision = 0.000001 #This tells us when to stop the algorithm
max_iters = 10000 # maximum number of iterations
df = lambda x: 2*(x+5) #Gradient of our function 

previous_step_size = 1
iters = 0
lrs = [0.1, 0.0001]
iters_history = [iters]
x_history = [cur_x]

iters_hist_all=[]
x_hist_all=[]

# In[ ]:


'''
MUST RESET
# Learning rate
#iteration counter
#Step size  
'''

for lr in lrs:
    while previous_step_size > precision and iters < max_iters:
        prev_x = cur_x #Store current x value in prev_x
        cur_x = cur_x - lr * df(prev_x) #Gradient descent
        previous_step_size = abs(cur_x - prev_x) # 取較大的值, Change in x
        iters = iters+1 #iteration count
        print("the learning rate is",lr, "\nIteration",iters,"\nX value is",cur_x) #Print iterations
         # Store parameters for plotting
        iters_history.append(iters)
        x_history.append(cur_x)
    print("\nTotally iteations: ", iters)
    print("The local minimum occurs at", cur_x,'\n')
      
    iters_hist_all.append(iters_history)
    x_hist_all.append(x_history)
    
    iters = 0
    previous_step_size = 1
    cur_x = 3
    iters_history = [iters]
    x_history = [cur_x]
    
    
import matplotlib.pyplot as plt
%matplotlib inline 

plt.figure(figsize=(14,5))
for i in range(2):
    plt.subplot(1,2,i+1)
    plt.plot(iters_hist_all[i], x_hist_all[i], 'o-', ms=3, lw=1, color='blue')
    plt.xlabel(r'$iters$', fontsize=14)
    plt.ylabel(r'$x$', fontsize=14)
    plt.title("learning rate= " + str(lrs[i]))
plt.show()





