'''
Sales[0] = 10

for t in 0 to N:
    Inventory[t+1] = Sales[t],50% or Sales[t]/2,50%
    Sales[t+1] = min(Inventory[t+1],10)

policy maps sales to inventory on next step.  
environment maps current inventory to current sales.  

=====

Actual learning: 

    Q-network takes (sales[t],inventory[t+1]) as inputs and estimates Q-value.  

    The target value for the Q-network is the 1-step TD update.  

    ||Q(s,a) - (reward + average(Q(s', a')))||

    reward = Sales[t+1]*2 - Inventory[t+1]

    -Averaged over actions inventory[t+2], compute the Q-network for sales[t+1],inventory[t+2]

    -Intuition is that for the action that you take on this step, you get to a certain next state, and you can score that state using your value estimator for the next state but considering all possible actions.  

    -Could you learn a direct estimator of Q(s) which is a NN mapping from s to Q(s)?  Probably?  Point?  Might be nice in cont. spaces

Take action on each step by selecting high value from Q-network.  

---------

'''

import random
import theano
import theano.tensor as T
from layers import param_init_fflayer, fflayer
import numpy as np

'''
sales[t] to Inventory[t+1]
'''
def basic_policy(last_sales):
    if random.uniform(0,1) < 0.9:
        return last_sales
    else:
        return int(last_sales / 2.0)

'''
Sales[t+1] from Inventory[t].  
'''
def basic_environment(inv):
    return min(inv, 10)

def init_params(p):

    p = param_init_fflayer({},p,prefix='qn_1',nin=2,nout=512)
    p = param_init_fflayer({},p,prefix='qn_2',nin=512,nout=512)
    p = param_init_fflayer({},p,prefix='qn_3',nin=512,nout=1)

    p = param_init_fflayer({},p,prefix='mqn_1',nin=1,nout=512)
    p = param_init_fflayer({},p,prefix='mqn_2',nin=512,nout=512)
    p = param_init_fflayer({},p,prefix='mqn_3',nin=512,nout=1)

    return p

'''
    State and action to q-value.  
    In this context these are sales,inventory
    sales is an (Nx1) matrix, inventory is an (Nx1) matrix.  
'''
def qnetwork(p,sales,inventory):
    inp = T.concatenate([sales,inventory],axis=1)
    h1 = fflayer(p,inp,options={},prefix='qn_1',activ='lambda x: T.nnet.relu(x)')
    h2 = fflayer(p,h1,options={},prefix='qn_2',activ='lambda x: T.nnet.relu(x)')
    qv = fflayer(p,h2,options={},prefix='qn_3',activ='lambda x: x')

    return qv

'''
    State to expected Q-value marginalized over actions.  
    In this context this is sales, which is an (Nx1) matrix.  
'''
def marginalqnetwork(p,sales):
    inp = sales
    h1 = fflayer(p,inp,options={},prefix='mqn_1',activ='lambda x: T.nnet.relu(x)')
    h2 = fflayer(p,h1,options={},prefix='mqn_2',activ='lambda x: T.nnet.relu(x)')
    qv = fflayer(p,h2,options={},prefix='mqn_3',activ='lambda x: x')

    return qv
    
p = init_params({})

tsales_curr = T.matrix()
tinv_curr = T.matrix()
tsales_next = T.matrix()

q = qnetwork(p, tsales, tinv)
mq = marginalqnetwork(p, tsales)

reward = 2.0*tsales - tinv

q_loss = T.mean(T.abs_(q - (mq + reward)))

mq_loss = T.mean(abs_(

'''
Figure out updates for q and mq.  Both are doing 1-step TD updates.  

'''

run_qnetwork = theano.function([tsales, tinv], outputs = [q])
run_mqnetwork = theano.function([tsales], outputs = [mq])

if __name__ == "__main__":

    sales = 10.0

    for iteration in range(0,20):
        inv = basic_policy(sales)
        sales = basic_environment(inv)

        inv_mat = np.asarray([[inv]]).astype('float32')
        sales_mat = np.asarray([[sales]]).astype('float32')

        print "qnetwork", run_qnetwork(sales_mat,inv_mat)
        print "marginal qnetwork", run_mqnetwork(sales_mat)









