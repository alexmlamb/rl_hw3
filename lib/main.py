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
import lasagne
from utils import init_tparams
import matplotlib.pyplot as plt

from theano.tensor.opt import register_canonicalize

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()
register_canonicalize(theano.gof.OpRemove(consider_constant), name='remove_consider_constant_2')


'''
sales[t] to Inventory[t+1]
'''
def basic_policy(last_sales):
    if random.uniform(0,1) < 0.9:
        return last_sales
    else:
        return int(last_sales / 2.0)

def random_policy(last_sales):
    return random.randint(0,50)

'''
Probability of exploring decays in the number of iterations.  
'''
def epsilon_greedy(inp, iteration):
    p_explore = (0.9999)**iteration

    if random.uniform(0,1) < p_explore:
        return random.randint(0,50)
    else:
        return inp

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
    inp = T.concatenate([sales,inventory],axis=1) / 50.0
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
    
p = init_tparams(init_params({}))

tsales_curr = T.matrix()
tinv_curr = T.matrix()
tsales_next = T.matrix()
tinv_next = T.matrix()

q = qnetwork(p, tsales_curr, tinv_curr)
mq = marginalqnetwork(p, tsales_next)
q_next = qnetwork(p, tsales_next, tinv_next)

reward = 2.0*tsales_curr - tinv_curr

q_loss = T.mean(T.abs_(q - (0.5*consider_constant(mq) + reward)))

mq_loss = T.mean(T.abs_(mq - consider_constant(q_next)))

loss = q_loss + mq_loss

'''
Figure out updates for q and mq.  Both are doing 1-step TD updates.  

'''

updates = lasagne.updates.adam(q_loss + mq_loss, p.values())

train_method = theano.function([tsales_curr,tinv_curr,tsales_next,tinv_next], outputs=[loss,reward], updates=updates)
run_q = theano.function([tsales_curr,tinv_curr], outputs = q)

#mode = "offpolicy"
mode = "onpolicy"

def tomat(inp):
    return np.asarray([[inp]]).astype('float32')

losses = []
rewards = []
loss_ma = None
reward_ma = None

if __name__ == "__main__":

    sales = 10.0

    for iteration in range(0,50000):
        if mode == "offpolicy":
            inv = basic_policy(sales)



        else:
            best_inv = -1
            best_inv_score = -9999.9
            for inv_try in range(0,20):
                val = run_q(tomat(sales),tomat(inv_try))
                if val > best_inv_score:
                    best_inv = inv_try
                    best_inv_score = val
            inv = epsilon_greedy(best_inv,iteration)
            
            print "greedy best", best_inv
            print "buy at", inv

        if iteration % 500 == 0:
            for inv_try in range(0,50):
                val = run_q(tomat(sales),tomat(inv_try))
                print inv_try, val


        sales = basic_environment(inv)

        inv_mat = np.asarray([[inv]]).astype('float32')
        sales_mat = np.asarray([[sales]]).astype('float32')

        if iteration > 1:
            r = train_method(sales_last, inv_last, sales_mat, inv_mat)

            if loss_ma == None:
                loss_ma = r[0]
                rewards_ma = r[1][0][0]
            else:
                loss_ma = 0.99*loss_ma + 0.01*r[0]
                rewards_ma = 0.99*rewards_ma + 0.01*r[1][0][0]

            losses.append(loss_ma)
            rewards.append(rewards_ma)

        sales_last = sales_mat
        inv_last = inv_mat

        if iteration % 5000 == 0:

            plt.plot(losses)
            plt.title("Q-Network TD-Error")
            plt.show()

            plt.plot(rewards)
            plt.title("Expected Returns with Exploration")
            print inv_try, val
            plt.show()






