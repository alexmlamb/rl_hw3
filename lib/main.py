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

'''

import random

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


sales = 10.0

for iteration in range(0,50):
    inv = basic_policy(sales)
    sales = basic_environment(inv)

    print inv,sales





