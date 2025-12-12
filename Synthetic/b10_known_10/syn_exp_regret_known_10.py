#!/usr/bin/env python
# coding: utf-8

# In[233]:


import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid oversubscribed solver threads in workers
import pandas as pd
import numpy as np
import numpy.linalg as npla
# import plotnine as gg
import scipy.linalg as spla
import cvxpy as cp
import multiprocessing as mp
from functools import partial

# In[234]:


# Simulation parameter

Simu =50 #number of simulations
T = 300 #number of timesteps
K = 10 #number of context categories
num_action = 5 
B = 10 #budget

q = 1 + num_action-1 + 1 + num_action-1   # num of dummy variables + num of context + num of interactions
#p = [0.1] * K
delta = 1/T #prob in Proposition 3

theta_std = 1
lamb = 1

temp = pd.Series(list(range(1, num_action+1)))
action = pd.get_dummies(temp)
action = action.iloc[:,1:]


# In[235]:


class Agent:
    def __init__(self, theta_std, lamb, dim, action, p, B, T, K, delta, theta_true, num_action, mu_max):
        self.theta_std = theta_std
        self.lamb = lamb
        self.dim = dim
        self.action = action
        self.theta_true = theta_true
        self.p = p
        self.B = B
        self.T = T
        self.K = K
        self.O_t = 0
        self.num_action = num_action
        self.delta = delta
        self.mu_max = mu_max
        
        #initialize b_hat, V_hat, b_tilde, V_tilde
        self.b_hat = np.zeros((self.dim,1))
        self.b_tilde = np.zeros((self.dim,1))
        self.V_hat = np.diag([self.lamb  / self.theta_std ** 2] * self.dim)
        self.V_tilde = np.diag([self.lamb / self.theta_std ** 2] * self.dim)
        
        self.alpha_hat = np.sqrt(2*np.log(1/self.delta) + np.log(npla.det(self.V_hat)/self.lamb**self.dim))+np.sqrt(self.lamb)*npla.norm(self.theta_true)
        self.alpha_tilde = np.sqrt(2*np.log(1/self.delta) + np.log(npla.det(self.V_tilde)/self.lamb**self.dim))+np.sqrt(self.lamb)*npla.norm(self.theta_true)
        # self.alpha_hat = 1
        # self.alpha_tilde = 1
    
   # Algorithm 2 Online primal-dual algorithm with learning component
    
    def get_observation(self, a, s_t):
        temp_action = self.action.iloc[a,].values
        intercept = np.array([1])  # Intercept term
        phi_a = np.concatenate([intercept, temp_action, [s_t], np.multiply(temp_action, s_t)]).reshape(self.dim, 1)
        
        return phi_a
    
       # calculate phi_bar
    def Barphi(self,a, S):
        barphi = np.zeros((self.dim, 1))
        for i in range(self.K):
            phi = self.get_observation(a, S[i])
            barphi += phi*self.p[i]
        return barphi
    
    def q1(self, s_t):
        return 1
    # 0.8 + 0.2 * s_t**2  # Quadratic term for variation

    def q0(self, s_t):
        return 0
    # 0.1 * s_t**2  # Smaller coefficients to ensure q1 > q0
    
    # Algorithm 3 Online learning algorithm
    def pick_action(self, s_t, S, Y_t0, Y_t1):
        theta_tilde = np.matmul(npla.inv(self.V_tilde), self.b_tilde) #update OLS estimates of theta
        theta_hat = np.matmul(npla.inv(self.V_hat), self.b_hat)
        
        upper_hat = np.zeros(self.num_action)
        upper_tilde_st = np.zeros(self.num_action)
        for a in range(self.num_action):
            barphi = self.Barphi(a,S)
            phi = self.get_observation(a, s_t)

            #recommender
#             upper_hat[a] = np.matmul(barphi.T, theta_hat) + self.alpha_hat*np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_hat)), barphi))
            upper_hat[a] = np.matmul(barphi.T, theta_hat).item() + np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_hat)), barphi).item()) 
            #revealer with st
#             upper_tilde_st[a] = np.matmul(phi.T, theta_tilde) + self.alpha_tilde*np.sqrt(np.matmul(np.matmul(phi.T, npla.inv(self.V_tilde)), phi)) 
            upper_tilde_st[a] = np.matmul(phi.T, theta_tilde).item() + np.sqrt(np.matmul(np.matmul(phi.T, npla.inv(self.V_tilde)), phi).item()) 
        #print(upper_tilde)

        A_hat = np.argmax(upper_hat)
        A_tilde_st = np.argmax(upper_tilde_st)       
        
#         if (self.B < self.T): 
#             print("A hat:", A_hat)
#             print("A tilde:", A_tilde_st)
            
        o_t = self.B/self.T
    
        O_t = np.random.binomial(1, o_t)
        
        if (O_t == 1):
            # Y_t = np.random.binomial(1, self.q1(s_t))
            if (Y_t1 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat
      
        else:
            # Y_t = np.random.binomial(1, self.q0(s_t))
            if (Y_t0 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat
            
        return A_hat_star, O_t
        
    def update_parameter(self, s_t, X_t, phi_A, O_t, Y_t1, Y_t0):       
        Y_t = Y_t1 if O_t == 1 else Y_t0
        if Y_t == 1:
            self.b_hat = self.b_tilde
            self.V_hat = self.V_tilde
            self.alpha_hat = self.alpha_tilde

        #update revealer
        self.V_tilde += np.matmul(phi_A, phi_A.T)
        self.b_tilde += phi_A * X_t
        self.alpha_tilde = np.sqrt(2*np.log(1/self.delta) + np.log(npla.det(self.V_tilde)/(self.lamb**self.dim)))+np.sqrt(self.lamb)*(npla.norm(self.theta_true))
        

class UCB_nonudge(Agent):
    def __init__(self, theta_std, lamb, dim, action, p, B, T, K, delta, theta_true, num_action, mu_max):
        Agent.__init__(self, theta_std, lamb, dim, action, p, B, T, K, delta, theta_true, num_action, mu_max)
    
        # Algorithm 3 Online learning algorithm
    def pick_action(self, s_t, S, Y_t0, Y_t1):
        theta_tilde = np.matmul(npla.inv(self.V_tilde), self.b_tilde) #update OLS estimates of theta
        theta_hat = np.matmul(npla.inv(self.V_hat), self.b_hat)
        
        upper_hat = np.zeros(self.num_action)
        upper_tilde_st = np.zeros(self.num_action)
        for a in range(self.num_action):
            barphi = self.Barphi(a,S)
            phi = self.get_observation(a, s_t)

            #recommender
#             upper_hat[a] = np.matmul(barphi.T, theta_hat) + self.alpha_hat*np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_hat)), barphi))
            upper_hat[a] = np.matmul(barphi.T, theta_hat).item() + np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_hat)), barphi).item()) 
            #revealer with st
#             upper_tilde_st[a] = np.matmul(phi.T, theta_tilde) + self.alpha_tilde*np.sqrt(np.matmul(np.matmul(phi.T, npla.inv(self.V_tilde)), phi)) 
            upper_tilde_st[a] = np.matmul(phi.T, theta_tilde).item() + np.sqrt(np.matmul(np.matmul(phi.T, npla.inv(self.V_tilde)), phi).item()) 
        #print(upper_tilde)

        A_hat = np.argmax(upper_hat)
        A_tilde_st = np.argmax(upper_tilde_st)       
        
#         if (self.B < self.T): 
#             print("A hat:", A_hat)
#             print("A tilde:", A_tilde_st)
            
        o_t = 0
        O_t = 0
        
            # Y_t = np.random.binomial(1, self.q0(s_t))
        if (Y_t0 == 1):
            A_hat_star = A_tilde_st
        else:
            A_hat_star = A_hat
            
        return A_hat_star, O_t    

class UCB_noreveal(Agent):
    def __init__(self, theta_std, lamb, dim, action, p, B, T, K, delta, theta_true, num_action, mu_max):
        Agent.__init__(self, theta_std, lamb, dim, action, p, B, T, K, delta, theta_true, num_action, mu_max)
    
        # Algorithm 3 Online learning algorithm
    def pick_action(self, s_t, S, Y_t0, Y_t1):
        # theta_tilde = np.matmul(npla.inv(self.V_tilde), self.b_tilde) #update OLS estimates of theta
        theta_hat = np.matmul(npla.inv(self.V_hat), self.b_hat)
        
        upper_hat = np.zeros(self.num_action)
        # upper_tilde_st = np.zeros(self.num_action)
        for a in range(self.num_action):
            barphi = self.Barphi(a,S)
            phi = self.get_observation(a, s_t)

            #recommender
#             upper_hat[a] = np.matmul(barphi.T, theta_hat) + self.alpha_hat*np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_hat)), barphi))
            upper_hat[a] = np.matmul(barphi.T, theta_hat).item() + np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_hat)), barphi).item()) 
            #revealer with st
#             upper_tilde_st[a] = np.matmul(phi.T, theta_tilde) + self.alpha_tilde*np.sqrt(np.matmul(np.matmul(phi.T, npla.inv(self.V_tilde)), phi)) 
            # upper_tilde_st[a] = np.matmul(phi.T, theta_tilde).item() + np.sqrt(np.matmul(np.matmul(phi.T, npla.inv(self.V_tilde)), phi).item()) 
        #print(upper_tilde)

        A_hat = np.argmax(upper_hat)
        # A_tilde_st = np.argmax(upper_tilde_st)       
        
#         if (self.B < self.T): 
#             print("A hat:", A_hat)
#             print("A tilde:", A_tilde_st)
            
        o_t = 0
        O_t = 0
        
        A_hat_star = A_hat
            
        return A_hat_star, O_t   
# In[236]:

class Two_Agent_pd2(Agent):
    def __init__(self, theta_std, lamb, dim, action, p, B, beta_t, T, K, delta, theta_true, num_action, c, mu_max):
        Agent.__init__(self, theta_std, lamb, dim, action, p, B, T, K, delta, theta_true, num_action, mu_max)
        self.c = c
        self.o_list = 0
        
        self.y = 0
        self.beta_t = beta_t
        
    def PrimalDual(self, u, v, tilde_a_t, hat_a_t, S, s_t):
        diff_open_prob = self.q1(s_t) - self.q0(s_t)
        barphi_tilde = self.Barphi(tilde_a_t, S)
        barphi_hat = self.Barphi(hat_a_t, S)
        if (tilde_a_t != hat_a_t and self.beta_t < npla.norm(barphi_tilde-barphi_hat) and (u - v) * diff_open_prob > 0):
            x_t = 1/npla.norm(barphi_tilde-barphi_hat) - self.beta_t/(npla.norm(barphi_tilde-barphi_hat))**2
            if (diff_open_prob * (u - v) - self.y <= 0):
                #  + npla.norm(barphi_tilde-barphi_hat)*1*x_t
                self.beta_t = npla.norm(barphi_tilde-barphi_hat)
                x_t = 0
        else:
            self.beta_t = npla.norm(barphi_tilde-barphi_hat)
            x_t = 0
        
        indicator = 1 if tilde_a_t != hat_a_t else 0
        if (self.y < 1 and (diff_open_prob * (u - v)  + npla.norm(barphi_tilde-barphi_hat)*indicator*x_t) / self.q1(s_t)  - self.y > 0):
            self.beta_t = max(self.beta_t, npla.norm(barphi_tilde-barphi_hat)*(1-self.B+self.o_list))
            z_t = (diff_open_prob * (u - v)  + npla.norm(barphi_tilde-barphi_hat)*indicator*x_t) / self.q1(s_t)  - self.y
            o_t = min((diff_open_prob * (u - v)  + npla.norm(barphi_tilde-barphi_hat)*indicator*x_t) / self.q1(s_t), self.B-self.o_list, 1)
            self.y = self.y*(1 + o_t/self.B) + o_t/((self.c - 1)*self.B)
        else:
            self.beta_t = npla.norm(barphi_tilde-barphi_hat)
            z_t = 0
            o_t = 0
            
        self.o_list += o_t
        return o_t
    
     # Online learning algorithm
    def pick_action(self, s_t, S, Y_t0, Y_t1):
        theta_tilde = np.matmul(npla.inv(self.V_tilde), self.b_tilde) #update OLS estimates of theta
        theta_hat = np.matmul(npla.inv(self.V_hat), self.b_hat)
        
        upper_tilde = np.zeros(self.num_action)
        upper_hat = np.zeros(self.num_action)
        upper_tilde_st = np.zeros(self.num_action)
        for a in range(self.num_action):
            barphi = self.Barphi(a,S)
            phi = self.get_observation(a, s_t)
            #revealer
#             upper_tilde[a] = np.matmul(barphi.T, theta_tilde) + self.alpha_tilde*np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_tilde)), barphi))
            upper_tilde[a] = np.matmul(barphi.T, theta_tilde).item() + np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_tilde)), barphi).item()) 
            #recommender
#             upper_hat[a] = np.matmul(barphi.T, theta_hat) + self.alpha_hat*np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_hat)), barphi)) 
            upper_hat[a] = np.matmul(barphi.T, theta_hat).item() + np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_hat)), barphi).item()) 
            #revealer with st
#             upper_tilde_st[a] = np.matmul(phi.T, theta_tilde) + self.alpha_tilde*np.sqrt(np.matmul(np.matmul(phi.T, npla.inv(self.V_tilde)), phi)) 
            upper_tilde_st[a] = np.matmul(phi.T, theta_tilde).item() + np.sqrt(np.matmul(np.matmul(phi.T, npla.inv(self.V_tilde)), phi).item()) 
        
        #print(upper_tilde)
        A_tilde = np.argmax(upper_tilde)
        A_hat = np.argmax(upper_hat)
        A_tilde_st = np.argmax(upper_tilde_st)
        
        #compute expected rewards of the revealer
        barphi_A_tilde = self.Barphi(A_tilde,S)/(2*self.mu_max)
        phi_A_tilde_st = self.get_observation(A_tilde_st, s_t)/(2*self.mu_max)        
           
#         u_S = []
#         for s in S:
#             phi_A_tilde_s = self.get_observation(A_tilde_st, s)
#             max_theta_tilde_s = cp.Variable(self.dim)
#             cons = [cp.SOC(self.alpha_tilde, np.matmul(spla.sqrtm(self.V_tilde),(max_theta_tilde_s - theta_tilde.flatten())))]
#             prob = cp.Problem(cp.Maximize(np.matmul(phi_A_tilde_s.T, max_theta_tilde_s)), cons)
#             u_S.append(prob.solve(solver = cp.SCS))
            
#         u_max = np.max(u_S)
#         u_min = np.min(u_S)
        
#         theta_tilde_scale = (theta_tilde - u_min)/u_max
        
        max_theta_tilde_st = cp.Variable(self.dim)
        cons1 = [cp.SOC(self.alpha_tilde, np.matmul(spla.sqrtm(self.V_tilde),(max_theta_tilde_st - theta_tilde.flatten())))]
        prob1 = cp.Problem(cp.Maximize(np.matmul(phi_A_tilde_st.T, max_theta_tilde_st)), cons1)
        u_st = prob1.solve(solver = cp.SCS)
#         print("u_st of pd2:",u_st)
        
        max_theta_tilde = cp.Variable(self.dim)
        cons2 = [cp.SOC(self.alpha_tilde, np.matmul(spla.sqrtm(self.V_tilde),(max_theta_tilde - theta_tilde.flatten())))]
        prob2 = cp.Problem(cp.Maximize(np.matmul(barphi_A_tilde.T, max_theta_tilde)), cons2)
        v = prob2.solve(solver = cp.SCS)
        
        
        o_t = self.PrimalDual(u_st, v,  A_tilde, A_hat, S, s_t)  
        O_t = np.random.binomial(1, o_t)
        if (O_t == 1):
            # Y_t = np.random.binomial(1, self.q1(s_t))
            if (Y_t1 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat
      
        else:
            # Y_t = np.random.binomial(1, self.q0(s_t))
            if (Y_t0 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat
            
        return A_hat_star, O_t, A_tilde, A_hat, o_t
    
    def update_beta_t (self, beta_t, t):
        self.beta_t = (1.2*beta_t)/(np.sqrt(t/10 + 1)+0.1)  #sqrt t
        # self.beta_t = 1.2 * beta_t * np.log(10) * np.sqrt(10) / (np.sqrt(t+0.01) * np.log(self.B))  #sqrt t




class Two_Agent_pd1(Two_Agent_pd2):
    def __init__(self, theta_std, lamb, dim, action, p, B, beta_t, T, K, delta, theta_true, num_action, c, mu_max):
        Two_Agent_pd2.__init__(self, theta_std, lamb, dim, action, p, B, beta_t, T, K, delta, theta_true, num_action, c, mu_max)
    
        # Algorithm 3 Online learning algorithm
    def pick_action(self, s_t, S, Y_t0, Y_t1):
        theta_tilde = np.matmul(npla.inv(self.V_tilde), self.b_tilde) #update OLS estimates of theta
        theta_hat = np.matmul(npla.inv(self.V_hat), self.b_hat)
        
        upper_tilde = np.zeros(self.num_action)
        upper_hat = np.zeros(self.num_action)
        upper_tilde_st = np.zeros(self.num_action)
        for a in range(self.num_action):
            barphi = self.Barphi(a,S)
            phi = self.get_observation(a, s_t)
            #revealer
#             upper_tilde[a] = np.matmul(barphi.T, theta_tilde) + self.alpha_tilde*np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_tilde)), barphi))
            upper_tilde[a] = np.matmul(barphi.T, theta_tilde).item() + np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_tilde)), barphi).item()) 
            #recommender
#             upper_hat[a] = np.matmul(barphi.T, theta_hat) + self.alpha_hat*np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_hat)), barphi)) 
            upper_hat[a] = np.matmul(barphi.T, theta_hat).item() + np.sqrt(np.matmul(np.matmul(barphi.T, npla.inv(self.V_hat)), barphi).item()) 
            #revealer with st
#             upper_tilde_st[a] = np.matmul(phi.T, theta_tilde) + self.alpha_tilde*np.sqrt(np.matmul(np.matmul(phi.T, npla.inv(self.V_tilde)), phi)) 
            upper_tilde_st[a] = np.matmul(phi.T, theta_tilde).item() + np.sqrt(np.matmul(np.matmul(phi.T, npla.inv(self.V_tilde)), phi).item()) 
        
        #print(upper_tilde)
        A_tilde = np.argmax(upper_tilde)
        A_hat = np.argmax(upper_hat)
        A_tilde_st = np.argmax(upper_tilde_st)
        
        #compute expected rewards of the revealer
        barphi_A_tilde = self.Barphi(A_tilde,S)/(2*self.mu_max)
        phi_A_tilde_st = self.get_observation(A_tilde_st, s_t)/(2*self.mu_max)
        
#         u_S = []
#         for s in S:
#             phi_A_tilde_s = self.get_observation(A_tilde_st, s)
#             max_theta_tilde_s = cp.Variable(self.dim)
#             cons = [cp.SOC(self.alpha_tilde, np.matmul(spla.sqrtm(self.V_tilde),(max_theta_tilde_s - theta_tilde.flatten())))]
#             prob = cp.Problem(cp.Maximize(np.matmul(phi_A_tilde_s.T, max_theta_tilde_s)), cons)
#             u_S.append(prob.solve(solver = cp.SCS))
            
#         u_max = np.max(u_S)
#         u_min = np.min(u_S)
        
#         theta_tilde_scale = (theta_tilde - u_min)/u_max
        
        max_theta_tilde_st = cp.Variable(self.dim)
        cons1 = [cp.SOC(self.alpha_tilde, np.matmul(spla.sqrtm(self.V_tilde),(max_theta_tilde_st - theta_tilde.flatten())))]
        prob1 = cp.Problem(cp.Maximize(np.matmul(phi_A_tilde_st.T, max_theta_tilde_st)), cons1)
        u_st = prob1.solve(solver = cp.SCS)
#         print("u_st of pd2:",u_st)
        
        max_theta_tilde = cp.Variable(self.dim)
        cons2 = [cp.SOC(self.alpha_tilde, np.matmul(spla.sqrtm(self.V_tilde),(max_theta_tilde - theta_tilde.flatten())))]
        prob2 = cp.Problem(cp.Maximize(np.matmul(barphi_A_tilde.T, max_theta_tilde)), cons2)
        v = prob2.solve(solver = cp.SCS)
                   
        o_t = self.PrimalDual(u_st, v,  1, 1, S, s_t) #no second constraint
        
        O_t = np.random.binomial(1, o_t)
        
        if (O_t == 1):
            # Y_t = np.random.binomial(1, self.q1(s_t))
            if (Y_t1 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat
      
        else:
            # Y_t = np.random.binomial(1, self.q0(s_t))
            if (Y_t0 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat
        return A_hat_star, O_t, o_t



# Omniscient
class Omniscient:
    def __init__(self, dim, action, p, B, c, T, K, num_action, theta_true):
        self.dim = dim
        self.action = action
        self.theta_true = theta_true
        self.p = p
        self.B = B
        self.c = c
        self.T = T
        self.K = K
        self.num_action = num_action
    
    def get_observation(self, a, s_t):
        temp_action = self.action.iloc[a,].values
        intercept = np.array([1])  # Intercept term
        phi_a = np.concatenate([intercept, temp_action, [s_t], np.multiply(temp_action, s_t)]).reshape(self.dim, 1)
        
        return phi_a
    
    # calculate phi_bar
    def Barphi(self,a, S):
        barphi = np.zeros((self.dim, 1))
        for i in range(self.K):
            phi = self.get_observation(a, S[i])
            barphi += phi*self.p[i]
        return barphi
    
    def q1(self, s_t):
        return 1
    # 0.8 + 0.2 * s_t**2  # Quadratic term for variation

    def q0(self, s_t):
        return 0
    # 0.1 * s_t**2  # Smaller coefficients to ensure q1 > q0
    
    def omniscient_reward(self, s_t, S):
        mu_omni = []
        v_omni = []
        
        for a in range(self.num_action):
            phi_omni = self.get_observation (a, s_t)
            barphi_omni = self.Barphi(a, S)
            mu_omni.append(np.matmul(phi_omni.T, self.theta_true))
            v_omni.append(np.matmul(barphi_omni.T, self.theta_true))

        mu_omni_star = np.max(mu_omni)
        v_omni_star = np.max(v_omni)
        
        return mu_omni_star, v_omni_star

    def sort_index(self, lst, rev=True):
        index = range(len(lst))
        s = sorted(index, reverse=rev, key=lambda i: lst[i])
        return s


    def omniscient_strategy(self, s_t_list, S, Y_t0_list, Y_t1_list):
        mu_omni_star = [0]*self.T
        v_omni_star = [0]*self.T
        diff_times_uv = [0]*self.T
        for t in range(self.T):
            mu_omni_star[t], v_omni_star[t] = self.omniscient_reward(s_t_list[t], S)
            diff_times_uv[t] = (self.q1(s_t_list[t]) - self.q0(s_t_list[t])) * (mu_omni_star[t] - v_omni_star[t])
        
        reward_omni_list = v_omni_star.copy() # create a list of reward_t

        index_ot1 = self.sort_index(diff_times_uv)[:self.B]  #index of the B largest mu_omni_star
              
        O_omni_list = np.zeros(T) # create a list of O_t
        #difference to be positive u-v
        for i in index_ot1:
            if (mu_omni_star[i] > v_omni_star[i]):
                O_omni_list[i] = 1
        
        for i in range(self.T):
            if (O_omni_list[i] == 1):
                reward_omni_list[i] = mu_omni_star[i] if Y_t1_list[i] == 1 else v_omni_star[i]
            else:
                reward_omni_list[i] = mu_omni_star[i] if Y_t0_list[i] == 1 else v_omni_star[i]
     
        return reward_omni_list, O_omni_list, v_omni_star, mu_omni_star



#function for calculating regret, with constraint 2
def simu_regret(s, theta_std, lamb, q, action, B, T, K, delta, num_action, theta_true):
    np.random.seed(123 + s)
    S = np.random.uniform(0, 1, K) #context   
    p_0 = np.random.uniform(0, 1, K)
    p = p_0/np.sum(p_0)

    
    # s_t_list = np.random.multinomial(1, p, T) #sequence
    # s_t_list = np.argmax(s_t_list==1, axis=1)
    # s_t_list = S[s_t_list]
    s_t_list = np.random.choice(K, T, p=p) #sequence
    s_t_list = S[s_t_list]

    Y_t0_list = np.zeros(T)
    Y_t1_list = np.zeros(T)
    for t in range(T):
        Y_t0_list[t] = 0
        # np.random.binomial(1, 0.1 * s_t_list[t]**2)
        Y_t1_list[t] = 1
        # np.random.binomial(1, 0.8 + 0.2 * s_t_list[t]**2)
    
    c = (1+1/B)**B
    
    # get omniscient reward and O_t
    omniscient = Omniscient(q, action, p, B, c, T, K, num_action, theta_true)
    reward_omni_list, O_omni_list,v_omni_star, mu_omni_star = omniscient.omniscient_strategy(s_t_list, S, Y_t0_list, Y_t1_list)
    mu_max = np.max(mu_omni_star)

    # saving cumu regret
    cumu_regret_naive = np.zeros(T)
    cumu_regret_algorithm1 = np.zeros(T)
    cumu_regret_algorithm2 = np.zeros(T)
    cumu_regret_ucb_nonudge = np.zeros(T)
    cumu_regret_ucb_noreveal = np.zeros(T)
    # saving o_t
    o_pd1_list = np.zeros(T)
    o_pd2_list = np.zeros(T)
    
    regret_naive = 0
    regret_ucb_nonudge = 0
    regret_ucb_noreveal = 0
    regret_algorithm2 = 0
    regret_algorithm1 = 0
    
    #calculate initial beta_t
    beta_t_list = []
    for a in range(num_action-1):
        for b in range(a+1, num_action): 
            if (a != b):
                beta_t_list.append(npla.norm(omniscient.Barphi(a, S)-omniscient.Barphi(b, S)))
    beta_t = np.min(beta_t_list)  #initial value
        
    agent = Agent(theta_std, lamb, q, action, p, B, T, K, delta, theta_true, num_action, mu_max)
    ucb_nonudge = UCB_nonudge(theta_std, lamb, q, action, p, B, T, K, delta, theta_true, num_action, mu_max)
    ucb_noreveal = UCB_noreveal(theta_std, lamb, q, action, p, B, T, K, delta, theta_true, num_action, mu_max)
    two_agent_pd2 = Two_Agent_pd2(theta_std, lamb, q, action, p, B, beta_t, T, K, delta, theta_true, num_action, c, mu_max)
    two_agent_pd1 = Two_Agent_pd1(theta_std, lamb, q, action, p, B, beta_t, T, K, delta, theta_true, num_action, c, mu_max)
    
    for t in range(T):
        s_t = s_t_list[t]
        reward_omni = reward_omni_list[t]
        # get action

        # get Y
        Y_t0 = Y_t0_list[t]
        Y_t1 = Y_t1_list[t]
        
        A_naive, O_naive = agent.pick_action(s_t, S, Y_t0, Y_t1)
        A_ucb_nonudge, O_ucb_nonudge = ucb_nonudge.pick_action(s_t, S, Y_t0, Y_t1)
        A_ucb_noreveal, O_ucb_noreveal = ucb_noreveal.pick_action(s_t, S, Y_t0, Y_t1)
        A_algorithm2, O_algorithm2, A_tilde, A_hat, o_algorithm2= two_agent_pd2.pick_action(s_t, S, Y_t0, Y_t1)
        A_algorithm1, O_algorithm1, o_algorithm1 = two_agent_pd1.pick_action(s_t, S, Y_t0, Y_t1)

        o_pd1_list[t] = o_algorithm1
        o_pd2_list[t] = o_algorithm2
    
        # get observation for updating paras
        phi_naive = agent.get_observation(A_naive, s_t)
        phi_ucb_nonudge = ucb_nonudge.get_observation(A_ucb_nonudge, s_t)
        phi_ucb_noreveal = ucb_noreveal.get_observation(A_ucb_noreveal, s_t)
        phi_algorithm2 = two_agent_pd2.get_observation(A_algorithm2, s_t)
        phi_algorithm1 = two_agent_pd1.get_observation(A_algorithm1, s_t)
    
        
        # calculate expected reward and observed reward
        expected_reward_naive = np.matmul(phi_naive.T, theta_true).item()
        observed_reward_naive = np.random.normal(expected_reward_naive, 0.1)
        if (O_naive == 0 and Y_t0 == 0):
            barphi_naive = agent.Barphi(A_naive, S)
            expected_reward_naive = np.matmul(barphi_naive.T, theta_true).item()
        if (O_naive == 1 and Y_t1 == 0):
            barphi_naive = agent.Barphi(A_naive, S)
            expected_reward_naive = np.matmul(barphi_naive.T, theta_true).item()

        expected_reward_ucb_nonudge = np.matmul(phi_ucb_nonudge.T, theta_true).item()
        observed_reward_ucb_nonudge = np.random.normal(expected_reward_ucb_nonudge, 0.1)
        if (Y_t0 == 0):
            barphi_ucb_nonudge = ucb_nonudge.Barphi(A_ucb_nonudge, S)
            expected_reward_ucb_nonudge = np.matmul(barphi_ucb_nonudge.T, theta_true).item()

        expected_reward_ucb_noreveal = np.matmul(phi_ucb_noreveal.T, theta_true).item()
        observed_reward_ucb_noreveal = np.random.normal(expected_reward_ucb_noreveal, 0.1)
        barphi_ucb_noreveal = ucb_noreveal.Barphi(A_ucb_noreveal, S)
        expected_reward_ucb_noreveal = np.matmul(barphi_ucb_noreveal.T, theta_true).item()
        
              
        expected_reward_algorithm2 = np.matmul(phi_algorithm2.T, theta_true).item()
        observed_reward_algorithm2 = np.random.normal(expected_reward_algorithm2, 0.1)
        if (O_algorithm2 == 0 and Y_t0 == 0):
            barphi_algorithm2 = two_agent_pd2.Barphi(A_algorithm2, S)
            expected_reward_algorithm2 = np.matmul(barphi_algorithm2.T, theta_true).item()
        if (O_algorithm2 == 1 and Y_t1 == 0):
            barphi_algorithm2 = two_agent_pd2.Barphi(A_algorithm2, S)
            expected_reward_algorithm2 = np.matmul(barphi_algorithm2.T, theta_true).item()
        # calculate expected reward and observed reward

        expected_reward_algorithm1 = np.matmul(phi_algorithm1.T, theta_true).item()
        observed_reward_algorithm1 = np.random.normal(expected_reward_algorithm1, 0.1)
        if (O_algorithm1 == 0 and Y_t0 == 0):
            barphi_algorithm1 = two_agent_pd1.Barphi(A_algorithm1, S)
            expected_reward_algorithm1 = np.matmul(barphi_algorithm1.T, theta_true).item()
        if (O_algorithm1 == 1 and Y_t1 == 0):
            barphi_algorithm1 = two_agent_pd1.Barphi(A_algorithm1, S)
            expected_reward_algorithm1 = np.matmul(barphi_algorithm1.T, theta_true).item()
        # calculate observed reward
        
        # update parameters
        agent.update_parameter(s_t, observed_reward_naive, phi_naive, O_naive, Y_t1, Y_t0)
        ucb_nonudge.update_parameter(s_t, observed_reward_ucb_nonudge, phi_ucb_nonudge, O_ucb_nonudge, Y_t1, Y_t0)
        ucb_noreveal.update_parameter(s_t, observed_reward_ucb_noreveal, phi_ucb_noreveal, O_ucb_noreveal, Y_t1, Y_t0)
        two_agent_pd2.update_parameter(s_t, observed_reward_algorithm2, phi_algorithm2, O_algorithm2, Y_t1, Y_t0)
        two_agent_pd2.update_beta_t(beta_t, t)  #update beta_t so that it is decreasing with time
        
        two_agent_pd1.update_parameter(s_t, observed_reward_algorithm1, phi_algorithm1, O_algorithm1, Y_t1, Y_t0)
        
        
        # calculate regret
        regret_naive += reward_omni - expected_reward_naive
        regret_algorithm2 += reward_omni - expected_reward_algorithm2
        regret_algorithm1 += reward_omni - expected_reward_algorithm1
        regret_ucb_nonudge += reward_omni - expected_reward_ucb_nonudge
        regret_ucb_noreveal += reward_omni - expected_reward_ucb_noreveal
        
        cumu_regret_naive[t] = regret_naive
        cumu_regret_algorithm2[t] = regret_algorithm2
        cumu_regret_algorithm1[t] = regret_algorithm1
        cumu_regret_ucb_nonudge[t] = regret_ucb_nonudge
        cumu_regret_ucb_noreveal[t] = regret_ucb_noreveal
    return cumu_regret_naive, cumu_regret_algorithm2, cumu_regret_algorithm1, cumu_regret_ucb_nonudge, cumu_regret_ucb_noreveal, o_pd1_list, o_pd2_list

if __name__ == "__main__":
    mp.freeze_support()  # required on Windows/macOS spawn; harmless elsewhere
    for m in range(50, 101):
        np.random.seed(m)
        theta_true = np.random.uniform(0, 1, q)
        theta_true = theta_true.reshape(q, 1) #true parameter
        num_processes = mp.cpu_count()  # number of available CPU cores

        # Define lists to store the simulation results
        results_naive = []
        results_ucb_nonudge = []
        results_ucb_noreveal = []
        results_algorithm1 = []
        results_algorithm2 = []
        # results_algorithm2_log = []
        # results_ucb = []

        o_pd1_all = []
        o_pd2_all = []

        simu_regret_partial = partial(simu_regret, theta_std=theta_std, lamb=lamb, q=q, action=action, B=B, T=T, K=K, delta=delta, num_action=num_action, theta_true=theta_true)
        # use context manager to ensure pool cleans up even on error
        with mp.Pool(processes=num_processes) as pool:
            for cumu_regret_naive, cumu_regret_algorithm2, cumu_regret_algorithm1, cumu_regret_ucb_nonudge, cumu_regret_ucb_noreveal, o_pd1_list, o_pd2_list in pool.imap_unordered(simu_regret_partial, range(Simu), chunksize=1):
                results_naive.append(cumu_regret_naive)
                results_ucb_nonudge.append(cumu_regret_ucb_nonudge)
                results_ucb_noreveal.append(cumu_regret_ucb_noreveal)
                results_algorithm1.append(cumu_regret_algorithm1)
                results_algorithm2.append(cumu_regret_algorithm2)
                o_pd1_all.append(o_pd1_list)
                o_pd2_all.append(o_pd2_list)
    #       results_algorithm2_log.append(cumu_regret_algorithm2_log)
    #       results_ucb.append(cumu_regret_ucb)

        # Create dataframes for results
        df_naive = pd.DataFrame(results_naive)
        df_ucb_nonudge = pd.DataFrame(results_ucb_nonudge)
        df_ucb_noreveal = pd.DataFrame(results_ucb_noreveal)
        df_algorithm1 = pd.DataFrame(results_algorithm1)
        df_algorithm2 = pd.DataFrame(results_algorithm2)
        o_pd1_all = pd.DataFrame(o_pd1_all)
        o_pd2_all = pd.DataFrame(o_pd2_all)

        # Save dataframes to csv files
        name_naive = "results_naive" + str(m) + ".csv"
        name_ucb_nonudge = "results_ucb_nonudge" + str(m) + ".csv"
        name_ucb_noreveal = "results_ucb_noreveal" + str(m) + ".csv"
        name_algorithm1 = "results_algorithm1" + str(m) + ".csv"
        name_algorithm2 = "results_algorithm2" + str(m) + ".csv"
        name_pd1 = "o_pd1" + str(m) + ".csv"
        name_pd2 = "o_pd2" + str(m) + ".csv"
        
        df_naive.to_csv(name_naive, index=False)
        df_ucb_nonudge.to_csv(name_ucb_nonudge, index=False)
        df_ucb_noreveal.to_csv(name_ucb_noreveal, index=False)
        df_algorithm1.to_csv(name_algorithm1, index=False)
        df_algorithm2.to_csv(name_algorithm2, index=False)
        o_pd1_all.to_csv(name_pd1, index=False)
        o_pd2_all.to_csv(name_pd2, index=False)