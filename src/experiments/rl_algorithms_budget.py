"""
Five RL Algorithms for the budget constraint experiment:
1. DABBI-Nudging I
2. DABBI-Nudging II
3. UCB-UniformNudge
4. UCB-NoNudge
5. UCB-NoReveal
"""
import reward_definition
import smoothing_function

import numpy as np
import pandas as pd
import os
import scipy.linalg as spla
import scipy.stats as stats
import cvxpy as cp
from datetime import datetime, timedelta

import sim_env_v4

class UCB_NoNudge:
    def __init__(self, theta_std, lamb, dim, B, T, delta, num_action):
        self.theta_std = theta_std
        self.lamb = lamb
        self.dim = dim
        self.B = B
        self.T = T
        self.num_action = num_action
        self.delta = delta

        # initialize b_hat, V_hat, b_tilde, V_tilde
        self.b_hat = np.zeros((self.dim, 1))
        self.b_tilde = np.zeros((self.dim, 1))
        self.V_hat = np.diag([self.lamb / self.theta_std ** 2] * self.dim)
        self.V_tilde = np.diag([self.lamb / self.theta_std ** 2] * self.dim)

        self.alpha_hat = np.sqrt(
            2 * np.log(1 / self.delta) + np.log(spla.det(self.V_hat) / self.lamb ** self.dim)) + np.sqrt(
            self.lamb) #* spla.norm(self.theta_true)
        self.alpha_tilde = np.sqrt(
            2 * np.log(1 / self.delta) + np.log(spla.det(self.V_tilde) / self.lamb ** self.dim)) + np.sqrt(
            self.lamb) #* spla.norm(self.theta_true)

        # Reward definition function
        self.reward_def_func = lambda brushing_quality, current_action, b_bar, a_bar: \
                      reward_definition.calculate_reward(brushing_quality, 100, 100, \
                                        current_action, b_bar, a_bar)

    def get_feature_dim(self):
        return self.dim

    @property
    def posterior_mean(self):
        return self.b_hat.flatten()

    @property
    def posterior_var(self):
        return self.V_hat.flatten()

    def process_alg_state(self, env_state):
        baseline_state = np.array([env_state[0], env_state[1], \
                                env_state[2], env_state[3], env_state[4], 1]) #, env_state[6]])
        advantage_state = np.copy(baseline_state)

        return advantage_state, baseline_state

    # def get_observation(self, a, advantage_state, baseline_state):
    #     # phi_a = np.concatenate([baseline_state, np.multiply(a, advantage_state)]).reshape(self.dim, 1)
    #     phi_a = np.hstack((baseline_state, np.multiply(advantage_state.T, a).T))
    #     return phi_a

    def get_observation(self, a, advantage_state, baseline_state):
        # Handle different input shapes - convert to 1D arrays first

        bs = np.array(baseline_state).flatten()  # Always convert to 1D
        adv = np.array(advantage_state).flatten()  # Always convert to 1D
        
        # # Replace nan values with 0
        # bs = np.nan_to_num(bs, nan=0.0)
        # adv = np.nan_to_num(adv, nan=0.0)

        baseline_dim = len(bs)
        adv_dim = len(adv)
        

        # One-hot block for action a: [0...adv...0] where block size = adv_dim and there are num_action blocks
        phi_adv = np.zeros(adv_dim * self.num_action)
        # Handle both scalar and vector inputs for action

        action_idx = int(a)
        if action_idx != 0:
            start = (action_idx -1) * adv_dim
            phi_adv[start:start + adv_dim] = adv[:adv_dim]

        # Final feature: [baseline; one-hot(a) ⊗ advantage]
        phi_a = np.concatenate([bs, phi_adv]).reshape(-1, 1)
        return phi_a

    def get_observation_bar(self, a, advantage_state, baseline_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values):
        bs = np.array(baseline_state).flatten()  # Always convert to 1D
        adv = np.array(advantage_state).flatten()  # Always convert to 1D

        baseline_dim = len(bs)
        adv_dim = len(adv)

        # change a_bar as a weighted sum of the discrete a_bar values
        a_bar = np.sum(prob_a_bar * a_bar_discrete_values)
        b_bar = np.sum(prob_b_bar * b_bar_discrete_values)
        bs[1] = b_bar
        bs[2] = a_bar
        adv[1] = b_bar
        adv[2] = a_bar

        phi_adv = np.zeros(adv_dim * self.num_action)
        action_idx = int(a)
        if action_idx != 0:
            start = (action_idx -1) * adv_dim
            phi_adv[start:start + adv_dim] = adv[:adv_dim]

        phi_a = np.concatenate([bs, phi_adv]).reshape(-1, 1)

        return phi_a

    def get_q0_q1(self, user_id, state):

        return sim_env_v4.get_logistic_app_open_probs(user_id, state)



    def pick_action(self, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, Y_t0, Y_t1, user_id = None, env_state = None, mu_max = None):
        theta_tilde = np.matmul(spla.inv(self.V_tilde), self.b_tilde)  # update OLS estimates of theta
        theta_hat = np.matmul(spla.inv(self.V_hat), self.b_hat)

        upper_hat = np.zeros(self.num_action + 1)
        upper_tilde_st = np.zeros(self.num_action + 1)
        for a in range(self.num_action + 1):
            phi_tilde = self.get_observation(a, advantage_state_tilde, baseline_state_tilde)
            phi_hat = self.get_observation_bar(a, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
            upper_tilde_st[a] = np.matmul(phi_tilde.T, theta_tilde) + np.sqrt(np.matmul(np.matmul(phi_tilde.T, spla.inv(self.V_tilde)), phi_tilde))
            upper_hat[a] = np.matmul(phi_hat.T, theta_hat) + np.sqrt(np.matmul(np.matmul(phi_hat.T, spla.inv(self.V_hat)), phi_hat))

        A_hat = np.argmax(upper_hat)
        A_tilde_st = np.argmax(upper_tilde_st)

        phi_hat = self.get_observation_bar(A_hat, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
        phi_tilde = self.get_observation(A_tilde_st, advantage_state_tilde, baseline_state_tilde)
        gamma_hat = np.matmul(phi_hat.T, theta_hat)
        mu_hat = np.matmul(phi_tilde.T, theta_tilde)
        o_t = 0
        O_t = 0

        if (Y_t0 == 1):
            A_hat_star = A_tilde_st
        else:
            A_hat_star = A_hat
        
        # Track whether the two argmax actions agree (1 if same action, 0 otherwise)
        delta = int(A_tilde_st == A_hat)
        distance = spla.norm(phi_hat - phi_tilde)

        return A_hat_star, O_t, o_t, mu_hat, gamma_hat, delta, distance

    def update_revealer_parameter(self, X_t, advantage_state, baseline_state, action):
        phi_A = self.get_observation(action, advantage_state, baseline_state)
        self.V_tilde += np.matmul(phi_A.T, phi_A) + np.diag([self.lamb / self.theta_std ** 2] * self.dim)
        # X_t is a scalar reward, so we multiply phi_A by X_t
        self.b_tilde += phi_A * X_t
        self.alpha_tilde = np.sqrt(
            2 * np.log(1 / self.delta) + np.log(spla.det(self.V_tilde) / (self.lamb ** self.dim))) + np.sqrt(
            self.lamb) #* (spla.norm(self.theta_true))
        # self.freq_tilde[index] = self.freq_tilde[index] + 1
        # self.p_tilde = self.freq_tilde / (np.sum(self.freq_tilde))

    def update_recommender_parameter(self):
        # update recommender
        self.b_hat = self.b_tilde.copy()
        self.V_hat = self.V_tilde.copy()
        self.alpha_hat = self.alpha_tilde.copy()

class UCB_UniformNudge(UCB_NoNudge):
    def __init__(self, theta_std, lamb, dim, B, T, delta, num_action, c):
        UCB_NoNudge.__init__(self, theta_std, lamb, dim, B, T, delta, num_action)

    def pick_action(self, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, Y_t0, Y_t1, user_id=None, env_state=None, mu_max = None):
        theta_tilde = np.matmul(spla.inv(self.V_tilde), self.b_tilde)  # update OLS estimates of theta
        theta_hat = np.matmul(spla.inv(self.V_hat), self.b_hat)

        upper_hat = np.zeros(self.num_action + 1)
        upper_tilde_st = np.zeros(self.num_action + 1)
        for a in range(self.num_action + 1):
            phi_tilde = self.get_observation(a, advantage_state_tilde, baseline_state_tilde)
            phi_hat = self.get_observation_bar(a, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
            upper_tilde_st[a] = np.matmul(phi_tilde.T, theta_tilde) + np.sqrt(np.matmul(np.matmul(phi_tilde.T, spla.inv(self.V_tilde)), phi_tilde))
            upper_hat[a] = np.matmul(phi_hat.T, theta_hat) + np.sqrt(np.matmul(np.matmul(phi_hat.T, spla.inv(self.V_hat)), phi_hat))

        A_hat = np.argmax(upper_hat)
        A_tilde_st = np.argmax(upper_tilde_st)

        phi_hat = self.get_observation_bar(A_hat, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
        phi_tilde = self.get_observation(A_tilde_st, advantage_state_tilde, baseline_state_tilde)
        gamma_hat = np.matmul(phi_hat.T, theta_hat)
        mu_hat = np.matmul(phi_tilde.T, theta_tilde)

        o_t = self.B / self.T
        

        O_t = np.random.binomial(1, o_t)

        if (O_t == 1):
            if (Y_t1 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat
      
        else:
            if (Y_t0 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat
        
        delta = int(A_tilde_st == A_hat)
        distance = spla.norm(phi_hat - phi_tilde)

        return A_hat_star, O_t, o_t, mu_hat, gamma_hat, delta, distance

class UCB_NoReveal(UCB_NoNudge):
    def __init__(self, theta_std, lamb, dim, B, T, delta, num_action):
        UCB_NoNudge.__init__(self, theta_std, lamb, dim, B, T, delta, num_action)

    def pick_action(self, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, Y_t0, Y_t1, user_id=None, env_state=None, mu_max = None):
        theta_tilde = np.matmul(spla.inv(self.V_tilde), self.b_tilde)  # update OLS estimates of theta
        theta_hat = np.matmul(spla.inv(self.V_hat), self.b_hat)

        upper_hat = np.zeros(self.num_action + 1)
        upper_tilde_st = np.zeros(self.num_action + 1)
        for a in range(self.num_action + 1):
            phi_tilde = self.get_observation(a, advantage_state_tilde, baseline_state_tilde)
            phi_hat = self.get_observation_bar(a, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
            upper_tilde_st[a] = np.matmul(phi_tilde.T, theta_tilde) + np.sqrt(np.matmul(np.matmul(phi_tilde.T, spla.inv(self.V_tilde)), phi_tilde))
            upper_hat[a] = np.matmul(phi_hat.T, theta_hat) + np.sqrt(np.matmul(np.matmul(phi_hat.T, spla.inv(self.V_hat)), phi_hat))

        A_hat = np.argmax(upper_hat)
        A_tilde_st = np.argmax(upper_tilde_st)


        phi_hat = self.get_observation_bar(A_hat, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
        phi_tilde = self.get_observation(A_tilde_st, advantage_state_tilde, baseline_state_tilde)
        gamma_hat = np.matmul(phi_hat.T, theta_hat)
        mu_hat = np.matmul(phi_tilde.T, theta_tilde)

        o_t = 0
        O_t = 0

        delta = int(A_tilde_st == A_hat)
        distance = spla.norm(phi_hat - phi_tilde)

        A_hat_star = A_hat

        return A_hat_star, O_t, o_t, mu_hat, gamma_hat, delta, distance


class DABBI_Nudging_II(UCB_NoNudge):
    def __init__(self, theta_std, lamb, dim, B, T, delta, num_action, c):
        UCB_NoNudge.__init__(self, theta_std, lamb, dim, B, T, delta, num_action)
        self.c = c
        self.o_list = 0

        self.y = 0
        # self.beta_t = beta_t

    def PrimalDual(self, u, v, tilde_a_t, hat_a_t, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, user_id, env_state, beta_t):
        barphi_tilde = self.get_observation_bar(tilde_a_t, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
        barphi_hat = self.get_observation_bar(hat_a_t, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)

        q0, q1 = self.get_q0_q1(user_id, env_state)
        diff_open_prob = q1 - q0
        self.beta_t = beta_t
        if (tilde_a_t != hat_a_t and self.beta_t < spla.norm(barphi_tilde - barphi_hat) and (u -v) * diff_open_prob  > 0):
            x_t = 1 / spla.norm(barphi_tilde - barphi_hat) - self.beta_t / (spla.norm(barphi_tilde - barphi_hat)) ** 2
            if ((u - v) * diff_open_prob - self.y <= 0):
                self.beta_t = spla.norm(barphi_tilde - barphi_hat)
                x_t = 0
        else:
            self.beta_t = spla.norm(barphi_tilde - barphi_hat)
            x_t = 0

        indicator = 1 if tilde_a_t != hat_a_t else 0

        if (self.y < 1 and (diff_open_prob * (u - v)  + spla.norm(barphi_tilde-barphi_hat)*indicator*x_t) / q1  - self.y > 0):
            self.beta_t = max(self.beta_t, spla.norm(barphi_tilde-barphi_hat) * (1 - self.B + self.o_list))
            z_t = (diff_open_prob * (u - v)  + spla.norm(barphi_tilde-barphi_hat)*indicator*x_t) / q1  - self.y
            o_t = min((diff_open_prob * (u - v)  + spla.norm(barphi_tilde-barphi_hat)*indicator*x_t) / q1, self.B-self.o_list, 1)
            self.y = self.y*(1 + o_t/self.B) + o_t/((self.c - 1)*self.B)
        else:
            self.beta_t = spla.norm(barphi_tilde - barphi_hat)
            z_t = 0
            o_t = 0

        self.o_list += o_t
        return o_t

    # Online learning algorithm
    def pick_action(self, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, Y_t0, Y_t1, user_id, env_state, mu_max = None, beta_t = None):
        theta_tilde = np.matmul(spla.inv(self.V_tilde), self.b_tilde)  # update OLS estimates of theta
        theta_hat = np.matmul(spla.inv(self.V_hat), self.b_hat)

        upper_tilde = np.zeros(self.num_action + 1)
        upper_hat = np.zeros(self.num_action + 1)
        upper_tilde_st = np.zeros(self.num_action + 1)
        for a in range(self.num_action + 1):
            barphi_hat = self.get_observation_bar(a, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
            barphi_tilde = barphi_hat.copy()
            phi = self.get_observation(a, advantage_state_tilde, baseline_state_tilde)
            upper_tilde[a] = np.matmul(barphi_tilde.T, theta_tilde) + np.sqrt(
                np.matmul(np.matmul(barphi_tilde.T, spla.inv(self.V_tilde)), barphi_tilde))
            upper_hat[a] = np.matmul(barphi_hat.T, theta_hat) + np.sqrt(
                np.matmul(np.matmul(barphi_hat.T, spla.inv(self.V_hat)), barphi_hat))
            upper_tilde_st[a] = np.matmul(phi.T, theta_tilde) + np.sqrt(
                np.matmul(np.matmul(phi.T, spla.inv(self.V_tilde)), phi))

        A_tilde = np.argmax(upper_tilde)
        A_hat = np.argmax(upper_hat)
        A_tilde_st = np.argmax(upper_tilde_st)

        barphi_hat = self.get_observation_bar(A_hat, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
        phi_hat = self.get_observation(A_tilde_st, advantage_state_tilde, baseline_state_tilde)
        gamma_hat = np.matmul(barphi_hat.T, theta_hat)
        mu_hat = np.matmul(phi_hat.T, theta_tilde)

        delta = int(A_tilde_st == A_hat)
        distance = spla.norm(barphi_hat - phi)

        # compute expected rewards of the revealer
        barphi_A_tilde = self.get_observation_bar(A_tilde, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values) / (2* mu_max)
        phi_A_tilde_st = self.get_observation(A_tilde_st, advantage_state_tilde, baseline_state_tilde) / (2*mu_max)

        max_theta_tilde_st = cp.Variable(self.dim)
        cons1 = [
            cp.SOC(self.alpha_tilde, np.matmul(spla.sqrtm(self.V_tilde), (max_theta_tilde_st - theta_tilde.flatten())))]
        prob1 = cp.Problem(cp.Maximize(np.matmul(phi_A_tilde_st.T, max_theta_tilde_st)), cons1)
        u_st = prob1.solve(solver=cp.SCS)

        max_theta_tilde = cp.Variable(self.dim)
        cons2 = [
            cp.SOC(self.alpha_tilde, np.matmul(spla.sqrtm(self.V_tilde), (max_theta_tilde - theta_tilde.flatten())))]
        prob2 = cp.Problem(cp.Maximize(np.matmul(barphi_A_tilde.T, max_theta_tilde)), cons2)
        v = prob2.solve(solver=cp.SCS)
        # u0 = u_st
        # v0 = v
        # u_st = (u_st - 100)/50
        # v = (v - 100)/50

        # TODO: Need to pass user_id and current_state to use logistic model
        o_t = self.PrimalDual(u_st, v, A_tilde, A_hat, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, user_id, env_state, beta_t)
        # print("ot of pd2:", o_t)
        O_t = np.random.binomial(1, o_t)

        if (O_t == 1):
            if (Y_t1 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat
      
        else:
            if (Y_t0 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat

        return A_hat_star, O_t, o_t, mu_hat, gamma_hat, delta, distance

    def update_beta_t(self, beta_t, t):
        self.beta_t = (1 * beta_t) / (np.sqrt(t / 8 + 1) + 0.1)  # sqrt t


class DABBI_Nudging_I(DABBI_Nudging_II):
    def __init__(self, theta_std, lamb, dim, B, T, delta, num_action, c):
        DABBI_Nudging_II.__init__(self, theta_std, lamb, dim, B, T, delta, num_action, c)

        # Algorithm 3 Online learning algorithm
    def PrimalDual(self, u, v, user_id, env_state):
        q0, q1 = self.get_q0_q1(user_id, env_state)
        diff_open_prob = q1 - q0

        if (self.y < 1 and (u - v) * diff_open_prob - self.y > 0):
            o_t = min((u - v) * diff_open_prob, self.B - self.o_list, 1)
            self.y = self.y * (1 + o_t / self.B) + o_t / ((self.c - 1) * self.B)
            z_t = (u - v) * diff_open_prob - self.y
        else:
            z_t = 0
            o_t = 0

        self.o_list += o_t
        return o_t

    def pick_action(self, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, Y_t0, Y_t1, user_id=None, env_state=None, mu_max = None):
        theta_tilde = np.matmul(spla.inv(self.V_tilde), self.b_tilde)  # update OLS estimates of theta
        theta_hat = np.matmul(spla.inv(self.V_hat), self.b_hat)

        upper_tilde = np.zeros(self.num_action + 1)
        upper_hat = np.zeros(self.num_action + 1)
        upper_tilde_st = np.zeros(self.num_action + 1)
        for a in range(self.num_action + 1):
            barphi_hat = self.get_observation_bar(a, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
            barphi_tilde = barphi_hat.copy()
            phi = self.get_observation(a, advantage_state_tilde, baseline_state_tilde)
            upper_tilde[a] = np.matmul(barphi_tilde.T, theta_tilde) + np.sqrt(
                np.matmul(np.matmul(barphi_tilde.T, spla.inv(self.V_tilde)), barphi_tilde))
            upper_hat[a] = np.matmul(barphi_hat.T, theta_hat) + np.sqrt(
                np.matmul(np.matmul(barphi_hat.T, spla.inv(self.V_hat)), barphi_hat))
            upper_tilde_st[a] = np.matmul(phi.T, theta_tilde) + np.sqrt(
                np.matmul(np.matmul(phi.T, spla.inv(self.V_tilde)), phi))

        A_tilde = np.argmax(upper_tilde)
        A_hat = np.argmax(upper_hat)
        A_tilde_st = np.argmax(upper_tilde_st)

        barphi_hat = self.get_observation_bar(A_hat, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
        phi= self.get_observation(A_tilde_st, advantage_state_tilde, baseline_state_tilde)
        gamma_hat = np.matmul(barphi_hat.T, theta_hat)
        mu_hat = np.matmul(phi.T, theta_tilde)

        delta = int(A_tilde_st == A_hat)
        distance = spla.norm(barphi_hat - phi)

        # compute expected rewards of the revealer
        barphi_A_tilde = self.get_observation_bar(A_tilde, advantage_state_tilde, baseline_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values) / (2 * mu_max)
        phi_A_tilde_st = self.get_observation(A_tilde_st, advantage_state_tilde, baseline_state_tilde) / (2 * mu_max)

        max_theta_tilde_st = cp.Variable(self.dim)
        cons1 = [
            cp.SOC(self.alpha_tilde, np.matmul(spla.sqrtm(self.V_tilde), (max_theta_tilde_st - theta_tilde.flatten())))]
        prob1 = cp.Problem(cp.Maximize(np.matmul(phi_A_tilde_st.T, max_theta_tilde_st)), cons1)
        u_st = prob1.solve(solver=cp.SCS)

        max_theta_tilde = cp.Variable(self.dim)
        cons2 = [
            cp.SOC(self.alpha_tilde, np.matmul(spla.sqrtm(self.V_tilde), (max_theta_tilde - theta_tilde.flatten())))]
        prob2 = cp.Problem(cp.Maximize(np.matmul(barphi_A_tilde.T, max_theta_tilde)), cons2)
        v = prob2.solve(solver=cp.SCS)
        # u0 = u_st
        # v0 = v
        # u_st = (u_st - 100)/50
        # v = (v - 100)/50
        o_t = self.PrimalDual(u_st, v,  user_id, env_state)  # no second constraint
        # print("ot of pd1:", o_t)
        O_t = np.random.binomial(1, o_t)

        if (O_t == 1):
            if (Y_t1 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat
      
        else:
            if (Y_t0 == 1):
                A_hat_star = A_tilde_st
            else:
                A_hat_star = A_hat

        return A_hat_star, O_t, o_t, mu_hat, gamma_hat, delta, distance


class Omniscient:
    def __init__(self, dim, B, c, T, num_action, theta_true):
        self.dim = dim
        # self.action = action
        self.theta_true = theta_true
        # self.p = p
        self.B = B
        self.c = c
        self.T = T
        # self.K = K
        self.num_action = num_action
    
    def get_observation(self, a, advantage_state, baseline_state):
        # Handle different input shapes - convert to 1D arrays first
        bs = np.array(baseline_state).flatten()  # Always convert to 1D
        adv = np.array(advantage_state).flatten()  # Always convert to 1D

        baseline_dim = len(bs)
        adv_dim = len(adv)

        # One-hot block for action a: [0...adv...0] where block size = adv_dim and there are num_action blocks
        phi_adv = np.zeros(adv_dim * self.num_action)
        # Handle both scalar and vector inputs for action

        action_idx = int(a)
        if action_idx != 0:
            start = (action_idx -1) * adv_dim
            phi_adv[start:start + adv_dim] = adv[:adv_dim]

        # Final feature: [baseline; one-hot(a) ⊗ advantage]
        phi_a = np.concatenate([bs, phi_adv]).reshape(-1, 1)
        return phi_a
    
    def get_observation_bar(self, a, advantage_state, baseline_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values):
        bs = np.array(baseline_state).flatten()  # Always convert to 1D
        adv = np.array(advantage_state).flatten()  # Always convert to 1D

        baseline_dim = len(bs)
        adv_dim = len(adv)

        # change a_bar as a weighted sum of the discrete a_bar values
        a_bar = np.sum(prob_a_bar * a_bar_discrete_values)
        b_bar = np.sum(prob_b_bar * b_bar_discrete_values)
        bs[1] = b_bar
        bs[2] = a_bar
        adv[1] = b_bar
        adv[2] = a_bar

        phi_adv = np.zeros(adv_dim * self.num_action)
        action_idx = int(a)
        if action_idx != 0:
            start = (action_idx -1) * adv_dim
            phi_adv[start:start + adv_dim] = adv[:adv_dim]

        phi_a = np.concatenate([bs, phi_adv]).reshape(-1, 1)

        return phi_a

    def get_q0_q1(self, user_id=None, state=None):
        if user_id is not None and state is not None:
            return sim_env_v4.get_logistic_app_open_probs(user_id, state)
        else:
            return 0, 1
    
    def omniscient_reward(self, advantage_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values):
        mu_omni = []
        v_omni = []
        
        for a in range(self.num_action + 1):
            phi_omni = self.get_observation(a, advantage_state_tilde, advantage_state_tilde)
            barphi_omni = self.get_observation_bar(a, advantage_state_tilde, advantage_state_tilde, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
            mu_omni.append(np.matmul(phi_omni.T, self.theta_true)[0])
            v_omni.append(np.matmul(barphi_omni.T, self.theta_true)[0])

        # print(f"advantage_state_tilde: {np.array(phi_omni)}")
        # print(f"barphi_omni: {np.array(barphi_omni)}")

        mu_omni_star = np.max(mu_omni)
        v_omni_star = np.max(v_omni)
        
        return mu_omni_star, v_omni_star

    def sort_index(self, lst, rev=True):
        index = range(len(lst))
        s = sorted(index, reverse=rev, key=lambda i: lst[i])
        return s

    def omniscient_strategy(self, advantage_state_tilde_list,Y_t0_list, Y_t1_list, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values):
        mu_omni_star = [0]*self.T
        v_omni_star = [0]*self.T
        diff_times_uv = [0]*self.T
        q0, q1 = self.get_q0_q1()
        for t in range(self.T):
            mu_omni_star[t], v_omni_star[t] = self.omniscient_reward(advantage_state_tilde_list[t], prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values)
            diff_times_uv[t] = (q1 - q0) * (mu_omni_star[t] - v_omni_star[t])
        
        reward_omni_list = v_omni_star.copy() # create a list of reward_t

        index_ot1 = self.sort_index(diff_times_uv)[:self.B]  #index of the B largest mu_omni_star
              
        O_omni_list = np.zeros(self.T) # create a list of O_t
        #difference to be positive u-v
        for i in index_ot1:
            if (mu_omni_star[i] > v_omni_star[i]):
                O_omni_list[i] = 1
        
        for i in range(self.T):
            if (O_omni_list[i] == 1):
                reward_omni_list[i] = q1 * mu_omni_star[i] + (1 - q1) * v_omni_star[i]
                # mu_omni_star[i] if Y_t1_list[i] == 1 else v_omni_star[i]
            else:
                reward_omni_list[i] = q0 * mu_omni_star[i] + (1 - q0) * v_omni_star[i]
                # mu_omni_star[i] if Y_t0_list[i] == 1 else v_omni_star[i]


     
        return reward_omni_list, mu_omni_star, v_omni_star
