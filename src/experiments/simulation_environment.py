# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.stats import norm

class SimulationEnvironment():
    def __init__(self, users_list, user_envs):
        # must be implemented by children
        self.version = None
        # List: users in the environment (can repeat)
        self.users_list = users_list
        # Dict: key: int trial_user_idx, val: user environment object
        self.all_user_envs = user_envs

    def get_version(self):
        return self.version

    # this method needs to be implemented by all children
    def generate_current_state(self):
        return None

    def generate_outcomes(self, user_idx, advantage_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, action, O, q0, q1):
        return self.all_user_envs[user_idx].generate_outcome(advantage_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, action, O, q0, q1)

    def get_states_for_user(self, user_idx):
        return self.all_user_envs[user_idx].get_states()

    def get_users(self):
        return self.users_list
    
    def get_user_envs(self):
        return self.all_user_envs

    def get_env_history(self, user_idx, property):
        return self.all_user_envs[user_idx].get_user_history(property)

    def update_responsiveness(self, user_idx, a1_cond, a2_cond, b_cond, j):
        self.all_user_envs[user_idx].update_responsiveness(a1_cond, a2_cond, b_cond, j)

class SimulationEnvironmentAppEngagement(SimulationEnvironment):
    def __init__(self, users_list, user_envs):
        super(SimulationEnvironmentAppEngagement, self).__init__(users_list, user_envs)

    def generate_app_engagement(self, user_idx):
        return self.all_user_envs[user_idx].generate_app_engagement()

    def get_user_prior_day_app_engagement(self, user_idx):
        return self.all_user_envs[user_idx].get_prior_day_app_engagement()

    def set_user_prior_day_app_engagement(self, user_idx, prior_app_engagement):
        self.all_user_envs[user_idx].set_prior_day_app_engagement(prior_app_engagement)

    def get_user_last_open_app_dt(self, user_idx):
        return self.all_user_envs[user_idx].get_last_open_app_dt()

    def set_user_last_open_app_dt(self, user_idx, j):
        self.all_user_envs[user_idx].set_last_open_app_dt(j)

    def simulate_app_opening_behavior(self, user_idx, j):
        # we simulate that we only know if users opened their app in the morning
        if j % 2 == 0:
            # simulate whether or not the user opened their app
            current_app_engagement = self.generate_app_engagement(user_idx)
            if current_app_engagement:
                self.set_user_last_open_app_dt(user_idx, j)
        # we do not save that current day's app engagement until after the evening dt
        else:
            current_app_engagement = int(self.get_user_last_open_app_dt(user_idx) == j - 1)
            self.set_user_prior_day_app_engagement(user_idx, current_app_engagement)


# ### NORMALIZTIONS ###
# def normalize_total_brush_quality(quality):
#     return (quality - 154) / 163
    # return (quality - np.mean(quality)) / np.std(quality)

def normalize_day_in_study(day):
    return (day - 35.5) / 34.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""### Functions for Environment Models
---
"""

def construct_model_and_sample(advantage_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, action, o, \
                                q0, q1, \
                                      bern_params, \
                                    #   y_params, \
                                      effect_func_bern=lambda state, action: 0
                                      ):
    a_bar = np.sum(prob_a_bar * a_bar_discrete_values)
    b_bar = np.sum(prob_b_bar * b_bar_discrete_values)
    new_advantage_state = np.array([advantage_state[0], b_bar, a_bar, advantage_state[3], advantage_state[4], advantage_state[5]]) #, advantage_state[6]])

    # if open == 1:
    #     state_use = advantage_state
    # else:
    #     state_use = new_advantage_state
    
    # print(f"advantage_state: {np.array(advantage_state)}")
    # print(f"new_advantage_state: {np.array(new_advantage_state)}")

    mu_hat = bern_params @ advantage_state
    gamma_hat = bern_params @ new_advantage_state
        
    if action != 0:
        mu_hat += effect_func_bern(advantage_state, action)
        gamma_hat += effect_func_bern(new_advantage_state, action)

    h = o * q1 + (1 - o) * q0
    expected_reward = h * mu_hat + (1 - h) * gamma_hat
    # if open == 1:
    #     expected_reward = mu_hat
    # else:
    #     expected_reward = gamma_hat

    expected_reward_no_reveal = gamma_hat

    # bern_linear_comp_sample = bern_params @ advantage_state
    # if action != 0:
        # bern_linear_comp_sample += effect_func_bern(advantage_state, action)
    sample = norm.rvs(loc=expected_reward, scale=1)
    return expected_reward, expected_reward_no_reveal, sample


class UserEnvironment():
    def __init__(self, user_id, model_type, user_sessions, user_effect_sizes, \
                user_params, 
                user_effect_func_bern):
        self.user_id = user_id
        self.model_type = model_type
        # vector: size (T, D) where D is the dimension of the env. state
        # T is the length of the study
        self.user_states = user_sessions
        self.user_effect_sizes = user_effect_sizes
        # reward generating function
        self.user_params = user_params
        self.user_effect_func_bern = user_effect_func_bern

        self.reward_generating_func = lambda advantage_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, action, O, q0, q1: construct_model_and_sample(advantage_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, action, O, q0, q1, \
                                      self.user_params, \
                                    #   self.user_params[1], \
                                      effect_func_bern=lambda state, action: self.user_effect_func_bern(state, action, self.user_effect_sizes))
        # user environment history
        self.user_history = {"actions":[], "outcomes":[]}

    def get_user_history(self, property):
        return self.user_history[property]

    def set_user_history(self, property, value):
        self.user_history[property].append(value)

    def generate_outcome(self, advantage_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, action, O, q0, q1):
        # save action and outcome
        self.set_user_history("actions", action)
        expected_reward, expected_reward_no_reveal, outcome = self.reward_generating_func(advantage_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, action, O, q0, q1)
        self.set_user_history("outcomes", outcome)

        return expected_reward, expected_reward_no_reveal, outcome

    def get_states(self):
        return self.user_states

    def get_user_effect_sizes(self):
        return self.user_effect_sizes

class UserEnvironmentAppEngagement(UserEnvironment):
    def __init__(self, user_id, model_type, user_effect_sizes, \
                user_params,   
                user_effect_func_bern):
        super(UserEnvironmentAppEngagement, self).__init__(user_id, model_type, None, user_effect_sizes, \
                  user_params, 
                  user_effect_func_bern)
        # probability of opening app, needs to be implemented by children
        self.app_open_base_prob = None
        # tracking prior day app engagement
        self.prior_day_app_engagement = 0
        # last dt user open app, init: all users are assumed to open their app on the first day
        self.last_open_app_dt = 0

    def generate_app_engagement(self):
        return bernoulli.rvs(self.app_open_base_prob)

    def get_prior_day_app_engagement(self):
        return self.prior_day_app_engagement

    def set_prior_day_app_engagement(self, prior_day_app_engage):
        self.prior_day_app_engagement = prior_day_app_engage

    def get_last_open_app_dt(self):
        return self.last_open_app_dt

    def set_last_open_app_dt(self, j):
        self.last_open_app_dt = j
