# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats import bernoulli
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import itertools

import read_write_info
import simulation_environment
import reward_definition
import os

"""## BASE ENVIRONMENT COMPONENT
---
"""

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
SIM_ENV_DATA_DIR = os.path.join(project_root, 'sim_env_data')
NON_STAT_PARAMS_DF = pd.read_csv(os.path.join(SIM_ENV_DATA_DIR, 'v4_non_stat_zip_model_params.csv'))
APP_OPEN_PROB_DF = pd.read_csv(os.path.join(SIM_ENV_DATA_DIR, 'v4_app_open_prob.csv'))

APP_OPEN_LOGIT_PARAMS_DF = pd.read_csv(os.path.join(SIM_ENV_DATA_DIR, 'V4_app_opening_logit_params.csv'))
SIM_ENV_USERS = np.array(NON_STAT_PARAMS_DF['User'])
# getting participant start and end dates
START_END_DATES_DF = pd.read_csv(os.path.join(SIM_ENV_DATA_DIR, 'v4_start_end_dates.csv'))
# values used by run_experiments
NUM_TRIAL_USERS = len(SIM_ENV_USERS)
TRIAL_START_DATE = "2023-09-22"
TRIAL_END_DATE = "2024-07-16"

"""### Generate States
---
"""

def get_previous_day_qualities_and_actions(j, Qs, As):
    if j > 1:
        if j % 2 == 0:
            return Qs, As
        else:
            # current evening dt does not use most recent quality or action
            return Qs[:-1], As[:-1]
    # first day return empty Qs and As back
    else:
        return Qs, As

# Stationary State Space
# 0 - time of day
# 1 - b_bar (normalized)
# 2 - a_bar (normalized)
# 3 - app engagement
# 4 - weekday vs. weekend
# 5 - bias

# Non-stationary state space
# 0 - time of day
# 1 - b_bar (normalized)
# 2 - a_bar (normalized)
# 3 - app engagement
# 4 - weekday vs. weekend
# 5 - bias
# 6 - day in study

# Note: We assume that all participants start the study on Monday (j = 0 denotes)
# Monday morning. Therefore the first weekend idx is j = 10 (Saturday morning)
def generate_env_state(j, user_qualities, user_actions, app_engagement):
    env_state = np.ones(7)
    # session type - either 0 or 1
    session_type = j % 2
    env_state[0] = session_type
    # b_bar, a_bar (normalized)
    Qs, As = get_previous_day_qualities_and_actions(j, user_qualities, user_actions)
    b_bar, a_bar = reward_definition.get_b_bar_a_bar(Qs, As)
    env_state[1] = reward_definition.normalize_b_bar(b_bar)
    env_state[2] = reward_definition.normalize_a_bar(a_bar)
    # app engagement
    env_state[3] = app_engagement
    # weekday vs. weekend
    env_state[4] = 1 if (j % 14 >= 10 and j % 14 <= 13) else 0
    # bias
    env_state[5] = 1
    # day in study if a non-stationary environment
    env_state[6] = simulation_environment.normalize_day_in_study(1 + (j // 2))

    return env_state

def get_app_open_prob(user_id):
    return APP_OPEN_PROB_DF[APP_OPEN_PROB_DF['user_id'] == user_id]['app_open_prob'].values[0]

def get_logistic_app_open_probs(user_id, state):
    """
    Calculate app opening probabilities using logistic model.
    Returns Y_t0 (no nudge) and Y_t1 (with nudge) probabilities.
    
    Args:
        user_id: user identifier
        state: environment state vector [tod, b_bar, a_bar, app_engage, day_type, bias, day_in_study]
    
    Returns:
        (Y_t0, Y_t1): tuple of probabilities for no nudge and nudge conditions
    """
    params = APP_OPEN_LOGIT_PARAMS_DF[APP_OPEN_LOGIT_PARAMS_DF['user_id'] == user_id]
    params = params.iloc[0]
    
    # Extract relevant state features (matching the logistic model columns)
    # State vector: [tod, b_bar, a_bar, app_engage, day_type, bias, day_in_study]
    tod = float(state[0]) if len(state) > 0 else 0.0
    b_bar = float(state[1]) if len(state) > 1 else 0.0
    a_bar = float(state[2]) if len(state) > 2 else 0.0
    bias = float(state[5]) if len(state) > 5 else 1.0
    # CSV uses 'state_modif' which corresponds to day_in_study (normalized)
    modif = float(state[6]) if len(state) > 6 else 0.0
    
    # Y_t0: No nudge (nudge = 0)
    logit_0 = (float(params['intercept']) + 
               float(params['state_tod']) * tod +
               float(params['state_b_bar']) * b_bar + 
               float(params['state_a_bar']) * a_bar +
               float(params['state_bias']) * bias +
               float(params['state_modif']) * modif)
    
    q0 = 1 / (1 + np.exp(-logit_0))
    
    # Calculate nudge effect components
    nudge_effect = (float(params['nudge']) +  # Main nudge effect
                   float(params['nudge_x_state_tod']) * tod +     # Nudge interactions
                   float(params['nudge_x_state_b_bar']) * b_bar +
                   float(params['nudge_x_state_a_bar']) * a_bar +
                   float(params['nudge_x_state_bias']) * bias +
                   float(params['nudge_x_state_modif']) * modif)
    
    # Y_t1: With nudge (nudge = 1)  
    logit_1 = logit_0 + nudge_effect
    q1 = 1 / (1 + np.exp(-logit_1))
    
    # Ensure Y_t1 >= Y_t0 (nudging should increase app opening probability)
    if q1 < q0:
        # Weighted combination: Y_t1_new = (1-Y_t0)*Y_t1 + Y_t0
        # This automatically ensures Y_t1_new >= Y_t0 and Y_t1_new <= 1
        q1 = (1 - q0) * q1 + q0
    
    return q0, q1

def get_user_start_date(user_id):
    return START_END_DATES_DF[START_END_DATES_DF['user_id'] == user_id]['user_start_day'].values[0]

def get_user_end_date(user_id):
    return START_END_DATES_DF[START_END_DATES_DF['user_id'] == user_id]['user_end_day'].values[0]

BASE_BERN_NAMES = [
    'state_tod.Base.Bern',
    'state_b_bar.norm.Base.Bern', 'state_a_bar.norm.Base.Bern',
    'state_app_engage.Base.Bern', 'state_day_type.Base.Bern',
    'state_bias.Base.Bern', 'state_day_in_study.Base.Bern'
]
BASE_POISSON_NAMES = [
    'state_tod.Base.Poisson',
    'state_b_bar.norm.Base.Poisson', 'state_a_bar.norm.Base.Poisson',
    'state_app_engage.Base.Poisson', 'state_day_type.Base.Poisson',
    'state_bias.Base.Poisson', 'state_day_in_study.Base.Poisson'
]

# note: since v4 only chose zip models, these are the following parameters
def get_base_params_for_user(user):
  bern_base = np.array(NON_STAT_PARAMS_DF[NON_STAT_PARAMS_DF['User'] == user][BASE_BERN_NAMES]).reshape(-1,)
  poisson_base = np.array(NON_STAT_PARAMS_DF[NON_STAT_PARAMS_DF['User'] == user][BASE_POISSON_NAMES]).reshape(-1,)

  # poisson parameters, bernouilli parameters
  return bern_base, poisson_base

# Multi-category action types: 0=control, 1=RP, 2=SR, 3=QA, 4=FB
ACTION_TYPES = ['Base', 'RP', 'SR', 'QA', 'FB']

def get_adv_params_for_user(user):
    """Get advantage parameters for all action types (5 actions x 7 state features each)"""
    bern_adv_all = []
    poisson_adv_all = []
    
    for action_type in ACTION_TYPES:
        bern_names = [
            f'state_tod.{action_type}.Bern',
            f'state_b_bar.norm.{action_type}.Bern', f'state_a_bar.norm.{action_type}.Bern',
            f'state_app_engage.{action_type}.Bern', f'state_day_type.{action_type}.Bern',
            f'state_bias.{action_type}.Bern', f'state_day_in_study.{action_type}.Bern'
        ]
        poisson_names = [
            f'state_tod.{action_type}.Poisson',
            f'state_b_bar.norm.{action_type}.Poisson', f'state_a_bar.norm.{action_type}.Poisson',
            f'state_app_engage.{action_type}.Poisson', f'state_day_type.{action_type}.Poisson',
            f'state_bias.{action_type}.Poisson', f'state_day_in_study.{action_type}.Poisson'
        ]
        
        bern_params = np.array(NON_STAT_PARAMS_DF[NON_STAT_PARAMS_DF['User'] == user][bern_names]).reshape(-1,)
        poisson_params = np.array(NON_STAT_PARAMS_DF[NON_STAT_PARAMS_DF['User'] == user][poisson_names]).reshape(-1,)
        
        bern_adv_all.append(bern_params)
        poisson_adv_all.append(poisson_params)
    
    # Return shape: (5, 7) for 5 actions x 7 state features
    return np.array(bern_adv_all), np.array(poisson_adv_all)

def get_user_effect_funcs():
    # negative treatment effect means users are more likely to brush
    # adv_params shape: (5, 7) for 5 actions x 7 state features
    def bern_adv_func(state, action, adv_params):
        action_params = adv_params[int(action)]  # Select params for this action
        return min(action_params @ state, 0)
    
    # positive treatment effect means users brush more seconds
    def y_adv_func(state, action, adv_params):
        action_params = adv_params[int(action)]  # Select params for this action
        return max(action_params @ state, 0)

    return bern_adv_func, y_adv_func

"""## Creating Simulation Environment Objects
---
"""

class UserEnvironmentV4(simulation_environment.UserEnvironmentAppEngagement):
    def __init__(self, user_id, model_type, adv_params, \
                user_params, user_effect_func_bern, user_effect_func_y):
        # Note: in the base UserEnvironment, it uses simulated user_effect_sizes,
        # but we replace it with adv_params, user's fitted advantage parameters
        super(UserEnvironmentV4, self).__init__(user_id, model_type, adv_params, \
                            user_params, user_effect_func_bern, user_effect_func_y)
        # Store user_id for logistic app opening model
        self.user_id = user_id
        # for incremental recruitment, we use the actual start date from the Oralytics MRT
        self.start_date = get_user_start_date(user_id)
        self.end_date = get_user_end_date(user_id)

    def get_start_date(self):
        return self.start_date
    
    def get_end_date(self):
        return self.end_date
    
    def get_q0_q1(self, current_state):
        """
        Get Y_t0 and Y_t1 app opening probabilities using logistic model.
        """
        return get_logistic_app_open_probs(self.user_id, current_state)
    
    def generate_app_engagement_y_t0_y_t1(self, current_state):
        """Get both Y_t0 and Y_t1 for the new experiment flow using logistic model."""
        q0, q1 = self.get_q0_q1(current_state)
        Y_t0 = bernoulli.rvs(q0)
        Y_t1 = bernoulli.rvs(q1)
        return Y_t0, Y_t1

# def create_user_envs(users_list):
#     all_user_envs = {}
#     for i, user_id in enumerate(users_list):
#       model_type = "zip" # note: all users in V4 have the zero-inflated poisson model
#       base_params = get_base_params_for_user(user_id)
#       adv_params = get_adv_params_for_user(user_id)
#       user_effect_func_bern, user_effect_func_y = get_user_effect_funcs()
#       new_user = UserEnvironmentV4(user_id, model_type, adv_params, \
#                         base_params, user_effect_func_bern, user_effect_func_y)
#       all_user_envs[i] = new_user

#     return all_user_envs

### HELPERS FOR CREATING ENV. VARIANT WITH NO ADVANTANGE IN A PARTICULAR STATE ###
# state interpolation
dimension1 = [0, 1] # time of day
dimension2 = [-0.75, 0, 0.75] # b_bar
dimension3 = [-0.75, 0, 0.75] # a_bar
dimension4 = [0, 1] # prior day app engagement
dimension5 = [0, 1] # weekday vs. weekend
dimension6 = [1] # intercept (always =1)
dimension7 = [-0.75, 0, 0.75] # day in study

# Create a list of lists, where each sublist represents possible values for each dimension
dimensions = [dimension1, dimension2, dimension3, dimension4, dimension5, dimension6, dimension7]
# Generate all possible combinations using itertools.product
combinations = list(itertools.product(*dimensions))

# Define the constraint function: P should generate a w that is in the null space of x
def constraint(P_flat, u, x):
    # Reshape P from flat array
    P = P_flat.reshape((len(x), -1))
    w = P @ u
    return np.dot(w, x)

# We want the projection w to be close to u
def objective_proxy(P_flat, u , x):
    P = P_flat.reshape((len(x), -1))
    w = P @ u
    vals_with_w = combinations @ w
    vals_with_u = combinations @ u
    
    return mean_squared_error(vals_with_u, vals_with_w)

def sample_P_matrix(u, x):
    # Define initial guess for matrix P (random initialization)
    initial_P = np.random.rand(len(x), len(u))
    initial_P_flat = initial_P.flatten()
    # Define bounds for optimization (if any)
    bounds = [(None, None)] * len(initial_P_flat)
    constraints = {'type': 'eq', 'fun': lambda P_flat: constraint(P_flat, u, x)}
    objective = lambda P_flat: objective_proxy(P_flat, u, x)
    # Optimize the matrix P
    result = minimize(objective, initial_P_flat, constraints=constraints, bounds=bounds)
    # Reshape the result back to matrix P
    optimized_P = result.x.reshape((len(x), -1))

    return optimized_P

# ordering for env. features are:
# 0 - time of day
# 1 - b_bar (normalized)
# 2 - a_bar (normalized)
# 3 - app engagement
# 4 - weekday vs. weekend
# 5 - bias
# 6 - day in study
def turn_alg_state_to_env_state(state):
    x_4_mean = 2/7
    x_6_mean = 0
    env_state = np.array([state[0], state[1], state[2], state[3], x_4_mean, 1, x_6_mean])

    return env_state

# modifies the advantage parameters depending on the state of interest specified
# if state is None then we make no modifications and the advantage parameters are the one fitted using the MRT data
def create_user_envs_under_no_advantage(users_list, state=None, no_adv_all_states=False):
    all_user_envs = {}
    for i, user_id in enumerate(users_list):
      model_type = "zip" # note: all users in V4 have the zero-inflated poisson model
      base_params = get_base_params_for_user(user_id)
      adv_params = get_adv_params_for_user(user_id)
      # if we want to simulate a variant where there's no advantage in any state, then we zero out all advantage parameters
      if no_adv_all_states:
          adv_params = np.zeros(7), np.zeros(7)
      else:
          if state is not None:
            # we project the original advantage parameters such that the resulting parameters are in the 
            # null space of the covariates specified by state
            # this is to simulate a variant where there is no advantage in this particular state
            env_state = turn_alg_state_to_env_state(state)
            u_bern, u_poisson = adv_params
            sampled_P_bern = sample_P_matrix(u_bern, env_state)
            sampled_P_poisson = sample_P_matrix(u_poisson, env_state)
            w_bern = sampled_P_bern @ u_bern
            w_poisson = sampled_P_poisson @ u_poisson
            adv_params = w_bern, w_poisson
      user_effect_func_bern, user_effect_func_y = get_user_effect_funcs()
      new_user = UserEnvironmentV4(user_id, model_type, adv_params, \
                        base_params, user_effect_func_bern, user_effect_func_y)
      all_user_envs[i] = new_user

    return all_user_envs

# Environment used to re-evaluate algorithm design decisions
class SimulationEnvironmentV4(simulation_environment.SimulationEnvironmentAppEngagement):
    def __init__(self, state=None):
        # note: v4 uses the exact 79 participants from the Oralytics MRT
        user_envs = create_user_envs_under_no_advantage(SIM_ENV_USERS, state)

        super(SimulationEnvironmentV4, self).__init__(SIM_ENV_USERS, user_envs)

        self.version = "V4"
        self.trial_start_date = TRIAL_START_DATE
        self.trial_end_date = TRIAL_END_DATE

    def get_trial_start_end_dates(self):
        return self.trial_start_date, self.trial_end_date

    def generate_current_state(self, user_idx, j):
        # prior day app_engagement is 0 for the first day
        prior_app_engagement = self.get_user_prior_day_app_engagement(user_idx)
        self.simulate_app_opening_behavior(user_idx, j)
        brushing_qualities = np.array(self.get_env_history(user_idx, "outcomes"))
        past_actions = np.array(self.get_env_history(user_idx, "actions"))

        return generate_env_state(j, brushing_qualities, past_actions, prior_app_engagement)
        
    def simulate_app_opening_behavior(self, user_idx, j):
        """Override parent method to avoid calling generate_app_engagement without state."""
        # we simulate that we only know if users opened their app in the morning
        if j % 2 == 0:
            # For morning decision times, we don't simulate app opening here
            # The actual app opening simulation happens in get_user_y_t0_y_t1
            pass
        # we do not save that current day's app engagement until after the evening dt
        else:
            current_app_engagement = int(self.get_user_last_open_app_dt(user_idx) == j - 1)
            self.set_user_prior_day_app_engagement(user_idx, current_app_engagement)

    def get_user_y_t0_y_t1(self, user_idx, j, env_state=None):
        """
        Get Y_t0 and Y_t1 binary app opening outcomes for a user at decision time j.
        
        Args:
            user_idx: user index in the environment
            j: decision time
            env_state: optional pre-computed environment state
            
        Returns:
            (Y_t0, Y_t1): tuple of binary app opening outcomes (no nudge, nudge) as 0 or 1
        """
        if env_state is None:
            # Fallback: generate state if not provided (for backward compatibility)
            prior_app_engagement = self.get_user_prior_day_app_engagement(user_idx)
            brushing_qualities = np.array(self.get_env_history(user_idx, "outcomes"))
            past_actions = np.array(self.get_env_history(user_idx, "actions"))
            env_state = generate_env_state(j, brushing_qualities, past_actions, prior_app_engagement)
        
        # Get binary outcomes from the user environment using the provided state
        return self.all_user_envs[user_idx].generate_app_engagement_y_t0_y_t1(env_state)
