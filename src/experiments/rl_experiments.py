import numpy as np
import pandas as pd
import reward_definition
import experiment_global_vars
import numpy.linalg as npla
from datetime import datetime, timedelta


TRIAL_LENGTH_IN_WEEKS = experiment_global_vars.TRIAL_LENGTH_IN_WEEKS
NUM_DECISION_TIMES = experiment_global_vars.NUM_DECISION_TIMES
FILL_IN_COLS = experiment_global_vars.FILL_IN_COLS

# helpers for recruitment by date
def get_date(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d').date()

def increment_date(date_string):
    next_date = get_date(date_string) + timedelta(days=1)
    return next_date.strftime("%Y-%m-%d") 

def dt_to_user_day_in_study(dt):
    return dt // 2 + 1

def calculate_day_in_study(start_date_str, curr_date_str):
    curr_date =  get_date(curr_date_str)
    start_date =  get_date(start_date_str)

    return (curr_date - start_date).days + 1

def get_morning_decision_t(user_day_in_study):
    return (2 * user_day_in_study) - 2

def get_evening_decision_t(user_day_in_study):
    return (2 * user_day_in_study) - 1

# assumes a weekly recruitment rate
# def compute_num_updates(users_groups, update_cadence):
#     last_group_idx = max(users_groups[:,1].astype(int))
#     num_study_decision_times = (last_group_idx + TRIAL_LENGTH_IN_WEEKS) * 7 * 2
#     # we subtract 1 because we do not update after the final week of the study
#     num_updates = num_study_decision_times / update_cadence

#     return int(num_updates)

def create_dfs_no_pooling(users, update_cadence, rl_algorithm_feature_dim):
    N = len(users) 
    batch_data_size = N * NUM_DECISION_TIMES
    ### data df ###
    data_dict = {}
    data_dict['user_idx'] = np.repeat(range(N), NUM_DECISION_TIMES)
    data_dict['user_id'] = np.repeat(users, NUM_DECISION_TIMES)
    data_dict['user_decision_t'] = np.stack([range(NUM_DECISION_TIMES) for _ in range(N)], axis=0).flatten()
    data_dict['trial_day_in_study'] = np.stack([1 + (np.arange(NUM_DECISION_TIMES) // 2) for _ in range(N)], axis=0).flatten()
    data_dict['day_in_study'] = np.stack([1 + (np.arange(NUM_DECISION_TIMES) // 2) for _ in range(N)], axis=0).flatten()
    for key in FILL_IN_COLS:
        data_dict[key] = np.full(batch_data_size, np.nan)
    data_df = pd.DataFrame.from_dict(data_dict)
    return data_df

# def create_dfs_full_pooling(user_ids, user_envs, rl_algorithm_feature_dim):
#     N = len(user_ids)
#     batch_data_size = N * NUM_DECISION_TIMES
#     ### data df ###
#     data_dict = {}
#     data_dict['user_id'] = np.repeat(user_ids, NUM_DECISION_TIMES)
#     data_dict['user_idx'] = np.repeat(list(user_envs.keys()), NUM_DECISION_TIMES)
#     data_dict['user_decision_t'] = np.stack([range(NUM_DECISION_TIMES) for _ in range(N)], axis=0).flatten()
#     data_dict['user_day_in_study'] = np.vectorize(dt_to_user_day_in_study)(data_dict['user_decision_t'])
#     for key in FILL_IN_COLS:
#         data_dict[key] = np.full(batch_data_size, np.nan)
#     data_df = pd.DataFrame.from_dict(data_dict)
#     ### udpate df ###
#     update_dict = {}
#     num_updates = 38 # the number of posterior updates in the real trial
#     update_dict['update_t'] = np.arange(0, num_updates)
#     for i in range(rl_algorithm_feature_dim):
#         update_dict['posterior_mu.{}'.format(i)] = np.full(num_updates, np.nan)
#     for i in range(rl_algorithm_feature_dim):
#         for j in range(rl_algorithm_feature_dim):
#             update_dict['posterior_var.{}.{}'.format(i, j)] = np.full(num_updates, np.nan)
#     update_df = pd.DataFrame.from_dict(update_dict)

#     return data_df, update_df

# regex pattern '.*' gets you everything
# Note: if regex pattern only refers to one column, then you need to .flatten() the resulting array
def get_data_df_values_for_users(data_df, user_idxs, trial_day_in_study, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'].isin(user_idxs)) & (data_df['trial_day_in_study'] <= trial_day_in_study)].filter(regex=(regex_pattern)))

def get_user_data_values_from_decision_t(data_df, user_idx, decision_t, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'] == user_idx) & (data_df['user_decision_t'] < decision_t)].filter(regex=(regex_pattern)))

def set_data_df_values_for_user(data_df, user_idx, decision_time, trial_day_in_study, policy_idx, action, O, OP, regret, quality, alg_state, mu_hat, gamma_hat, delta, distance, q0, q1):
    data_df.loc[(data_df['user_idx'] == user_idx) & (data_df['user_decision_t'] == decision_time), FILL_IN_COLS] = np.concatenate([[trial_day_in_study, policy_idx, action, O, OP, regret, quality, mu_hat, gamma_hat, decision_time, delta, distance, q0, q1], alg_state])

### for full pooling experiments ###
def set_update_df_values(update_df, update_t, posterior_mu, posterior_var):
    update_df.iloc[update_df['update_t'] == update_t, 1:] = np.concatenate([posterior_mu, posterior_var.flatten()])

### for no pooling experiments ###
def set_update_df_values_for_user(update_df, user_idx, update_t, posterior_mu, posterior_var):
    update_df.iloc[(update_df['update_t'] == update_t) & (update_df['user_idx'] == user_idx), 3:] = np.concatenate([posterior_mu, posterior_var])

# if user did not open the app at all before the decision time, then we simulate
# the algorithm selecting action based off of a stale state (i.e., b_bar is the b_bar from when the user last opened their app)
# if user did open the app, then the algorithm selecting action based off of a fresh state (i.e., b_bar stays the same)
# def get_alg_state_from_app_opening(user_last_open_app_dt, data_df, user_idx, j, advantage_state):

#     # if morning dt we check if users opened the app in the morning
#     # if evening dt we check if users opened the app in the morning and in the evening
#     if j % 2 == 0:
#         user_opened_app_today = (user_last_open_app_dt == j)
#     else:
#         # we only simulate users opening the app for morning dts
#         user_opened_app_today = (user_last_open_app_dt == j - 1)
#     # if not user_opened_app_today:
#     #     # impute b_bar with stale b_bar and prior day app engagement = 0
#     #     stale_b_bar = get_user_data_values_from_decision_t(data_df, user_idx, user_last_open_app_dt + 1, 'state.b.bar').flatten()[-1]
#     #     # refer to rl_algorithm.py process_alg_state functions for V2, V3
#     #     advantage_state[1] = stale_b_bar
#     #     advantage_state[3] = 0

#     if not user_opened_app_today:
#         stale_history_b = get_user_data_values_from_decision_t(
#             data_df, user_idx, user_last_open_app_dt + 1, 'state.b.bar'
#         ).flatten()
#         stale_history_a = get_user_data_values_from_decision_t(
#             data_df, user_idx, user_last_open_app_dt + 1, 'state.a.bar'
#         ).flatten()
#         if stale_history_b.size > 0 and np.isfinite(stale_history_b[-1]):
#             advantage_state[1] = stale_history_b[-1]
#         else:
#             advantage_state[1] = 0.0
#         if stale_history_a.size > 0 and np.isfinite(stale_history_a[-1]):
#             advantage_state[2] = stale_history_a[-1]
#         else:
#             advantage_state[2] = 0.0
#         advantage_state[3] = 0.0

#     return advantage_state

# def get_last_open_dt(data_df, user_idx, current_j):
#     mask = (
#         (data_df['user_idx'] == user_idx) &
#         (data_df['user_decision_t'] < current_j) &
#         (data_df['OP'] == 1)          # whatever column now stores `open`
#     )
#     prev_opens = data_df.loc[mask, 'user_decision_t']
#     if prev_opens.empty:
#         return 0   # or another default
#     return int(prev_opens.iloc[-1])


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

def execute_decision_time(data_df, user_idx, j, trial_day_in_study, alg_candidate, sim_env, policy_idx, Y_t0, 
                          Y_t1, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, 
                          advantage_state, day_in_study=None, reward_omni=None, mu_omni_star=None, v_omni_star=None, mu_max=None, alg_name=None, beta_t=None):

    # simulate app opening issue
    # user_last_open_dt = get_last_open_dt(data_df, user_idx, j)
    # alg_state_hat = get_alg_state_from_app_opening(user_last_open_dt, data_df, user_idx, j, advantage_state.copy())

    ## ACTION SELECTION ##
    user_id = sim_env.get_users()[user_idx]
    if alg_name == "DABBI_Nudging_II":
        action, O, o_t, mu_hat, gamma_hat, delta, distance = alg_candidate.pick_action(advantage_state, advantage_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, Y_t0, Y_t1, user_id, advantage_state, mu_max, beta_t)
    else:
        action, O, o_t, mu_hat, gamma_hat, delta, distance = alg_candidate.pick_action(advantage_state, advantage_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, Y_t0, Y_t1, user_id, advantage_state, mu_max)
    q0, q1 = alg_candidate.get_q0_q1(user_id, advantage_state)
    mu_hat = mu_hat.item()
    gamma_hat = gamma_hat.item()

    open = Y_t1 if O == 1 else Y_t0

    expected_reward, expected_reward_no_reveal, quality = sim_env.generate_outcomes(user_idx, advantage_state, prob_a_bar, prob_b_bar, a_bar_discrete_values, b_bar_discrete_values, action, o_t, q0, q1)
    


    regret = reward_omni - expected_reward
    if alg_name == "UCB_NoReveal":
        regret = reward_omni - expected_reward_no_reveal
    quality = expected_reward

    # if regret < 0:
    #     print(f"expected_reward: {expected_reward}, reward_omni: {reward_omni}, regret: {regret}")

    a_bar = np.sum(prob_a_bar * a_bar_discrete_values)
    b_bar = np.sum(prob_b_bar * b_bar_discrete_values)
    alg_state_hat = np.array([advantage_state[0], b_bar, a_bar, advantage_state[3], advantage_state[4], advantage_state[5]]) #, advantage_state[6]])
    ## SAVE VALUES ##
    set_data_df_values_for_user(data_df, user_idx, j, trial_day_in_study, policy_idx, action, O, open, regret, quality, alg_state_hat, mu_hat, gamma_hat, delta, distance, q0, q1)

    return open

def run_experiment(alg_candidates, sim_env, alg_name, omniscient_dict, sampled_users):
    # env_users = sim_env.get_users()
    env_users = sampled_users
    # all alg_candidates have the same update cadence and feature dimension
    update_cadence = 2# alg_candidates[0].get_update_cadence()
    data_df = create_dfs_no_pooling(env_users, update_cadence, alg_candidates[0].get_feature_dim())
    policy_idxs = np.zeros(len(env_users))
    # Prior values are not tracked since update_df is removed
    # Pre-compute all states and Y values for all users and time steps
    advantage_state_tilde_list = np.zeros((len(env_users), NUM_DECISION_TIMES, 6)) #, 7))
    Y_t0_list = np.zeros((len(env_users), NUM_DECISION_TIMES))
    Y_t1_list = np.zeros((len(env_users), NUM_DECISION_TIMES))
    
    
    for user_idx in range(len(env_users)):
        # print("User {}: User ID {}".format(user_idx, env_users[user_idx]))
        for j in range(NUM_DECISION_TIMES):
            # print("User {}: Decision Time {}".format(user_idx, j))
            env_state = sim_env.generate_current_state(user_idx, j)
            Y_t0_list[user_idx, j], Y_t1_list[user_idx, j] = sim_env.get_user_y_t0_y_t1(user_idx, j, env_state)
            
            # Get the algorithm for this user
            alg_candidate = alg_candidates[user_idx]
            # process_alg_state returns (advantage_state, baseline_state)
            advantage_state, _ = alg_candidate.process_alg_state(env_state)
            

            advantage_state_tilde_list[user_idx, j] = advantage_state

    a_bar_discrete_values = np.array([0, 1, 2, 3, 4])
    # np.unique(advantage_state_tilde_list[:, :, 2])
    b_bar_discrete_values = np.array([0, 1, 2, 3, 4])
    # np.unique(advantage_state_tilde_list[:, :, 1])

    prob_a_bar_list = np.zeros((len(env_users), 5))
    prob_b_bar_list = np.zeros((len(env_users), 5))
    for user_idx in range(len(env_users)):
        prob_a_bar_list[user_idx] = np.bincount(advantage_state_tilde_list[user_idx, :, 2].astype(int), minlength=5) / NUM_DECISION_TIMES
        prob_b_bar_list[user_idx] = np.bincount(advantage_state_tilde_list[user_idx, :, 1].astype(int), minlength=5) / NUM_DECISION_TIMES

    for user_idx in range(len(env_users)):
        alg_candidate = alg_candidates[user_idx]
        omniscient = omniscient_dict[env_users[user_idx]]

        reward_omni_list, mu_omni_star_list, v_omni_star_list = omniscient.omniscient_strategy(advantage_state_tilde_list[user_idx], Y_t0_list[user_idx], Y_t1_list[user_idx],
                                                          prob_a_bar_list[user_idx], prob_b_bar_list[user_idx], 
                                                          a_bar_discrete_values, b_bar_discrete_values)
        mu_max = np.max(mu_omni_star_list)

        # get initial beta from data


        for j in range(NUM_DECISION_TIMES):
            day_in_study = 1 + (j // 2)
            # Generate state once and use it for both Y_t0/Y_t1 calculation and decision time execution
            advantage_state = advantage_state_tilde_list[user_idx, j]
            Y_t0 = Y_t0_list[user_idx, j]
            Y_t1 = Y_t1_list[user_idx, j]
            reward_omni = reward_omni_list[j]
            mu_omni_star = mu_omni_star_list[j]
            v_omni_star = v_omni_star_list[j]
            advantage_state_temp = advantage_state_tilde_list[user_idx, j]
            prob_a_bar_temp = prob_a_bar_list[user_idx]
            prob_b_bar_temp = prob_b_bar_list[user_idx]
            beta_t_list = []
            for a in range(4):
                for b in range(a+1, 5): 
                    if (a != b):
                        beta_t_list.append(npla.norm(omniscient.get_observation_bar(a, advantage_state_temp, advantage_state_temp, prob_a_bar_temp, prob_b_bar_temp, a_bar_discrete_values, b_bar_discrete_values)-omniscient.get_observation_bar(b, advantage_state_temp, advantage_state_temp, prob_a_bar_temp, prob_b_bar_temp, a_bar_discrete_values, b_bar_discrete_values)))
            # beta_t = np.min(beta_t_list)  #initial value
            beta_t = 4
            # print("beta_t: ", beta_t)
            beta_t_new = (1 * beta_t) / (np.sqrt(j / 2 + 1) + 0.1)
            open = execute_decision_time(data_df, user_idx, j, dt_to_user_day_in_study(j), alg_candidate, sim_env, 
                                         policy_idxs[user_idx], Y_t0, Y_t1, prob_a_bar_list[user_idx], 
                                         prob_b_bar_list[user_idx], a_bar_discrete_values, b_bar_discrete_values, 
                                         advantage_state, day_in_study, reward_omni, mu_omni_star, v_omni_star, mu_max, alg_name, beta_t_new)
            
            # each user's first week is pure exploration using the prior
            # note: we only update if the algorithm is online, or else the prior is used for the whole trial
            if ((j >= 0)):
                # j % update_cadence == (update_cadence - 1) and
                # Get data for exactly decision time j
                # Get the state columns using the correct column names
                state_cols = ['state.tod', 'state.b.bar', 'state.a.bar', 'state.app.engage', 'state.day.type', 'state.bias'] #, 'state.day.in.study']
                current_state = data_df.loc[(data_df['user_idx'] == user_idx) & (data_df['user_decision_t'] == j)][state_cols].values
                current_action = data_df.loc[(data_df['user_idx'] == user_idx) & (data_df['user_decision_t'] == j)]['action'].values
                current_reward = data_df.loc[(data_df['user_idx'] == user_idx) & (data_df['user_decision_t'] == j)]['quality'].values
                
                alg_candidate.update_revealer_parameter(current_reward, current_state, current_state, current_action)
                # if alg_name == "DABBI_Nudging_II":
                    # alg_candidate.update_beta_t(alg_candidate.beta_t, j)
                if open == 1:
                   alg_candidate.update_recommender_parameter()

                            
                policy_idxs[user_idx] += 1
                update_idx = int(policy_idxs[user_idx])
                # print("Update Time {} for {}".format(update_idx, user_idx))
                # update_df tracking removed

    return data_df

# either gets all users with that start date or end date
# type needs to be either "start" or "end"
# def get_users_for_date(user_envs, date, type):
#     users = []
#     for user_idx, user_env in user_envs.items():
#         if type == "start" and user_env.get_start_date() == date:
#             users.append(user_idx)
#         elif type == "end" and user_env.get_end_date() == date:
#             users.append(user_idx)

#     return users

### runs experiment with full pooling and incremental recruitment
# def run_incremental_recruitment_exp(alg_candidate, sim_env):
#     # instantiating dataframes
#     env_users = sim_env.get_users()
#     user_envs = sim_env.get_user_envs()
#     data_df, update_df = create_dfs_full_pooling(env_users, user_envs, alg_candidate.get_feature_dim())
#     # add in prior values to posterior dataframe
#     update_idx = 0
#     set_update_df_values(update_df, update_idx, alg_candidate.posterior_mean, alg_candidate.posterior_var)
#     current_date_str, trial_end_date_str = sim_env.get_trial_start_end_dates()
#     trial_day_in_study = 1
#     # get current users
#     current_user_idxs = get_users_for_date(user_envs, current_date_str, "start")
#     while current_date_str != trial_end_date_str:
#         # check if it's update time
#         if ((get_date(current_date_str) in alg_candidate.get_update_dates()) and alg_candidate.check_is_online()):
#             ### UPDATE TIME ###
#             alg_states = get_data_df_values_for_users(data_df, current_user_idxs, trial_day_in_study, 'state.*')
#             actions = get_data_df_values_for_users(data_df, current_user_idxs, trial_day_in_study, 'action').flatten()
#             pis = get_data_df_values_for_users(data_df, current_user_idxs, trial_day_in_study, 'prob').flatten()
#             rewards = get_data_df_values_for_users(data_df, current_user_idxs, trial_day_in_study, 'reward').flatten()
#             alg_candidate.update(alg_states, actions, pis, rewards)
#             update_idx += 1
#             print(f"Update Time: {update_idx}")
#             set_update_df_values(update_df, update_idx, alg_candidate.posterior_mean, alg_candidate.posterior_var)
#         # execute morning and evening decision times for the current day
#         for user_idx in current_user_idxs:
#             user_start_date_str = user_envs[user_idx].get_start_date()
#             user_day_in_study = calculate_day_in_study(user_start_date_str, current_date_str)
#             morning_dt = get_morning_decision_t(user_day_in_study)
#             evening_dt = get_evening_decision_t(user_day_in_study)
#             execute_decision_time(data_df, user_idx, morning_dt, trial_day_in_study, alg_candidate, sim_env, update_idx)
#             execute_decision_time(data_df, user_idx, evening_dt, trial_day_in_study, alg_candidate, sim_env, update_idx)

#         # increment day
#         current_date_str = increment_date(current_date_str)
#         trial_day_in_study += 1
#         # add users by start date
#         current_user_idxs += get_users_for_date(user_envs, current_date_str, "start")
#         # remove users if they have finished the trial
#         finished_users = get_users_for_date(user_envs, current_date_str, "end")
#         current_user_idxs = [user for user in current_user_idxs if user not in finished_users]

#     return data_df, update_df

