import rl_experiments
import rl_algorithm
import rl_algorithms_budget
import sim_env_v4
import experiment_global_vars
import compute_metrics

import pickle
import numpy as np
import pandas as pd
import copy

### experiment parameters ###
'''
### Algorithm Candidates
* 'cluster_size': 'number of users within a cluster'
* 'offline_or_online': 'whether the algorithm updates online (updates posterior) or just uses the prior that was designed offline'
'''

MAX_SEED_VAL = experiment_global_vars.MAX_SEED_VAL
NUM_TRIAL_USERS = sim_env_v4.NUM_TRIAL_USERS

def get_cluster_size(pooling_type):
    if pooling_type == "full_pooling":
        return NUM_TRIAL_USERS
    elif pooling_type == "no_pooling":
        return 1
    else:
        print("ERROR: NO CLUSTER_SIZE FOUND - ", pooling_type)

def create_budget_algorithms(sampled_users):
    """
    Create the 5 budget constraint RL algorithms with default parameters.
    """
    # Algorithm parameters (you may need to adjust these)
    theta_std = 1.0
    lamb = 0.1
    B = 10   # Budget
    # beta_t = 20
    T = 140   # Time horizon
    delta = 1/T
    num_action = 4  # 4 action types (not 5!)
    c = (1 + 1/B) ** B
    dim = 30  # Feature dimension: 5 (baseline) + 20 (one-hot advantage: 4 actions × 5 dims) = 25
    
    algorithms = [
        ("UCB_NoNudge", rl_algorithms_budget.UCB_NoNudge(theta_std, lamb, dim, B, T, delta, num_action)),
        ("UCB_UniformNudge", rl_algorithms_budget.UCB_UniformNudge(theta_std, lamb, dim, B, T, delta, num_action, c)),
        ("UCB_NoReveal", rl_algorithms_budget.UCB_NoReveal(theta_std, lamb, dim, B, T, delta, num_action)),
        ("DABBI_Nudging_I", rl_algorithms_budget.DABBI_Nudging_I(theta_std, lamb, dim, B, T, delta, num_action, c)),
        ("DABBI_Nudging_II", rl_algorithms_budget.DABBI_Nudging_II(theta_std, lamb, dim, B, T, delta, num_action, c))
    ]
    PARAMS_DF = sim_env_v4.NON_STAT_PARAMS_DF
    omniscient_dict = {}
    for user_id in sampled_users:
        base_params = sim_env_v4.get_base_params_for_user(user_id)
        adv_params = sim_env_v4.get_adv_params_for_user(user_id).flatten()
        user_params = np.concatenate([base_params, adv_params[6:]]) #, adv_params[7:]])
        # print("user_params: ", user_params)
        omniscient_dict[user_id] = rl_algorithms_budget.Omniscient(dim, B, c, T, num_action, user_params)
    
    return algorithms, omniscient_dict

def run_experiment(exp_kwargs, exp_path, job_type):
    if job_type == "simulations":
        run_simulations(exp_kwargs, exp_path)
    elif job_type == "compute_metrics":
        run_compute_metrics(exp_path)
    else:
        print("ERROR: NO JOB_TYPE FOUND - ", job_type)

# Note: hard-coded values are the exact values used in the Oralytics MRT
def run_simulations(exp_kwargs, exp_path):
    ## HANDLING RL ALGORITHM CANDIDATE ##
    cluster_size = get_cluster_size(exp_kwargs["cluster_size"])
    # offline_or_online = exp_kwargs["offline_or_online"]
    current_seed = exp_kwargs["seed"]
    state = exp_kwargs["state"]
    print(f"State: {state}")
    print("SEED: ", current_seed)
    np.random.seed(current_seed)
    # alg_candidate = rl_algorithm.OralyticsMRTAlg(offline_or_online)

    all_user = sim_env_v4.SIM_ENV_USERS
    sampled_users = np.random.choice(all_user, size=72, replace=False)

    data_pickle_template = exp_path + '/{}_data_df.p'
    update_pickle_template = exp_path + '/{}_update_df.p'

    if cluster_size == 1:
        # Test all 5 budget algorithms
        budget_algorithms, omniscient_dict = create_budget_algorithms(sampled_users)
        
        for alg_name, alg_candidate in budget_algorithms:
            print(f"Testing algorithm: {alg_name}")

            environment_module = sim_env_v4.SimulationEnvironmentV4(state)
            
            # Run experiment with one algorithm per user
            # env_users = environment_module.get_users()
            alg_candidates = [copy.deepcopy(alg_candidate) for _ in range(len(sampled_users))]
            data_df = rl_experiments.run_experiment(alg_candidates, environment_module, alg_name, omniscient_dict, sampled_users)
            
            # Save results with algorithm name
            data_df_pickle_location = exp_path + f'/{alg_name}_{current_seed}_data_df.p'

            print(f"TRIAL DONE for {alg_name}, PICKLING NOW")
            pd.to_pickle(data_df, data_df_pickle_location)

    else:
        print("ERROR: NO CLUSTER_SIZE FOUND - ", cluster_size)
    # elif cluster_size == NUM_TRIAL_USERS:
    #     print("SEED: ", current_seed)
    #     np.random.seed(current_seed)
    #     environment_module = sim_env_v4.SimulationEnvironmentV4(state)
    #     data_df, update_df = rl_experiments.run_incremental_recruitment_exp(alg_candidate, environment_module)
    #     data_df_pickle_location = data_pickle_template.format(current_seed)
    #     update_df_pickle_location = update_pickle_template.format(current_seed)

    #     print("TRIAL DONE, PICKLING NOW")
    #     pd.to_pickle(data_df, data_df_pickle_location)
    #     pd.to_pickle(update_df, update_df_pickle_location)

def run_compute_metrics(exp_path):
    # Define algorithms to process
    algorithms = ["UCB_NoNudge", "UCB_UniformNudge", "UCB_NoReveal", "DABBI_Nudging_I", "DABBI_Nudging_II"]
    
    # Define staleness bins (fixed value ranges)
    tau_bins = [0, 2, 5, 10, 20, np.inf]  # [0-2], [3-5], [6-10], [11-20], >20
    tau_labels = ["0-2", "3-5", "6-10", "11-20", ">20"]
    
    # Process each algorithm separately
    for alg_name in algorithms:
        print(f"\nProcessing algorithm: {alg_name}")
        
        # Build staleness dataframe for this algorithm
        outcomes_df = compute_metrics.build_outcomes_df_for_algorithm(exp_path, MAX_SEED_VAL, alg_name)
        
        # Compute all staleness-based metrics for this algorithm
        metrics_results = {}
        
        if len(outcomes_df) > 0:
            print(f"Found {len(outcomes_df)} records for {alg_name}")
            
            # quality trajectories (3D tensor)
            metrics_results['quality_trajectories'] = compute_metrics.quality_trajectories(outcomes_df)
            metrics_results['regret_trajectories'] = compute_metrics.regret_trajectories(outcomes_df)
            # Nudge efficiency trajectories
            metrics_results['nudge_efficiency'] = compute_metrics.nudge_efficiency_trajectories(outcomes_df)[0]
            metrics_results['total_dq_cum'] = compute_metrics.nudge_efficiency_trajectories(outcomes_df)[1]
            metrics_results['total_nudges_cum'] = compute_metrics.nudge_efficiency_trajectories(outcomes_df)[2]

            
            # Staleness-based metrics (3D tensors)
            metrics_results['nudge_rate_by_tau'] = compute_metrics.nudge_rate_by_staleness(outcomes_df, tau_bins, tau_labels)
            metrics_results['open_rate_by_tau'] = compute_metrics.open_rate_by_staleness(outcomes_df, tau_bins, tau_labels)
            metrics_results['dq_gain_by_tau'] = compute_metrics.dq_gain_by_staleness(outcomes_df, tau_bins, tau_labels)
            metrics_results['dq_gain_by_tau_and_y'] = compute_metrics.dq_gain_by_staleness_and_y(outcomes_df, tau_bins, tau_labels)
            metrics_results['dq_lift_by_tau'] = compute_metrics.dq_lift_by_staleness(outcomes_df, tau_bins, tau_labels)
            
            # Action disagreement metrics
            distance_bins = [0.33, 0.67]  # 3 bins for distance
            metrics_results['action_disagreement'] = compute_metrics.action_disagreement_by_seed_user(outcomes_df, distance_bins)
            
            # Per-user heterogeneity analysis
            user_df = compute_metrics.compute_per_user_heterogeneity(outcomes_df)
            classification_results = compute_metrics.classify_users_by_heterogeneity(user_df)
            metrics_results['user_heterogeneity'] = {
                "user_metrics": user_df,
                "classification": classification_results
            }
        else:
            print(f"Warning: No data found for {alg_name}")
            metrics_results = {"error": "No data available"}
        
        # Save metrics for this algorithm
        metrics_pickle_location = exp_path + f'/{alg_name}_comprehensive_metrics.p'
        with open(metrics_pickle_location, 'wb') as handle:
            pickle.dump(metrics_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Comprehensive metrics for {alg_name} saved to: {metrics_pickle_location}")
        print(f"Available metrics: {list(metrics_results.keys())}")
    
    print(f"\nCompleted processing all algorithms in: {exp_path}")