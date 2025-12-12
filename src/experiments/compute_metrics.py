import pandas as pd
import numpy as np
from scipy import stats
import experiment_global_vars
import sim_env_v4

### GLOBAL VALUES ###
NUM_TRIAL_USERS = sim_env_v4.NUM_TRIAL_USERS
NUM_DECISION_TIMES = experiment_global_vars.NUM_DECISION_TIMES

def get_data_df_values_for_users(data_df, user_idxs, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'].isin(user_idxs))].filter(regex=(regex_pattern)))

# string_prefix is env type, clipping vals, b logistic val
def format_qualities(folder_name, max_seed_val):
  total_qualities = np.zeros(shape=(max_seed_val, NUM_TRIAL_USERS, NUM_DECISION_TIMES))

  # extract pickle
  for i in range(max_seed_val):
    try:
        pickle_name = folder_name + "/{}_data_df.p".format(i)
        data_df = pd.read_pickle(pickle_name)
    except:
        print("Couldn't for {}".format(pickle_name))
    # user_idx is both the unique identifier and an index
    for user_idx in np.unique(data_df['user_idx']):
        user_qualities = get_data_df_values_for_users(data_df, [user_idx], 'quality').flatten()
        total_qualities[i][user_idx] = user_qualities

  return total_qualities


# ------------------------------------------------------------------
# Build one tidy dataframe with staleness, nudges, opens, decision-quality gain, etc.
def build_outcomes_df_for_algorithm(folder_name, max_seed_val, algorithm_name):
    """
    Build outcomes dataframe for a specific algorithm.
    
    Args:
        folder_name: Path to folder containing pickle files
        max_seed_val: Maximum seed value to process
        algorithm_name: Name of the algorithm (e.g., 'DABBI_Nudging_II')
    
    Returns:
        DataFrame with all outcomes data for the specified algorithm
    """
    records = []
    for seed in range(max_seed_val):
        # Try algorithm-specific naming first, then fallback to generic
        pickle_paths = [
            f"{folder_name}/{algorithm_name}_{seed}_data_df.p",
            f"{folder_name}/{seed}_data_df.p"  # Fallback
        ]
        
        data_df = None
        for pickle_path in pickle_paths:
            try:
                data_df = pd.read_pickle(pickle_path)
                break  # Found the file, exit the loop
            except FileNotFoundError:
                continue
        
        if data_df is None:
            print(f"Missing pickle for {algorithm_name} seed {seed}")
            continue

        for user_idx in np.unique(data_df["user_idx"]):
            user_rows = (
                data_df[data_df["user_idx"] == user_idx]
                .sort_values("user_decision_t")
                .reset_index(drop=True)
            )

            t = user_rows["user_decision_t"].to_numpy(dtype=float)
            Y = user_rows["O"].to_numpy(dtype=float)            # nudge decision
            O = user_rows["OP"].to_numpy(dtype=float)           # app open outcome
            mu_hat = user_rows["mu_hat"].to_numpy(dtype=float)
            gamma_hat = user_rows["gamma_hat"].to_numpy(dtype=float)
            quality = user_rows["quality"].to_numpy(dtype=float)
            delta = user_rows["delta"].to_numpy(dtype=float)
            distance = user_rows["distance"].to_numpy(dtype=float)
            q0 = user_rows["q0"].to_numpy(dtype=float)
            q1 = user_rows["q1"].to_numpy(dtype=float)

            # last time app opened prior to (and including) current decision
            last_open = np.where(Y == 1, t, np.nan)
            lt = pd.Series(last_open).ffill().fillna(0).to_numpy()
            tau = t - lt
            regret = user_rows["regret"].to_numpy(dtype=float)

            dq = O * (mu_hat - gamma_hat)
            rg = O * (quality - gamma_hat)

            records.append(pd.DataFrame({
                "seed": seed,
                "user_idx": user_idx,
                "user_decision_t": t,
                "regret": regret,
                "tau": tau,
                "Y": Y,
                "O": O,
                "delta": delta,
                "distance": distance,
                "q0": q0,
                "q1": q1,
                "mu_hat": mu_hat,
                "gamma_hat": gamma_hat,
                "quality": quality,
                "DQ": dq,
                "RG": rg,
                "algorithm": algorithm_name  # Add algorithm name for identification
            }))

    if not records:
        return pd.DataFrame(columns=[
            "seed","user_idx","user_decision_t","tau","Y","O", "delta", "distance", "q0", "q1",
            "mu_hat","gamma_hat","quality","DQ","RG","algorithm"
        ])
    return pd.concat(records, ignore_index=True)


def build_outcomes_df(folder_name, max_seed_val):
    records = []
    for seed in range(max_seed_val):
        pickle_path = f"{folder_name}/{seed}_data_df.p"
        try:
            data_df = pd.read_pickle(pickle_path)
        except FileNotFoundError:
            print(f"Missing pickle: {pickle_path}")
            continue

        for user_idx in np.unique(data_df["user_idx"]):
            user_rows = (
                data_df[data_df["user_idx"] == user_idx]
                .sort_values("user_decision_t")
                .reset_index(drop=True)
            )

            t = user_rows["user_decision_t"].to_numpy(dtype=float)
            Y = user_rows["O"].to_numpy(dtype=float)            # nudge decision
            O = user_rows["OP"].to_numpy(dtype=float)           # app open outcome
            mu_hat = user_rows["mu_hat"].to_numpy(dtype=float)
            gamma_hat = user_rows["gamma_hat"].to_numpy(dtype=float)
            quality = user_rows["quality"].to_numpy(dtype=float)
            delta = user_rows["delta"].to_numpy(dtype=float)
            distance = user_rows["distance"].to_numpy(dtype=float)
            q0 = user_rows["q0"].to_numpy(dtype=float)
            q1 = user_rows["q1"].to_numpy(dtype=float)

            # last time app opened prior to (and including) current decision
            last_open = np.where(Y == 1, t, np.nan)
            lt = pd.Series(last_open).ffill().fillna(0).to_numpy()
            tau = t - lt

            dq = O * (mu_hat - gamma_hat)
            rg = O * (quality - gamma_hat)

            records.append(pd.DataFrame({
                "seed": seed,
                "user_idx": user_idx,
                "user_decision_t": t,
                "tau": tau,
                "Y": Y,
                "O": O,
                "delta": delta,
                "distance": distance,
                "q0": q0,
                "q1": q1,
                "mu_hat": mu_hat,
                "gamma_hat": gamma_hat,
                "quality": quality,
                "DQ": dq,
                "RG": rg,
            }))

    if not records:
        return pd.DataFrame(columns=[
            "seed","user_idx","user_decision_t","tau","Y","O", "delta", "distance", "q0", "q1",
            "mu_hat","gamma_hat","quality","DQ","RG"
        ])
    return pd.concat(records, ignore_index=True)

# ------------------------------------------------------------------
def nudge_efficiency_trajectories(df):
    """
    Return a 3-D tensor [seed, user_idx, time] with the cumulative
    decision-quality gain per cumulative nudges for each trajectory.
    """
    seeds = df["seed"].unique()
    total = np.zeros((len(seeds), NUM_TRIAL_USERS, NUM_DECISION_TIMES))
    total_dq_cum = np.zeros((len(seeds), NUM_TRIAL_USERS, NUM_DECISION_TIMES))
    total_nudges_cum = np.zeros((len(seeds), NUM_TRIAL_USERS, NUM_DECISION_TIMES))
    for seed_idx, seed in enumerate(seeds):
        seed_df = df[df["seed"] == seed]
        users = seed_df["user_idx"].unique()
        for user_idx, user in enumerate(users):
            user_rows = (
                seed_df[seed_df["user_idx"] == user]
                .sort_values("user_decision_t")
                .reset_index(drop=True)
            )
            dq_cum = np.cumsum(user_rows["DQ"].to_numpy(dtype=float))
            nudges_cum = np.cumsum(user_rows["Y"].to_numpy(dtype=float))
            eff = dq_cum / np.maximum(1.0, nudges_cum)
            total[seed_idx, user_idx, :eff.size] = eff
            total_dq_cum[seed_idx, user_idx, :dq_cum.size] = dq_cum
            total_nudges_cum[seed_idx, user_idx, :nudges_cum.size] = nudges_cum
    return total, total_dq_cum, total_nudges_cum

# Generic binning helper
def add_tau_bins(df, bins, labels=None):
    """Add tau_bin column to dataframe based on tau values."""
    out = df.copy()
    out["tau_bin"] = pd.cut(out["tau"], bins=bins, labels=labels, include_lowest=True)
    return out.dropna(subset=["tau_bin"])

    
# Nudge rate p_nudge(tau) - returns 3D tensor [seed, user_idx, bin]
def nudge_rate_by_staleness(df, bins, labels=None):
    """Calculate nudge rate by staleness bins for each user in each seed."""
    seeds = sorted(df["seed"].unique())
    num_bins = len(bins) - 1  # bins are edges, so num_bins = len(bins) - 1
    total = np.zeros((len(seeds), NUM_TRIAL_USERS, num_bins))
    
    for seed_idx, seed in enumerate(seeds):
        seed_df = df[df["seed"] == seed]
        users = seed_df["user_idx"].unique()
        for user_idx, user in enumerate(users):
            user_rows = (
                seed_df[seed_df["user_idx"] == user]
                .sort_values("user_decision_t")
                .reset_index(drop=True)
            )
            binned = add_tau_bins(user_rows, bins, labels)
            rate = binned.groupby("tau_bin", observed=False)["Y"].mean()
            total[seed_idx, user_idx, :rate.size] = rate
    return total

# Open rates p_open^(1), p_open^(0) - returns 3D tensor [seed, user_idx, bin]
def open_rate_by_staleness(df, bins, labels=None):
    """Calculate app opening rates by staleness bins for each user in each seed."""
    seeds = sorted(df["seed"].unique())
    num_bins = len(bins) - 1
    total = np.zeros((len(seeds), NUM_TRIAL_USERS, num_bins))
    
    for seed_idx, seed in enumerate(seeds):
        seed_df = df[df["seed"] == seed]
        users = seed_df["user_idx"].unique()
        for user_idx, user in enumerate(users):
            user_rows = (
                seed_df[seed_df["user_idx"] == user]
                .sort_values("user_decision_t")
                .reset_index(drop=True)
            )
            binned = add_tau_bins(user_rows, bins, labels)
            rate = binned.groupby("tau_bin", observed=False)["O"].mean()
            total[seed_idx, user_idx, :rate.size] = rate
    return total

# Decision-quality gain vs staleness
def dq_gain_by_staleness(df, bins, labels=None):
    seeds = sorted(df["seed"].unique())
    num_bins = len(bins) - 1
    total = np.zeros((len(seeds), NUM_TRIAL_USERS, num_bins))
    for seed_idx, seed in enumerate(seeds):
        seed_df = df[df["seed"] == seed]
        users = seed_df["user_idx"].unique()
        for user_idx, user in enumerate(users):
            user_rows = (
                seed_df[seed_df["user_idx"] == user]
                .sort_values("user_decision_t")
                .reset_index(drop=True)
            )
            binned = add_tau_bins(user_rows, bins, labels)
            rate = binned.groupby("tau_bin", observed=False)["DQ"].mean()
            total[seed_idx, user_idx, :rate.size] = rate
    return total

# Decision-quality gain split by Y (optionally condition on O=1)
def dq_gain_by_staleness_and_y(df, bins, labels=None):
    seeds = sorted(df["seed"].unique())
    num_bins = len(bins) - 1
    total_dq_gain_y0 = np.zeros((len(seeds), NUM_TRIAL_USERS, num_bins))
    total_dq_gain_y1 = np.zeros((len(seeds), NUM_TRIAL_USERS, num_bins))
    for seed_idx, seed in enumerate(seeds):
        seed_df = df[df["seed"] == seed]
        users = seed_df["user_idx"].unique()
        for user_idx, user in enumerate(users):
            user_rows = (
                seed_df[seed_df["user_idx"] == user]
                .sort_values("user_decision_t")
                .reset_index(drop=True)
            )
            binned = add_tau_bins(user_rows, bins, labels)
            rate = binned.groupby(["tau_bin", "Y"], observed=False)["DQ"].mean()
            total_dq_gain_y0[seed_idx, user_idx, :rate.size] = rate.iloc[0]
            total_dq_gain_y1[seed_idx, user_idx, :rate.size] = rate.iloc[1]
    return total_dq_gain_y0, total_dq_gain_y1

# Lift(tau) = E[DQ | Y=1] - E[DQ | Y=0]
def dq_lift_by_staleness(df, bins, labels=None):
    seeds = sorted(df["seed"].unique())
    num_bins = len(bins) - 1
    total_dq_gain_y0, total_dq_gain_y1 = dq_gain_by_staleness_and_y(df, bins, labels)
    total_dq_lift = total_dq_gain_y1 - total_dq_gain_y0
    return total_dq_lift

# Action-disagreement & feature-distance test
def action_disagreement_by_seed_user(df, distance_bins=None):
    """
    Compute action disagreement metrics for each seed and user separately.
    Returns a 3D tensor [seed, user, bin] for each metric.
    
    Args:
        df: Dataframe with columns: seed, user_idx, delta, distance, Y, O, mu_hat, gamma_hat
        distance_bins: List of quantiles for distance binning
        
    Returns:
        dict with tensors for EDQ, p_Y, VPN
    """
    # Filter to disagreement cases only
    disagreement_df = df[df["delta"] == 1].copy()
    
    if len(disagreement_df) == 0:
        return {"error": "No disagreement cases found"}
    
    # Define distance bins if not provided (as quantiles)
    if distance_bins is None:
        distance_bins = [0.33, 0.67]  # Creates 3 bins: Q1, Q2, Q3
    
    # Add distance bins using quantile-based binning
    # duplicates="drop" may result in fewer than 3 bins, so let pandas auto-label
    disagreement_df["distance_bin"] = pd.qcut(
        disagreement_df["distance"], 
        q=4,  # 4 quantiles = 3 bins (or fewer if duplicates)
        duplicates="drop"
    )
    
    disagreement_df = disagreement_df.dropna(subset=["distance_bin"])
    
    if len(disagreement_df) == 0:
        return {"error": "No valid distance bins after filtering"}
    
    # Compute DQ
    disagreement_df["DQ"] = disagreement_df["O"] * (disagreement_df["mu_hat"] - disagreement_df["gamma_hat"])
    
    seeds = sorted(disagreement_df["seed"].unique())
    num_bins = len(disagreement_df["distance_bin"].cat.categories)
    
    # Initialize tensors
    edq_tensor = np.zeros((len(seeds), NUM_TRIAL_USERS, num_bins))
    p_y_tensor = np.zeros((len(seeds), NUM_TRIAL_USERS, num_bins))
    vpn_tensor = np.zeros((len(seeds), NUM_TRIAL_USERS, num_bins))
    
    for seed_idx, seed in enumerate(seeds):
        seed_df = disagreement_df[disagreement_df["seed"] == seed]
        users = seed_df["user_idx"].unique()
        
        for user_idx, user in enumerate(users):
            user_df = seed_df[seed_df["user_idx"] == user]
            
            # Group by distance bin for this user
            grouped = user_df.groupby("distance_bin", observed=False)
            
            for bin_idx, (bin_name, bin_data) in enumerate(grouped):
                n_obs = len(bin_data)
                sum_dq = bin_data["DQ"].sum()
                sum_y = bin_data["Y"].sum()
                
                # Compute metrics
                edq = sum_dq / n_obs if n_obs > 0 else 0
                p_y = sum_y / n_obs if n_obs > 0 else 0
                vpn = sum_dq / sum_y if sum_y > 0 else 0
                
                edq_tensor[seed_idx, user_idx, bin_idx] = edq
                p_y_tensor[seed_idx, user_idx, bin_idx] = p_y
                vpn_tensor[seed_idx, user_idx, bin_idx] = vpn
    
    return {
        "EDQ": edq_tensor,
        "p_Y": p_y_tensor,
        "VPN": vpn_tensor,
        "bin_names": list(disagreement_df["distance_bin"].cat.categories)
    }

# per user heterogeneity (who benefits most from the nudge)


# average across trajectories, average across users,
#  average across trials
def report_mean_quality(total_rewards):
  a = np.mean(np.mean(total_rewards, axis=2), axis=1)

  return "{:.3f} ({:.3f})".format(round(np.mean(a), 3), round(stats.sem(a), 3))

# average across trajectories, lower 25th percentile of users,
# average across trials
def report_lower_25_quality(total_rewards):
  a = np.percentile(np.mean(total_rewards, axis=2), 25, axis=1)



  return "{:.3f} ({:.3f})".format(round(np.mean(a), 3), round(stats.sem(a), 3))

def get_metric_values(folder_name, max_seed_val):
    total_qualities = format_qualities(folder_name, max_seed_val)

    return report_mean_quality(total_qualities), report_lower_25_quality(total_qualities)


def compute_per_user_heterogeneity(df):
    """
    Compute per-user heterogeneity metrics as described in the analysis plan.
    
    Args:
        df: DataFrame with columns: seed, user_idx, Y, O, mu_hat, gamma_hat, quality, q0, q1
        
    Returns:
        dict with user-level metrics and cell classifications
    """
    user_metrics = []
    
    for seed in sorted(df["seed"].unique()):
        seed_df = df[df["seed"] == seed]
        
        for user_idx in sorted(seed_df["user_idx"].unique()):
            user_data = seed_df[seed_df["user_idx"] == user_idx].sort_values("user_decision_t")
            
            if len(user_data) == 0:
                continue
                
            # Extract user-level data
            Y = user_data["Y"].to_numpy()
            O = user_data["O"].to_numpy()
            mu_hat = user_data["mu_hat"].to_numpy()
            gamma_hat = user_data["gamma_hat"].to_numpy()
            quality = user_data["quality"].to_numpy()
            q0 = user_data["q0"].to_numpy()
            q1 = user_data["q1"].to_numpy()
            DQ_real = user_data["RG"].to_numpy()
            
            # 1. Baseline engagement (theoretical)
            b_i = np.mean(q0)  # Always available
            
            # 2. Baseline engagement (empirical) - only if sufficient no-nudge data
            no_nudge_mask = (Y == 0)
            if np.sum(no_nudge_mask) >= 3:  # Require at least 3 no-nudge observations
                b_i_emp = np.mean(O[no_nudge_mask])
            else:
                b_i_emp = np.nan
            
            # 3. Predicted benefit (theoretical uplift)
            B_i = np.mean((q1 - q0) * (mu_hat - gamma_hat))
                      
            # 5. Within-user realized uplift
            nudge_mask = (Y == 1)
            if np.sum(nudge_mask) >= 1 and np.sum(no_nudge_mask) >= 1:
                DQ_nudge = np.mean(DQ_real[nudge_mask])
                DQ_no_nudge = np.mean(DQ_real[no_nudge_mask])
                Delta_i = DQ_nudge - DQ_no_nudge
            else:
                Delta_i = np.nan
            
            # 6. Value per nudge (fallback)
            total_DQ = np.sum(DQ_real)
            total_nudges = max(1, np.sum(Y))
            VPN_i = total_DQ / total_nudges
            
            user_metrics.append({
                "seed": seed,
                "user_idx": user_idx,
                "b_i": b_i,
                "b_i_emp": b_i_emp,
                "B_i": B_i,
                "Delta_i": Delta_i,
                "VPN_i": VPN_i,
                "total_DQ": total_DQ,
                "total_nudges": np.sum(Y),
                "total_time": len(user_data)
            })
    
    user_df = pd.DataFrame(user_metrics)
    return user_df


def classify_users_by_heterogeneity(user_df):
    """
    Classify users into four cells based on baseline engagement and predicted benefit.
    
    Args:
        user_df: DataFrame from compute_per_user_heterogeneity
        
    Returns:
        dict with classification results and cell summaries
    """
    if len(user_df) == 0:
        return {"error": "No user data available"}
    
    # Robust split points (medians)
    b_star = np.median(user_df["b_i"])
    B_star = np.median(user_df["B_i"])
    
    # Classify users into four cells
    user_df["cell"] = "Unknown"
    user_df.loc[(user_df["b_i"] <= b_star) & (user_df["B_i"] >= B_star), "cell"] = "LB/HB"
    user_df.loc[(user_df["b_i"] > b_star) & (user_df["B_i"] >= B_star), "cell"] = "HB/HB"
    user_df.loc[(user_df["b_i"] <= b_star) & (user_df["B_i"] < B_star), "cell"] = "LB/LB"
    user_df.loc[(user_df["b_i"] > b_star) & (user_df["B_i"] < B_star), "cell"] = "HB/LB"
    
    # Compute cell summaries
    cell_summaries = []
    
    for cell in ["LB/HB", "HB/HB", "LB/LB", "HB/LB"]:
        cell_users = user_df[user_df["cell"] == cell]
        
        if len(cell_users) == 0:
            continue
            
        # Basic counts
        n_g = len(cell_users)
        total_time_weight = cell_users["total_time"].sum()
        
        # Share of nudges
        total_nudges_all = user_df["total_nudges"].sum()
        share_y_g = cell_users["total_nudges"].sum() / max(1, total_nudges_all)
        
        # Mean realized uplift (only for users with valid Delta_i)
        valid_delta = cell_users.dropna(subset=["Delta_i"])
        if len(valid_delta) > 0:
            mean_delta_g = valid_delta["Delta_i"].mean()
        else:
            mean_delta_g = np.nan
        
        # Mean value per nudge
        mean_vpn_g = cell_users["VPN_i"].mean()
        
        # Share of realized value
        total_DQ_all = user_df["total_DQ"].sum()
        share_dq_g = cell_users["total_DQ"].sum() / max(1, total_DQ_all)
        
        cell_summaries.append({
            "cell": cell,
            "n_g": n_g,
            "total_time_weight": total_time_weight,
            "share_y_g": share_y_g,
            "mean_delta_g": mean_delta_g,
            "mean_vpn_g": mean_vpn_g,
            "share_dq_g": share_dq_g,
            "b_star": b_star,
            "B_star": B_star
        })
    
    return {
        "user_df": user_df,
        "cell_summaries": pd.DataFrame(cell_summaries),
        "split_points": {"b_star": b_star, "B_star": B_star}
    }


def quality_trajectories(df):
    """
    Compute quality trajectories as 3D tensor (num_seeds, NUM_TRIAL_USERS, NUM_DECISION_TIMES).
    
    Args:
        df: DataFrame with columns: seed, user_idx, user_decision_t, quality
        
    Returns:
        3D numpy array with quality values over time for each user in each seed
    """
    seeds = sorted(df["seed"].unique())
    total = np.zeros((len(seeds), NUM_TRIAL_USERS, NUM_DECISION_TIMES))
    
    for seed_idx, seed in enumerate(seeds):
        seed_df = df[df["seed"] == seed]
        users = seed_df["user_idx"].unique()
        for user_idx, user in enumerate(users):
            user_rows = (
                seed_df[seed_df["user_idx"] == user]
                .sort_values("user_decision_t")
                .reset_index(drop=True)
            )
            qualities = user_rows["quality"].to_numpy()
            total[seed_idx, user_idx, :qualities.size] = qualities
    
    return total


def analyze_user_heterogeneity(folder_name, max_seed_val):
    """
    Complete analysis of per-user heterogeneity.
    
    Args:
        folder_name: Path to folder containing pickle files
        max_seed_val: Maximum seed value to process
        
    Returns:
        dict with complete heterogeneity analysis
    """
    # Build staleness dataframe
    staleness_df = build_outcomes_df(folder_name, max_seed_val)
    
    if len(staleness_df) == 0:
        return {"error": "No data available"}
    
    # Compute per-user metrics
    user_df = compute_per_user_heterogeneity(staleness_df)
    
    # Classify users and compute cell summaries
    classification_results = classify_users_by_heterogeneity(user_df)
    
    return {
        "staleness_df": staleness_df,
        "user_metrics": user_df,
        "classification": classification_results
    }

def regret_trajectories(df):
    """
    Compute regret trajectories as 3D tensor (num_seeds, NUM_TRIAL_USERS, NUM_DECISION_TIMES).
    
    Args:
        df: DataFrame with columns: seed, user_idx, user_decision_t, regret
        
    Returns:
        3D numpy array with regret values over time for each user in each seed
    """
    seeds = sorted(df["seed"].unique())
    total = np.zeros((len(seeds), NUM_TRIAL_USERS, NUM_DECISION_TIMES))
    
    for seed_idx, seed in enumerate(seeds):
        seed_df = df[df["seed"] == seed]
        users = seed_df["user_idx"].unique()
        for user_idx, user in enumerate(users):
            user_rows = (
                seed_df[seed_df["user_idx"] == user]
                .sort_values("user_decision_t")
                .reset_index(drop=True)
            )
            regrets = user_rows["regret"].to_numpy()
            total[seed_idx, user_idx, :regrets.size] = regrets
    
    return total