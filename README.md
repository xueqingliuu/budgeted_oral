# Simulation Environment

We build on the Oralytics simulation environment from Trella et al. and modify it in three key ways:

a. App-opening mechanism (information-revealing action).
We incorporate an app-opening feature, which is a key model component. The recommender agent can only access and update its belief state using the most recent historical information if and only if the user takes the information-revealing action of opening the app at that time step.

b. Differential app-opening probabilities.
A user’s probability of opening the app is allowed to vary depending on whether or not they received a nudge from the clinicians. Thus, app engagement is modeled as a behavior that may be influenced by clinician nudges.

c. Multi-category action space.
We consider a more challenging intervention setting with a multi-category action space instead of the common binary decision space in mHealth. Specifically, there are five possible actions: a no-message action and four distinct message types.

The scripts described below are used to fit and evaluate this modified simulation environment to the Oralytics trial data. Under `\synthetic`, we provide code and results of our synthetic experiments.



# oralytics-post-deployment-analysis
This repository contains code from the [Anonymous Paper] for performing re-sampling analyses to re-evaluate algorithm decisions made for the RL algorithm deployed in the MRT (phase 1) of the Oralytics trial.


## Fitting Simulation Environment
* Running `python3 src/dev_scripts/fitting_user_models.py` will fit each Oralytics participant to a non-stationary base model class (zero-inflated poisson model) and save parameters to `v4_non_stat_zip_model_params.csv`.
* Running `python3 src/dev_scripts/app_opening_prob_calculation.py.py` will fit an app opening probability for each Oralytics participant and save the probabilities to `v4_app_open_prob.csv`.
* Running `python3 src/dev_scripts/get_participant_start_end_dates.py` will get the start and end dates (i.e., when the participant started and completed the trial) of each Oralytics participant and save the info. to `v4_start_end_dates.csv`.

To run experiments:

1. Fill in the read and write path in `read_write_info.py`. This specifies what path to read data from and what path to write results to.
2. In `run.py`, specify experiment parameters as instructed in the file. Example: specify the simulation environment variants and algorithm candidate properties. You must have the field value `JOB_TYPE = "simulations"` to run these experiments. In addition, you must modify the DRYRUN field to specify running jobs in parallel or sequentially. DRYRUN = True runs jobs one after the other (this is a good practice to test out new code initially). Switch to DRYRUN = False to run experiments in parallel.

The experiment parameters one can specify in `run.py` are:
a. `cluster_size = ["full_pooling", "no_pooling"]` (full-pooling vs. no-pooling algorithm)
b. `offline_or_online = ["online", "offline"]` (online algorithm that updates as data accrues vs. offline algorithm that only uses the prior and does not update the policy)
c. `seed = range(MAX_SEED_VAL)` (`MAX_SEED_VAL` specifies how many Monte-Carlo repititions you want to run)
d. `state` (if `state=[None]` then you are running experiment type (1) for re-evaluating design decisions, otherwise, you are running experiment type (2) did we learn? and `state` needs to be a list with 4 elements corresponding to state feature values that together form the state of interest)

3. Run `python3 src/experiments/submit_batch.py` on the cluster to submit jobs and run in parallel.

## Computing Metrics and Plotting Figures
* For experiment type (1) re-evaluating design decisions, we calculate various metrics on the re-sampled outcomes. To calculate these metrics after running experiments in the above section, first change the `JOB_TYPE` field in `run.py` to `JOB_TYPE = "compute_metrics"` and then run `python3 src/experiments/submit_batch.py` 
* For experiment type (2) did we learn?, we visualize the standardized predicted advantage in state s throughout the trial. To plot these visualization using results after running experiments in the above section, run `python3 src/experiments/plot_predicted_advs.py` 
