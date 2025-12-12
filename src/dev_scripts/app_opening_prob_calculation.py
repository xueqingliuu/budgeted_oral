# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
import os
from sklearn.linear_model import LogisticRegression

NUDGE_OR = float(os.getenv('NUDGE_OR', os.getenv('REMINDER_OR', 2.0)))
NUDGE_LOG_ODDS = float(np.log(NUDGE_OR))
NUDGE_INTERACTION_SCALE = float(os.getenv('NUDGE_INTERACTION_SCALE', 0.3))

COL_NAMES = ['user_id', 'user_decision_t', 'decision_time', 'action', 'quality', 'state_tod', 'state_b_bar', 'state_a_bar', 'state_app_engage', 'state_bias']
script_dir = os.path.dirname(os.path.abspath(__file__))
mrt_data_path = os.path.join(script_dir, '../../../OralyticsMRT/oralytics_mrt_data.csv')
MRT_DATA = pd.read_csv(mrt_data_path)
MRT_USERS = MRT_DATA['user_id'].unique()
### HELPERS ###
def get_user_data(df, user_id):
  return df[df['user_id'] == user_id]

### HELPERS ###
def get_datetime(datetime_string):
    return datetime.strptime(datetime_string, '%Y-%m-%d %H:%M:%S')

"""## App Engagement For MRT Users
---
"""

# returns a dataframe with date and prior_day_opened_app values
def get_user_app_engagement_data(user_id):
  user_df = get_user_data(MRT_DATA, user_id)[['decision_time','state_app_engage']]
  user_df['date'] = user_df['decision_time'].apply(lambda datetime_string: get_datetime(datetime_string).date())
  user_df = user_df.groupby('date', as_index=False).sum(numeric_only=True)
  user_df['prior_day_opened_app'] = user_df['state_app_engage'].apply(lambda x: 1 if x > 0 else 0)

  return user_df[["date", "prior_day_opened_app"]]

# for user_id in MRT_USERS:
#   app_opening_data = get_user_app_engagement_data(user_id)['prior_day_opened_app']
#   open_app_idxs = np.nonzero(np.array(app_opening_data))[0]
#   num_days = len(open_app_idxs)
#   total_days = len(app_opening_data)
#   print("{} opened their app {}/{} days" .format(user_id, num_days, total_days))

# prop_open_app = []
# for user_id in MRT_USERS:
#   app_opening_data = get_user_app_engagement_data(user_id)['prior_day_opened_app']
#   open_app_idxs = np.nonzero(np.array(app_opening_data))[0]
#   open_app_prob = len(open_app_idxs) / len(app_opening_data)
#   prop_open_app.append(open_app_prob)

# # saving values to a csv
# app_open_df = pd.DataFrame({'user_id': MRT_USERS, 'app_open_prob': prop_open_app})
# app_open_df.to_csv('../../sim_env_data/v4_app_open_prob.csv')

"""## Per-user logistic regression for app opening
---
Fits a logistic regression per user predicting app opening from other state_* features.
Output: ../../sim_env_data/app_opening_logit_params.csv
"""


def get_user_dt_features_and_labels(user_id):
  user_df = get_user_data(MRT_DATA, user_id).copy()
  state_cols = [c for c in user_df.columns if c.startswith('state_')]
  if 'state_app_engage' not in state_cols:
    return None, None, []
  # Coerce all state_* to numeric
  for c in state_cols:
    user_df[c] = pd.to_numeric(user_df[c], errors='coerce')
  feature_cols = [c for c in state_cols if c != 'state_app_engage']
  # Outcome at decision-time: did the user open the app (1 if >0 else 0)
  user_df['opened'] = user_df['state_app_engage'].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
  # Drop features that are entirely NaN for this user, then impute remaining NaNs with column means
  valid_features = [c for c in feature_cols if user_df[c].notna().any()]
  if len(valid_features) == 0:
    return None, None, []
  user_df[valid_features] = user_df[valid_features].fillna(user_df[valid_features].mean())
  X = user_df[valid_features].to_numpy()
  y = user_df['opened'].to_numpy()
  return X, y, valid_features

coef_rows = []
for user_id in MRT_USERS:
  # Use per-decision-time data (no daily aggregation)
  X, y, feature_cols = get_user_dt_features_and_labels(user_id)
  # if X is None or y is None or len(y) == 0 or len(np.unique(y)) < 2:
  #   continue
  # try:
  model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
  model.fit(X, y)
  # except Exception:
  #   continue
  row = {'user_id': user_id, 'intercept': float(model.intercept_[0])}
  for name, coef in zip(feature_cols, model.coef_[0]):
    row[name] = float(coef)
  # assumed positive effect of nudge on opening probability
  row['nudge'] = NUDGE_LOG_ODDS
  # interactions of nudge with states: scaled by main effect
  for name, coef in zip(feature_cols, model.coef_[0]):
    row[f'nudge_x_{name}'] = float(NUDGE_INTERACTION_SCALE * coef)
  coef_rows.append(row)

if len(coef_rows) > 0:
  coef_df = pd.DataFrame(coef_rows)
  output_path = os.path.join(script_dir, '../../sim_env_data/V4_app_opening_logit_params.csv')
  coef_df.to_csv(output_path, index=False)