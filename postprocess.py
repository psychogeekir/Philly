# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

# %%
baseline_folder = '/srv/data/qiling/Projects/Philly/scenarios/baseline/input_files_philly/record'
scenario1_folder = '/srv/data/qiling/Projects/Philly/scenarios/scenario1/input_files_philly/record'
scenario2_folder = '/srv/data/qiling/Projects/Philly/scenarios/scenario2/input_files_philly/record'
# %%
link_df = pd.read_csv(os.path.join(baseline_folder, 'driving_link_cong_raw.txt'), sep=' ', header=None, skiprows=1)
link_df1 = pd.read_csv(os.path.join(scenario1_folder, 'driving_link_cong_raw.txt'), sep=' ', header=None, skiprows=1)
link_df2 = pd.read_csv(os.path.join(scenario2_folder, 'driving_link_cong_raw.txt'), sep=' ', header=None, skiprows=1)
colnames = ('timestamp', 'driving_link_ID', 'car_inflow', 'truck_inflow', 'car_tt', 'truck_tt', 'car_fftt', 'truck_fftt', 'car_speed', 'truck_speed')
link_df.columns = colnames
link_df1.columns = colnames
link_df2.columns = colnames

# %%
tolled_links = (
22921,
147753)

# westbound, eastbound
# detour_links = {22921: (107779, 122915), 147753: (7779, 22915)}
detour_links = (107779, 122915, 7779, 22915)

# %%
tolled_link_df = link_df.loc[link_df['driving_link_ID'].apply(lambda x: x in tolled_links), :]
tolled_link_df1 = link_df1.loc[link_df1['driving_link_ID'].apply(lambda x: x in tolled_links), :]
tolled_link_df2 = link_df2.loc[link_df2['driving_link_ID'].apply(lambda x: x in tolled_links), :]

# %%
tolled_link_df.loc[:, ['driving_link_ID', 'car_inflow', 'truck_inflow']].groupby(by='driving_link_ID').sum()
# %%
tolled_link_df1.loc[:, ['driving_link_ID', 'car_inflow', 'truck_inflow']].groupby(by='driving_link_ID').sum()
# %%
tolled_link_df2.loc[:, ['driving_link_ID', 'car_inflow', 'truck_inflow']].groupby(by='driving_link_ID').sum()

# %%
detour_link_df = link_df.loc[link_df['driving_link_ID'].apply(lambda x: x in detour_links), :]
detour_link_df1 = link_df1.loc[link_df1['driving_link_ID'].apply(lambda x: x in detour_links), :]
detour_link_df2 = link_df2.loc[link_df2['driving_link_ID'].apply(lambda x: x in detour_links), :]

# %%
detour_link_df.loc[:, ['driving_link_ID', 'car_inflow', 'truck_inflow']].groupby(by='driving_link_ID').sum()
# %%
detour_link_df1.loc[:, ['driving_link_ID', 'car_inflow', 'truck_inflow']].groupby(by='driving_link_ID').sum()
# %%
detour_link_df2.loc[:, ['driving_link_ID', 'car_inflow', 'truck_inflow']].groupby(by='driving_link_ID').sum()