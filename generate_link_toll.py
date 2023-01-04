# %%
import pandas as pd
import numpy as np

# %%
link_file = 'input_files_philly/MNM_input_link'
link_toll_file = 'input_files_philly/MNM_input_link_toll'
link_df = pd.read_csv(link_file, header=None, skiprows=1, delimiter=" ")
# %%
link_ctm_df = link_df.loc[link_df[1] == 'CTM', :]
# %%
link_ctm_sampled = link_ctm_df.loc[:, [0]].sample(frac=0.1, replace=False, axis=0)
link_ctm_sampled.rename(columns={0:'link_ID'}, inplace=True)
# %%
link_ctm_sampled['toll'] = np.random.rand(link_ctm_sampled.shape[0]) * 4
# %%
link_ctm_sampled.to_csv(link_toll_file, index=False, sep=" ")
# %%
