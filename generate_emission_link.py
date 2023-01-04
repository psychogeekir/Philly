# %%
import os
import pandas as pd
import numpy as np

# %%
folder = '/srv/data/qiling/Projects/Philly/scenarios/baseline/input_files_philly'

link_file = os.path.join(folder, 'MNM_input_link')
emission_link_file = os.path.join(folder, 'MNM_input_emission_linkID')

# %%
link_df = pd.read_csv(link_file, sep=' ')
ctm_IDs = link_df.loc[link_df['Type'] == 'CTM', '#ID'].values

# %%
np.savetxt(emission_link_file, ctm_IDs, fmt='%d')
# %%
