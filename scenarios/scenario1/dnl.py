# %%
import os
import sys
import pickle
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


macposts_dir = '/home/qiling/Documents/MAC-POSTS'

# %%
MNM_nb_folder = os.path.join(macposts_dir, 'side_project', 'network_builder')
sys.path.append(MNM_nb_folder)

python_lib_folder = os.path.join(macposts_dir, 'src', 'pylib')
sys.path.append(python_lib_folder)

import MNMAPI   # main DTA package
from MNM_mcnb import MNM_network_builder


# %% baseline
# scenario_folder = '/srv/data/qiling/Projects/Philly/scenarios/baseline'
# num_of_tolled_link = 0

# %% scenario 1
scenario_folder = '/srv/data/qiling/Projects/Philly/scenarios/scenario1'
num_of_tolled_link = 4

# %%
network_data_folder = os.path.join(scenario_folder, 'input_files_philly')

training_data_car_truck_file = os.path.join('/srv/data/qiling/Projects/Philly/philly_final/training_data_AM/philly_AM_data.pickle')

result_folder = os.path.join('/srv/data/qiling/Projects/Philly/result_AM')

epoch = 45



# %%
with open(training_data_car_truck_file, 'rb') as f:  # python 2 to python 3, https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/
    [m_car, L_car, m_truck, L_truck, m_speed_car, m_speed_truck, observed_link_driving_list] = pickle.load(f) 


# %% check demand
# num_intervals = nb.config.config_dict['DTA']['max_interval']

# driving_demand_AM_file = os.path.join(new_folder, 'MNM_input_demand')
# driving_demand_AM = pd.read_csv(driving_demand_AM_file, skiprows=1, header=None, delimiter=' ')
# driving_demand_AM = driving_demand_AM.values
# driving_demand_AM = driving_demand_AM[:, :2*num_intervals + 2]
# assert(driving_demand_AM.shape[1] == 2*num_intervals + 2)

# car_demand = driving_demand_AM[:, 2:num_intervals + 2]
# truck_demand = driving_demand_AM[:, num_intervals + 2:]
# print("car demand: min {}, max {}, truck demand : min {}, max {}".format(np.min(car_demand), np.max(car_demand), np.min(truck_demand), np.max(truck_demand)))

# %%
nb = MNM_network_builder() 
nb.load_from_folder(network_data_folder)
print(nb)

# %%
loss, loss_dict, _, _, f_car, f_truck, \
        x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand \
        = pickle.load(open(os.path.join(result_folder, 'record', str(epoch) + '_iteration.pickle'), 'rb'))

nb.update_demand_path2(f_car, f_truck)
nb.config.config_dict['DTA']['total_interval'] = 3600*2
nb.config.config_dict['DTA']['num_of_tolled_link'] = num_of_tolled_link
nb.config.config_dict['STAT']['rec_volume'] = 1
nb.config.config_dict['STAT']['volume_load_automatic_rec'] = 0
nb.config.config_dict['STAT']['volume_record_automatic_rec'] = 0
nb.config.config_dict['STAT']['rec_tt'] = 1
nb.config.config_dict['STAT']['tt_load_automatic_rec'] = 0
nb.config.config_dict['STAT']['tt_record_automatic_rec'] = 0
nb.dump_to_folder(network_data_folder)

# %%

dta = MNMAPI.mcdta_api()
dta.initialize(network_data_folder)

observed_links = observed_link_driving_list  # np.array([link.ID for link in nb.link_list], dtype=int)
paths_list = np.array([ID for ID in nb.path_table.ID2path.keys()], dtype=int)
dta.register_links(observed_links)
dta.register_paths(paths_list)

dta.install_cc()

# %%
dta.run_whole(True)

# %%
dta.print_simulation_results(os.path.join(network_data_folder, 'record'), 180)

# %%
travel_stats = dta.get_travel_stats()
avg_delay = np.mean(np.nan_to_num(dta.get_waiting_time_at_intersections())) / 60
avg_delay_car = np.mean(np.nan_to_num(dta.get_waiting_time_at_intersections_car())) / 60 
avg_delay_truck = np.mean(np.nan_to_num(dta.get_waiting_time_at_intersections_truck())) / 60 

assert(len(travel_stats) == 4)
print("\n************ travel stats ************")
print("car count: {}".format(travel_stats[0]))
print("truck count: {}".format(travel_stats[1]))
print("car total travel time (hours): {}".format(travel_stats[2]))
print("truck total travel time (hours): {}".format(travel_stats[3]))
print("average vehicle delay (minutes): {}".format(avg_delay))
print("average car delay (minutes): {}".format(avg_delay_car))
print("average truck delay (minutes): {}".format(avg_delay_truck))
print("************ travel stats ************\n")

f = open(os.path.join(network_data_folder, 'record', 'simulation.txt'), 'w')
f.write("************ travel stats ************\n")
f.write("car count: {}\n".format(travel_stats[0]))
f.write("truck count: {}\n".format(travel_stats[1]))
f.write("car total travel time (hours): {}\n".format(travel_stats[2]))
f.write("truck total travel time (hours): {}\n".format(travel_stats[3]))
f.write("average vehicle delay (minutes): {}\n".format(avg_delay))
f.write("average car delay (minutes): {}\n".format(avg_delay_car))
f.write("average truck delay (minutes): {}\n".format(avg_delay_truck))
f.write("************ travel stats ************\n")
f.close()

dta.delete_all_agents()

# %%
emission_text = dta.print_emission_stats()
f = open(os.path.join(network_data_folder, 'record', 'emission.txt'), 'w')
f.write(emission_text)
f.close()