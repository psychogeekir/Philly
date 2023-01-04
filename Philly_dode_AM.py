# %%
import scipy
from scipy.sparse import csr_matrix
import numpy as np
import sys
import os
import pickle
import pandas as pd

macposts_dir = '/home/qiling/Documents/MAC-POSTS'

# %%
MNM_nb_folder = os.path.join(macposts_dir, 'side_project', 'network_builder')
sys.path.append(MNM_nb_folder)

python_lib_folder = os.path.join(macposts_dir, 'src', 'pylib')
sys.path.append(python_lib_folder)

import MNMAPI   # main DTA package
from MNM_mcnb import MNM_network_builder
from mcDODE import MCDODE, PostProcessing


# %%
network_data_folder = os.path.join('./input_files_philly/')

training_data_folder = os.path.join('./philly_final/training_data_AM')
training_data_car_truck_file = os.path.join(training_data_folder, 'philly_AM_data.pickle')

result_folder = os.path.join('./result_AM/')
if not os.path.exists(result_folder):
    os.mkdir(result_folder)
if not os.path.exists(os.path.join(result_folder, 'record')):
    os.mkdir(os.path.join(result_folder, 'record'))

# %%
with open(training_data_car_truck_file, 'rb') as f:  # python 2 to python 3, https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/
    [m_car, L_car, m_truck, L_truck, m_speed_car, m_speed_truck, observed_link_driving_list] = pickle.load(f) 
assert(~np.all(np.isnan(m_car)))
assert(~np.all(np.isnan(m_truck)))
assert(~np.all(np.isnan(m_speed_car)))
assert(~np.all(np.isnan(m_speed_truck)))

# %%
nb = MNM_network_builder() 
nb.load_from_folder(network_data_folder)
print(nb)

num_interval = nb.config.config_dict['DTA']['max_interval']

# %%
data_dict = dict()

# %% count
data_dict['car_count_agg_L_list'] = [L_car]
data_dict['truck_count_agg_L_list'] = [L_truck]

data_dict['car_link_flow'] = [m_car]
data_dict['truck_link_flow'] = [m_truck]

# %% travel time
link_length = np.array([nb.get_link(link_ID).length for link_ID in observed_link_driving_list])  # mile
m_tt_car = link_length[np.newaxis, :] / m_speed_car.reshape(num_interval, -1) * 3600  # seconds
m_tt_truck = link_length[np.newaxis, :] / m_speed_truck.reshape(num_interval, -1) * 3600  # seconds
data_dict['car_link_tt'] = [m_tt_car.flatten(order='C')]
data_dict['truck_link_tt'] = [m_tt_truck.flatten(order='C')]

# %% origin registration data
zipcode_origin_biclass_demand = pd.read_csv(os.path.join(training_data_folder, 'dode_zipcode_origin_biclass_demand.csv'), dtype={"zipc": int, "origin_ID": str, "car": float, "truck": float})
zipcode_origin_biclass_demand['origin_ID'] = zipcode_origin_biclass_demand['origin_ID'].apply(lambda x: [int(xi) for xi in x.split(",")])
# zipcode_origin_biclass_demand.loc[0, 'origin_ID']
data_dict['origin_vehicle_registration_data'] = list() 
data_dict['origin_vehicle_registration_data'].append(zipcode_origin_biclass_demand)

# %%
config = dict()
config['use_car_link_flow'] = True
config['use_truck_link_flow'] = True
config['use_car_link_tt'] = False
config['use_truck_link_tt'] = False
config['car_count_agg'] = True
config['truck_count_agg'] = True
config['link_car_flow_weight'] = 1
config['link_truck_flow_weight'] = 1
config['link_car_tt_weight'] = 1e-1
config['link_truck_tt_weight'] = 1e-1
config['num_data'] = 1  # number of data entries for training, e.e., number of days collected
config['observed_links'] = observed_link_driving_list
config['paths_list'] = np.arange(nb.config.config_dict['FIXED']['num_path'])


config['compute_car_link_flow_loss'] = True
config['compute_truck_link_flow_loss'] = True
config['compute_car_link_tt_loss'] = True
config['compute_truck_link_tt_loss'] = True

# origin registration data
config['use_origin_vehicle_registration_data'] = False
config['compute_origin_vehicle_registration_loss'] = True
config['origin_vehicle_registration_weight'] = 1e-6

# %%
dode = MCDODE(nb, config, num_procs=1)

# %%
# is_updated, is_driving_link_covered = dode.check_registered_links_covered_by_registered_paths(result_folder + '/corrected_input_files', add=True)
# data_dict['mask_driving_link'] = is_driving_link_covered

# print('coverage: driving link {}%'.format(
#     data_dict['mask_driving_link'].sum() / len(data_dict['mask_driving_link']) * 100 if len(data_dict['mask_driving_link']) > 0 else 'NA'
# ))

# %%
# is_updated, is_driving_link_covered = dode.check_registered_links_covered_by_registered_paths(result_folder + '/corrected_input_files', add=False)
# assert(is_updated == 0)
# assert(len(is_driving_link_covered) == len(config['observed_links']))
# data_dict['mask_driving_link'] = is_driving_link_covered

# print('coverage: driving link {}%'.format(
#     data_dict['mask_driving_link'].sum() / len(data_dict['mask_driving_link']) * 100 if len(data_dict['mask_driving_link']) > 0 else 'NA'
# ))
# 85%

# %%
dode.add_data(data_dict)

# %%
max_epoch = 100
starting_epoch = 102
link_car_flow_weight = np.ones(max_epoch) * 1
link_truck_flow_weight = np.ones(max_epoch) * 1
link_car_tt_weight = np.ones(max_epoch) * 1e-2  # 1e-2 0-25,
link_truck_tt_weight = np.ones(max_epoch) * 1e-2  # 1e-2 0-25,
origin_vehicle_registration_weight = np.ones(max_epoch) * 1e-2 # 1e-5 0-25, 1e-2 26-45
column_generation = False # np.array([True if (i > 0) and (i // 5 == 0) else False for i in range(max_epoch)])

# %%
# 0-25, car_step_size = 0.1, truck_step_size = 0.1, 
# 26-45, car_step_size = 0.4, truck_step_size = 0.2, 
(f_car, f_truck, x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand, loss_list) = \
    dode.estimate_path_flow_pytorch(max_epoch = max_epoch, algo='NAdam', normalized_by_scale = False,
                                    car_step_size = 0.05, truck_step_size = 0.02, 
                                    car_init_scale = 0.1, truck_init_scale = 0.05, 
                                    link_car_flow_weight=link_car_flow_weight, link_truck_flow_weight=link_truck_flow_weight, 
                                    link_car_tt_weight=link_car_tt_weight, link_truck_tt_weight=link_truck_tt_weight, 
                                    origin_vehicle_registration_weight=origin_vehicle_registration_weight, 
                                    starting_epoch=starting_epoch, store_folder=os.path.join(result_folder, 'record'),
                                    use_file_as_init=None if starting_epoch == 0 else os.path.join(result_folder, 'record', '{}_iteration.pickle'.format(starting_epoch - 1)),
                                    column_generation=column_generation)

# (f_car, f_truck, x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand, loss_list) = \
#     dode.estimate_path_flow(max_epoch=max_epoch, adagrad=True,
#                             car_step_size = 0.2, truck_step_size = 0.1, 
#                             car_init_scale = 0.5, truck_init_scale = 0.3, 
#                             link_car_flow_weight=link_car_flow_weight, link_truck_flow_weight=link_truck_flow_weight, 
#                             link_car_tt_weight=link_car_tt_weight, link_truck_tt_weight=link_truck_tt_weight, 
#                             origin_vehicle_registration_weight=origin_vehicle_registration_weight, 
#                             starting_epoch=starting_epoch,
#                             store_folder=os.path.join(result_folder, 'record'),
#                             use_file_as_init=None if starting_epoch == 0 else os.path.join(result_folder, 'record', '{}_iteration.pickle'.format(starting_epoch - 1)))

pickle.dump([f_car, f_truck,
             x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand,
             loss_list, config, data_dict], open(os.path.join(result_folder, 'record', 'philly_dode_AM.pickle'), 'wb'))

# %%
f_car, f_truck, \
    x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand, \
    loss_list, config, data_dict = pickle.load(open(os.path.join(result_folder, 'record', 'philly_dode_AM.pickle'), 'rb'))

# %%
# %% rerun the dta with training result
nb = MNM_network_builder()  # from MNM_mmnb, for python analysis
nb.load_from_folder(os.path.join(result_folder, 'record', 'input_files_estimate_path_flow'))

dode = MCDODE(nb, config)
dode.add_data(data_dict)
# dta = dode._run_simulation(f_car, f_truck, counter=0, run_mmdta_adaptive=False)
# dta.print_simulation_results(os.path.join(result_folder, 'record'), 180)

start_intervals = np.arange(0, dode.num_loading_interval, dode.ass_freq)
end_intervals = np.arange(0, dode.num_loading_interval, dode.ass_freq) + dode.ass_freq

# %%
# postproc = PostProcessing(dode, dta, f_car, f_truck, result_folder=result_folder)
postproc = PostProcessing(dode, None, f_car, f_truck, x_e_car, x_e_truck, tt_e_car, tt_e_truck, O_demand, result_folder=result_folder)
postproc.get_one_data(start_intervals, end_intervals, j=0)

# %% total loss
postproc.plot_total_loss(loss_list, 'total_loss_pathflow.png')

# %% breakdown loss
postproc.plot_breakdown_loss(loss_list, 'breakdown_loss_pathflow.png')

# %% count
postproc.cal_r2_count()
postproc.scatter_plot_count('link_flow_scatterplot_pathflow.png')

# %% travel time
postproc.cal_r2_cost()
postproc.scatter_plot_cost('link_cost_scatterplot_pathflow.png')
# %%
