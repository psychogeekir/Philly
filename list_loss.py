# %%
import pickle
import os

result_folder = os.path.join('./result_AM/')

for i in range(0, 100):
    loss, loss_dict, _, _, f_car_i, f_truck_i, \
        x_e_car_i, x_e_truck_i, tt_e_car_i, tt_e_truck_i, O_demand_i \
        = pickle.load(open(os.path.join(result_folder, 'record', str(i) + '_iteration.pickle'), 'rb'))

    print("epoch: ", i, "total loss: ", loss, "loss components: ", loss_dict)

# %%
