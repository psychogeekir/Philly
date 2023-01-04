# %%
import pandas as pd
import numpy as np
import os

# # %% demand

# num_intervals = 20
# driving_demand_header_line = "#OriginID DestID <car demand by interval>  <truck demand by interval>\n"

# demand_AM_file = os.path.join('/srv/data/qiling/Projects/Philly/input_files_philly/MNM_input_demand')

# demand_AM = np.loadtxt(demand_AM_file, delimiter=' ', skiprows=1)
# assert(demand_AM.shape[1] == num_intervals + 2)

# demand_AM = np.concatenate([demand_AM, np.zeros((demand_AM.shape[0], num_intervals))], 1)

# # %%
# np.savetxt(demand_AM_file, demand_AM, delimiter=' ', fmt='%d %d ' + 2*num_intervals*'%f ')
# f = open(demand_AM_file, 'r')
# log = f.readlines()
# f.close()

# log.insert(0, driving_demand_header_line)

# f = open(demand_AM_file, 'w')
# f.writelines(log)
# f.close()

# # %% node
# node_header_line = "#ID Type Convert_factor(only for Inout node)\n"
# node_file = "/srv/data/qiling/Projects/Philly/input_files_philly/MNM_input_node"

# f = open(node_file, 'r')
# log = f.readlines()
# f.close()

# log[0] = node_header_line
# for i in range(1, len(log)):
#     log[i] = log[i][:-1] + " 1.5\n"

# # %%
# f = open(node_file, 'w')
# f.writelines(log)
# f.close()

# %% link
link_header_line = "#ID Type LEN(mile) FFS_car(mile/h) Cap_car(v/hour) RHOJ_car(v/miles) Lane FFS_truck(mile/h) Cap_truck(v/hour) RHOJ_truck(v/miles) Convert_factor(1)\n"
original_link_file = "/home/qiling/Documents/MAC-POSTS/data/input_files_philly/MNM_input_link"
link_file = "/srv/data/qiling/Projects/Philly/input_files_philly/MNM_input_link"
#ID Type LEN(mile) FFS(mile/h) Cap(v/hour) RHOJ(v/miles) Lane
f = open(original_link_file, 'r')
log = f.readlines()
f.close()

log[0] = link_header_line
for i in range(1, len(log)):
    tmp = log[i].split()
    if tmp[1] == "CTM":
        if float(tmp[4]) >= 5000:
            tmp[4] = str(1800)

        if float(tmp[4]) / float(tmp[3]) >= float(tmp[5]):
            tmp[5] = str(float(tmp[4]) / float(tmp[3]) + 1)

        truck_part = [
        4/5 * float(tmp[3]),
        4/5 * float(tmp[4]),
        4/5 * float(tmp[5]),
        1.25]

        # while truck_part[2] <= truck_part[1] / truck_part[0]:
        #     truck_part[2] += 1/280 * float(tmp[5])

        # if truck_part[2] >= float(tmp[5]):
        #     raise Exception('Wrong truck parameter for link!')

    elif tmp[1] == "PQ":
        truck_part = [
        float(tmp[3]),
        float(tmp[4]),
        float(tmp[5]),
        1.25]
    else:
        raise Exception('Wrong link type!')
    truck_part = " " + " ".join([str(x) for x in truck_part]) + "\n"
    log[i] = " ".join(tmp) + truck_part

# %%
f = open(link_file, 'w')
f.writelines(log)
f.close()
# %%
