# Water-Truck-Assigment


from __future__ import print_function
import pandas as pd
import numpy as np
import os
import random
import time
from datetime import datetime, timedelta
import pyodbc
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

#DATA PREP
## IMPORT DATA
#======== HIVE CONNECTION



#=============### Import Greenpath & WT_Properties From Hive

df_greenpath = pd.read_sql(sql, con)
df_wt_prop = pd.read_sql(sql2, con)

#=========================================================
### Create New Column For df_wt_prop_regions (Site+region)

df_wt_prop['site_region'] = df_wt_prop.hivesite + '_'+ df_wt_prop.region

#========================================================
### SQL for Road Properties

# Connection parameters

constr1 = f'DRIVER={{ODBC Driver 11 for SQL Server}}; SERVER={server}; DATABASE={database}; UID={username}; PWD={password}'

con1 = pyodbc.connect(constr1, autocommit=True)
cursor1 = con1.cursor()

df_road_prop = pd.read_sql(sql1, con1)

#=====================================
# Reformat All Dataset to Template

road_prop = df_road_prop[df_road_prop.waterfill_region.notnull()]
road_prop = df_road_prop[['road','density','last_sprayed','length','waterfill_region','hivesite','delta_time','demand','dt_min']]
road_prop = road_prop.rename(columns = {'waterfill_region':'region'})
road_prop.density = pd.to_numeric(road_prop.density, downcast='float')
road_prop.dt_min = pd.to_numeric(road_prop.dt_min, downcast='float')

#================================
#Column Rename

road_data = df_greenpath.rename(columns = {'startpoint':'From','endpoint':'To','distance':'Distance'})

#===============================
#Column Rename WT_Prop
wt_prop = df_wt_prop[['hivesite','unit','capacity','region']]
wt_prop = wt_prop.rename(columns = {'unit':'CN'})

#=============================
### Investigate Greenpath Dataset
for i in df_greenpath.district.unique():
    count_row = len(df_greenpath[df_greenpath.district == i])
    count_ideal = len(df_greenpath[df_greenpath.district == i].startpoint.unique())**2
    if count_row != count_ideal:
        print("WARNING!!")
        print('Imbalance Data District : ', i,'| Actual Row : ',count_row, '| ideal Row : ',count_ideal)

#=============================
### Road Data Input
road_data_ADRO = road_data[road_data.district == 'ADRO']
road_data_TCMM = road_data[road_data.district == 'TCMM']
road_data_MTBU = road_data[road_data.district == 'MTBU']

dry_time = np.average(df_wt_prop.general_dry_time_minute) #min after sprayed
nozle_debit = np.average(df_wt_prop.water_spray_debit) #litres per meter 

wt_prop.region = wt_prop.hivesite+'_'+wt_prop.region

#=============================
### Create Num of Truck and Capacity List
# format --> num_truck, CN, capacity = wt_data[region]
regs = wt_prop.region
wt_data = {}
for reg in wt_prop.region.unique():
    wt_data[reg] = [len(wt_prop[regs == reg].CN),list(wt_prop[regs == reg].capacity),list(wt_prop[regs == reg].CN)]
    
#============================
# Data Preparation
## Create Priority Class

priority_class = ['density','dt_min']
for i in priority_class:
    high = road_prop[i].quantile(0.75)
    mh = road_prop[i].quantile(0.5)
    ml = road_prop[i].quantile(0.15)
    low = road_prop[i].quantile(0.05)
    road_prop['score_'+i] = [4 if i > high else 3 if i > mh else 2 if i > ml else 1 for i in road_prop[i]]
road_prop['score_norm'] = road_prop['score_density'] * road_prop['score_dt_min']
road_prop['site_reg'] = road_prop['hivesite'] + '_' + road_prop['region']

#==========================
### Get Waterfill Location

for j in road_prop.hivesite.unique():
    for i in road_prop[road_prop.hivesite == j].region.unique():
        a = len(road_prop[(road_prop.hivesite == j)])
        b = len(road_prop[road_prop.hivesite == j].region.unique())
        c = len(road_prop[road_prop.region == i])
        print('Region : ',i,' | Road Num : ',c)
    print('Total : ',a)
    print('Region Num : ', b)
    
waterfill = {}
for reg in df_wt_prop.site_region.unique():
    wf = df_wt_prop[df_wt_prop.site_region == reg].waterfill_loc_pos_name.iloc[0]
    waterfill[reg] = wf
    
###===========================####+==========================###+=========
###===========================####+==========================###+=========
###===========================####+==========================###+=========
# Create Alert for Unmatched Waterfill_Region
if len(wt_prop.region.unique()) != len(road_prop.site_reg.unique()):
    a = road_prop.site_reg.unique()
    b = wt_prop.region.unique()
    print('Region Not Found From Road Properties : \n', [i for i in a if i not in b], '\n')
    print('Region Not Found From WT Properties : \n', [i for i in b if i not in a], '\n')
    
if len(road_prop.site_reg.isnull()):
    print('Road With Null WT_REGION : \n', road_prop[road_prop.region.isnull()].road.unique())


#===========================================
#Picked Road to be Served & Demand Mtrx

#selected road
score_thres = 0
road_reg = road_prop[road_prop.score_norm > score_thres]

picked_road = {}
demands = {}

reg_list = road_reg.site_reg.unique()
reg_list = reg_list[~pd.isnull(reg_list)]

for site in reg_list:
    #take out road if its waterfill_region is not listed in waterfill_properties
    if site not in waterfill.keys():
        continue
    #create picked road list
    picked_road[site] = list(road_reg[(road_reg.site_reg == site) & (road_reg.road != waterfill[site]
                                                                    )].sort_values(['road'],ascending = True).road)
    #insert waterfill location into picked list
    picked_road[site].insert(0,waterfill[site])
 
    #create demand list
    demands[site] = list(road_reg[(road_reg.site_reg == site) & (road_reg.road != waterfill[site]
                                                                    )].sort_values(['road'],ascending = True).demand)
    #insert 0 demand in waterfill (waterfill at indx 0)
    demands[site].insert(0,0.0)
    
#============================================
#Create Distance Matrix Optimized

road_conn = {}
dist_mtrx = {}
for site in picked_road.keys():
    road_conn[site] = road_data[(road_data.From.isin(picked_road[site])) & (road_data.To.isin(picked_road[site]))].iloc[:,:3]
    road_conn[site] = road_conn[site].sort_values(['From','To'], ascending = [True,True]).reset_index(drop = True)
    dist_mtrx[site]= [road_conn[site].Distance.iloc[i:i+len(picked_road[site])].tolist() for i in range(0,len(road_conn[site]),len(picked_road[site]))]

    
#============================================
#ALERT
###Data Source
wtp = ''
rp = ''
gp = ''
for i in ['ADRO','MTBU', 'TCMM']:
    wtp = wtp + 'Data WT_Prop ' + i +' : ' + str(len(df_wt_prop[df_wt_prop.hivesite == i])) + '\n'
    rp = rp + 'Data Road_Prop ' + i +' : ' + str(len(df_road_prop[df_road_prop.hivesite == i])) + '\n'
    gp = gp + 'Data Greenpath ' + i +' : ' + str(len(df_greenpath[df_greenpath.district == i])) + '\n'    
ds = wtp+rp+gp

### No Region
nr = ''
for i in ['ADRO','MTBU', 'TCMM']:
    nr = nr + 'No Region Road_Prop ' + i + ' : ' + str(len(road_prop[(road_prop.hivesite == i)&(road_prop.region.isnull())])) + '\n'
nr

### No Waterfill in Connection
nwcx = ''
wf = df_wt_prop.waterfill_loc_pos_name.unique()
dgp = df_greenpath.startpoint.unique()
nwc = [i for i in wf if i not in dgp]
for i in nwc:
    nwcx = nwcx + 'wt_not_in_conn : ' + i + ' | site_reg : '+ \
    df_wt_prop[df_wt_prop.waterfill_loc_pos_name == i].site_region.tolist()[0] + '\n'
    
### Imbalance Data before Optimizer
idbo = ''
for reg in demands.keys():
    idbo = idbo + reg + ' : ' + 'picked_road - ' + str(len(picked_road[reg])) + '|' + \
            'demands - ' + str(len(demands[reg])) + '|' + \
            'dist_mtrx - ' + str(len(dist_mtrx[reg])) + '\n'

### Current Datetime
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
dt_string = datetime.strptime(dt_string, "%d/%m/%Y %H:%M:%S")

###Store to wt_notification table

constr1 = f'DRIVER={{ODBC Driver 11 for SQL Server}}; SERVER={server}; DATABASE={database}; UID={username}; PWD={password}'

con1 = pyodbc.connect(constr1, autocommit=True)
cursor1 = con1.cursor()
cursor1.execute("""INSERT INTO wt_notification (running_date, datasource,no_region,no_waterfill_connection,imbalance_data) values(?,?,?,?,?)""", 
                dt_string, ds, nr, nwcx, idbo)
con1.commit()
cursor1.close()

#==============================================
# Optimizing Work Route

def create_data_model(truck_num, capacity, dist_mtrx, demands):
#def create_data_model(dist_mtrx, truck_num):
    """Stores the data for the problem."""
    #road_num = 100
    data = {}
    data['distance_matrix'] = dist_mtrx
    #data['distance_matrix'] = [[0 if j == i else random.randint(20,2500) for i in range(road_num)] for j in range(road_num)]
    #data['demands']=[0 if i == 0 else random.randint(800,1700) for i in range(len(dist_mtrx[0]))]
    data['demands']= demands
    data['vehicle_capacities'] = capacity
    data['num_vehicles'] = truck_num
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    # Display dropped nodes.
    dropped_nodes = 'Dropped nodes:'
    dropped = []
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
            dropped.append(manager.IndexToNode(node))
    #print(dropped)        
    #print(dropped_nodes)
    # Display routes
    total_distance = 0
    total_load = 0
    rute = {}
    water_rem = {}
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        
        rute['truck '+str(vehicle_id)] = []
        water_rem[vehicle_id] = []
        water_cap = data['vehicle_capacities'][vehicle_id]
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} acc. water ({1}) -> '.format(node_index, route_load)
            
            water_cap = water_cap - data['demands'][node_index] #tes
            rute['truck '+str(vehicle_id)].append(node_index) #tes
            water_rem[vehicle_id].append(water_cap) #tes
            
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} acc. water ({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        #print(plan_output)
        total_distance += route_distance
        total_load += route_load
    #print('Total Distance of all routes: {}m'.format(total_distance))
    #print('Total Water of all routes: {}'.format(total_load))
    
    return rute, water_rem, dropped


def main(truck_num, capacity, dist_mtrx, demands):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model(truck_num, capacity, dist_mtrx, demands)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    # Allow to drop nodes.
    penalty = 10000
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        recommendation, water_rem, dropped = print_solution(data, manager, routing, assignment)
        #print_solution(data, manager, routing, assignment)
        
    return recommendation, water_rem, dropped


#===================================
#RECONSTRUCTION

def reconstruct(road_data_reg, recommendation, reg, picked_road):
    from datetime import datetime
    output = {}
    output['road_route'] = []
    output['status_wt'] = []
    output['wt_num'] = []
    output['region'] = []
    output['wo_num'] = []
    output['created_time'] = []
    
    now = datetime.now()
    ctime = now.strftime("%Y-%m-%d %H:%M:%S")

    #print('output :', output)
    #idt -> get truck index to be joined with wt_data and get WT codenumber(CN)
    for idt, truck in enumerate(recommendation.keys()):
        #get truck CN
        truck_name = wt_data[reg][2][idt]
        for idx, i in enumerate(recommendation[truck]):
            time_stamp = ''.join([str(i) for i in time.localtime(time.time())][:-3])
            road_list = []
            status = []

            if idx < len(recommendation[truck])-1:
                j = idx+1
                k = recommendation[truck][j]
                road = picked_road[i]
                dest = picked_road[k]
                
                #get route for each recommended road to be sprayed
                df = road_data_reg[(road_data_reg.From == road) & (road_data_reg.To == dest)]
                #split data into list of connecting roads
                df_list = [[]] if df.greenpath.values == 'Direct' else df.greenpath.str.split(',').tolist()
                
                #reconstruct the connecting road into route list, and give red flag for "connecting only" road
                if idx < len(recommendation[truck])-2:
                    road_list += [road]  +df_list[0]
                    status += ['GREEN'] + ['RED' for i in range(len(df_list[0]))]
                else:
                    road_list += [road]+df_list[0]+[dest]
                    status += ['GREEN'] + ['RED' for i in range(len(df_list[0]))] + ['GREEN']

                #create data list after recnstruction
                output['wo_num'] += [reg+'_'+truck+'_'+time_stamp for i in range(len(status))]
                output['road_route'] += road_list
                output['status_wt'] += status
                #output['wt_num'] += [truck for i in range(len(status))]
                output['wt_num'] += [truck_name for i in range(len(status))]
                output['region'] += [reg for i in range(len(status))]
                output['created_time'] += [ctime for i in range(len(status))]
            else:
                continue

    #outputs = pd.DataFrame.from_dict(output)
    outputs = output
    return outputs

#==========================================
# Running Programmes and Outputs

start = time.time()
for idx, reg in enumerate(dist_mtrx.keys()):
    print('========================Generating Region :',idx+1, '/', len(dist_mtrx.keys()),'==>', reg)
    truck_num = wt_data[reg][0]
    capacity = wt_data[reg][1]
    recommendation, water_rem, dropped = main(truck_num, capacity, dist_mtrx[reg], demands[reg])
    #eval function is to make string as a variable
    outputs = reconstruct(eval('road_data_'+reg[:4]), recommendation, reg, picked_road[reg])
    df = pd.DataFrame.from_dict(outputs)
    if idx == 0:
        df_result = df
        continue
    df_result = df_result.append(df, ignore_index = True)
    
process_time = time.time() - start
print('======================== FINISHED IN : ',round(process_time,2), ' seconds')
df_result.to_csv('result.csv', index= False)


#================================
#Insert Data to SQL Server Table


constr1 = f'DRIVER={{ODBC Driver 11 for SQL Server}}; SERVER={server}; DATABASE={database}; UID={username}; PWD={password}'
constr1


for index, row in df_result.iterrows():
    print(row)
    cursor1.execute("""INSERT INTO wt_assignment ([road_route], [status_wt],[wt_num],[region],[wo_num],[created_time]) values(?,?,?,?,?,?)""",
                        row['road_route'], row['status_wt'], row['wt_num'], row['region'], row['wo_num'], row['created_time'])
con1.commit()
cursor1.close()
