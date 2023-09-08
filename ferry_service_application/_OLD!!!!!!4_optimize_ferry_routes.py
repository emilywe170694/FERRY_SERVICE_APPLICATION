import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from scipy.spatial.distance import cdist


DISTANCE_MATRIX = '/Users/emilyjlw/PycharmProjects/DARP05/Application3/Data/input/distanzmatrix.csv'
TIME_MATRIX = '/Users/emilyjlw/PycharmProjects/DARP05/Application3/Data/input/zeitmatrix.csv'
PARAMS = '/Users/emilyjlw/PycharmProjects/DARP05/Application3/Data/input/Parameter.csv'
CLEANED_REQUESTS = "/Users/emilyjlw/PycharmProjects/DARP05/Application3/Data/output/PassengerRequests_infeasibleRemoved.csv"


def read_params_from_csv(path):
    parameter_file = '/Data/input/parameter_settings.csv'
    params_df = pd.read_csv(path, header=0,index_col=None)
    params = params_df.set_index('param').to_dict()['value']
    return params



def read_pairedStations_from_csv(path):  # auch in AssignStationsToRequests.py
    requests = []
    request_df = pd.read_csv(path, header=0)
    for index, row in request_df.iterrows():
        requests.append(
            (((row['station_start']), (row['station_end']))))
    return requests


def read_stationCoordinates_from_csv(path):  # auch in AssignStationsToRequests.py
    stations = []
    stations_df = pd.read_csv(path, header=0)
    for index, row in stations_df.iterrows():
        stations.append(
            (float(row['lat']), float(row['lon'])))
    return stations


def createDistanceMatrixStations(station_coordinates):
    # eig wird hier die distanzmatrix eingelesen.
    distances = cdist(station_coordinates, station_coordinates)
    return distances


def readPrecomputedDistanceMatrixStations(path):
    dataframe = pd.read_csv(path, header=None, delimiter=';')
    result = dataframe.to_numpy()
    return result

# ---------
# step 1 read in requests to modell

def readTimeMatrixInMinutes(path):
    dataframe = pd.read_csv(path, header=None, delimiter=';').div(60).round().astype(int)
    result = dataframe.to_numpy()
    return result


# for every request in paired station P.append request[0]
def getPickup():
    paired_stations = read_pairedStations_from_csv(CLEANED_REQUESTS)
    P = []
    for request in paired_stations:
        P.append(request[0])
    return P


# for every request in paired station P.append request[1


# def assign_station_coordinates(requests_assigned):
#     assigned_stations = []
#     for i, j in requests_assigned:
#         assigned_stations.append((stations[i], stations[j]))

# erstellt neue distanzmatrix basierend aus neuem V = {Depot, P, D, Depot}
def specifyMatrix(original_distance_matrix, new_V):
    row = []
    for i in new_V:
        col = []
        for j in new_V:
            col.append(original_distance_matrix[i][j])
        row.append(col)
        result = np.array(row)
    return result


# -------------------
# Input
# todo ich muss jede einzelne route sortieren. denn so nimmt das depot (=21) doppelte Werte an.
def resultingTimeToCsv(route, time):
    t = {}
    for n in range(len(route)):
        ro = dict(zip(route[n], time[n]))
        t.update(ro)
    sort = dict(sorted(t.items()))
    time_sorted = []
    for e in sort:
        time_sorted.append(sort[e])

    request_df = pd.DataFrame(time_sorted)
    #request_df.to_csv(resultingTimes, index=False, header=False)
    return time_sorted


def finalTimetable(route, time):
    t = {}
    for n in range(len(route)):
        ro = dict(zip(route[n], time[n]))
        t.update(ro)
    sort = dict(sorted(t.items()))
    time_sorted = [0]
    for e in sort:
        time_sorted.append(sort[e])

    request_df = pd.DataFrame(time_sorted)
    #request_df.to_csv(resultingTimes, index=False, header=False)
    return time_sorted

def time_str2minutes(time):
    hours, minutes = map(int, time.split(':'))
    return hours * 60 + minutes


def readColumn(column_name, file):
    result = []
    result_df = pd.read_csv(file, header=0)
    for index, row in result_df.iterrows():
        result.append(
            row[column_name])
    return result


def addTime(departure, time_to):
    r = []
    result = []
    for i in range(0, len(departure)):
        result.append(departure[i] + time_to[i])
    return result


def substractTime(departure, time_to):
    r = []
    result = []
    for i in range(0, len(departure)):
        result.append(departure[i] - time_to[i])
    return result


def timeOnWater(station_start, station_end, e_i):
    stations_tm = readTimeMatrixInMinutes(TIME_MATRIX)
    tow = []
    for i in range(0, len(station_start)):
        tow.append(stations_tm[station_start[i]][station_end[i]] + e_i[i])
    return tow


def adaptToModel(output, ascending):  # model gibt immer nach 0,1,2,3,4 aus.. , reihenfolge muss aber nach route gehen
    new_list = []
    for element in output:
        new_list.append(ascending[element])
    return new_list


#TODO bei gleicher Station fehlt ein Index, weil loop
def logGurobiResults(timetable_array, P, D):
    pickup_time = []
    dropout_time = []
    for e in P:
        pickup_time.append(timetable_array[e])
    #pd_pickup = pd.DataFrame(pickup_time, columns=['pickup_time'])

    for f in D:
        dropout_time.append(timetable_array[f])
    #pd_dropout = pd.DataFrame(dropout_time, columns=['dropout_time'])

    params = read_params_from_csv(PARAMS)
    vehicles = params['K']
    kn = [vehicles] * len(D)
    kdf = pd.DataFrame(kn, columns=['#vehicles'])

    request_file = pd.read_csv(CLEANED_REQUESTS)
    request_file['dropout_time'] = dropout_time
    request_file['pickup_time'] = pickup_time
    request_file['#vehicles'] = kdf
    #columns_to_add = pd.concat([request_file, pd_pickup, pd_dropout, kdf], axis=1)

    request_file.to_csv(CLEANED_REQUESTS, index=False)

def get_f_arrival(timetable_array, D, t_from_station):
    dropout_time = []
    for f in D:
        dropout_time.append(timetable_array[f])

    f_arrival = []
    for i in range(len(dropout_time)):
        f_arrival.append(dropout_time[i]+t_from_station[i])
    return f_arrival


def add_arrival_times_to_csv(c_arrival, f_arrival):
    arrival_c = []
    for e in c_arrival:
        arrival_c.append(e)
    #c_arrival = pd.DataFrame(arrival_c, columns=['c_arrival'])

    arrival_f = []
    for f in f_arrival:
        arrival_f.append(f)
    #f_arrival = pd.DataFrame(arrival_f, columns=['f_arrival'])

    request_file = pd.read_csv(CLEANED_REQUESTS)
    request_file['c_arrival'] = arrival_c
    request_file['f_arrival'] = arrival_f
    request_file.to_csv(CLEANED_REQUESTS,  index=False)

def get_fleet_size(parameters, n):
    K = parameters['K']
    if n < K:
        vehicles = n
    else:
        vehicles = K
    return vehicles

def update_feasibility(f):
    params = pd.read_csv(PARAMS)
    params.loc[len(params.index)] = ['Feasibility', f]
    params.to_csv(PARAMS, index=None)

def run():
    infeasible_runs = 0
    parameter = read_params_from_csv(PARAMS)
    pickup_stations = readColumn('station_start', CLEANED_REQUESTS)
    drop_stations = readColumn('station_end', CLEANED_REQUESTS)
    depot_p_d_depot = [34] + pickup_stations + drop_stations + [34]

    conventional_times = readColumn('conventional_t (in min)', CLEANED_REQUESTS)
    t_to_station = readColumn('to_station_time', CLEANED_REQUESTS)
    t_from_station = readColumn('from_station_time', CLEANED_REQUESTS)
    departure_time = readColumn('departure_time', CLEANED_REQUESTS)
    conventional_arrival = addTime(departure_time, conventional_times)
    max_waiting_time = parameter['w_t']

    # EARLIEST AN START-STATION: departure + time_to
    eTW_start = addTime(departure_time, t_to_station)
    lTW_start = [(i + max_waiting_time) for i in eTW_start]

    # EARLIEST AN ZIEL-STATION: eTW_start + Fahrtzeit laut Matrix
    eTW_ziel = timeOnWater(pickup_stations, drop_stations, eTW_start)  # ankunft bei direktfahrt
    lTW_ziel = substractTime(conventional_arrival, (t_from_station))

    n = len(pickup_stations)
    P = [i for i in range(n + 1) if i != 0]
    D = [i for i in range(len(P) + 1, 2 * n + 1)]
    V = [i for i in range(len(depot_p_d_depot))]

    dm = readPrecomputedDistanceMatrixStations(DISTANCE_MATRIX)
    distance = specifyMatrix(dm, depot_p_d_depot)

    tm = readTimeMatrixInMinutes(TIME_MATRIX)
    time = specifyMatrix(tm, depot_p_d_depot)

    # Fahrzeug & Kapazität
    q = [0] + [1 for i in range(n)] + [(-1) for i in range(n)] + [0]
    d = [0] + [i for i in range(n * 2)] + [0]  # todo -- wenn  station i = station j, dann muss d[j] = 0 sein
    K = [i for i in range(get_fleet_size(parameter, n))]
    Q = [100 for i in range(len(K))]

    # ____________________________________________________________
    # Zeitkomponenten
    eTW = [0] + eTW_start + eTW_ziel + [0]
    lTW = [5] + lTW_start + lTW_ziel + [1000]

    # eTW = [0 for i in V]
    # lTW = [1000 for i in V]
    Tk = [500 for i in range(len(K))]
    L = 500
    penaltyFactor = 100
    model = gp.Model("AFRS")

    x = model.addVars(V, V, K, vtype=GRB.BINARY, name="x_ijk")
    u = model.addVars(V, K, vtype=GRB.INTEGER, name="u_ik")
    w = model.addVars(V, K, vtype=GRB.INTEGER, name="w_ik")
    r = model.addVars(P, K, vtype=GRB.INTEGER, name="r_ik")

    var_p = model.addVars(P, K, vtype=GRB.CONTINUOUS, name="var_p") # integer
    var_d = model.addVars(D, K, vtype=GRB.CONTINUOUS, name="var_d")

    model.setObjective(#gp.quicksum(distance[i][j] * x[i, j, k] for i in V for j in V for k in K)
                        gp.quicksum(u[i + n, k] - u[i, k] for i in P for k in K)
                       + gp.quicksum(var_p[i, k] * penaltyFactor for i in P for k in K)
                       + gp.quicksum(var_d[i, k] * penaltyFactor for i in D for k in K), gp.GRB.MINIMIZE)

    # model.setObjective(gp.quicksum(u[i+n,k] - u[i, k] for i in P for k in K), GBR.minimize

    # Flow Constraints
    model.addConstrs((gp.quicksum(x[i, j, k] for j in V for k in K) == 1 for i in P), name="zwei")
    model.addConstrs((gp.quicksum(x[0, i, k] for i in V) == 1 for k in K), name="drei-a")
    model.addConstrs((1 == gp.quicksum(x[i, 2 * n + 1, k] for i in V) for k in K), name="drei-b")
    model.addConstrs((gp.quicksum(x[i, j, k] for j in V) - gp.quicksum(x[n + i, j, k] for j in V) == 0 for i in P for k in K), name="vier")
    model.addConstrs((gp.quicksum(x[j, i, k] for j in V) - gp.quicksum(x[i, j, k] for j in V) == 0 for i in P + D for k in K), name="fuenf")

    # Time and Load Constraints
    model.addConstrs((u[j, k] >= (u[i, k] + d[i] + time[i][j]) * x[i, j, k] for i in V for j in V for k in K), name="sechs")

    model.addConstrs((w[j, k] >= (w[i, k] + q[j]) * x[i, j, k] for i in V for j in V for k in K), name="sieben")
    model.addConstrs((r[i, k] >= u[n + i, k] - (u[i, k] + d[i]) for i in P for k in K), name="acht")
    model.addConstrs((u[2 * n + 1, k] - u[0, k] <= Tk[k] for k in K), name="neun")
    model.addConstrs((u[i, k] + var_p[i, k] >= eTW[i] for i in P for k in K), name="zehn-a-1")
    model.addConstrs((u[i, k] + var_p[i, k] <= lTW[i] for i in P for k in K), name="zehn-b-1")
    model.addConstrs((u[i, k] + var_d[i, k] >= eTW[i] for i in D for k in K), name="zehn-a-2")
    model.addConstrs((u[i, k] + var_d[i, k] <= lTW[i] for i in D for k in K), name="zehn-b-2")
    model.addConstrs((u[0, k] >= eTW[0] for k in K), name="zehn-a-3")
    model.addConstrs((u[0, k] <= lTW[0] for k in K), name="zehn-b-3")

    model.addConstrs((var_p[i, k] <= 5 for i in P for k in K))
    model.addConstrs((var_d[i, k] <= 5 for i in P for k in K))

    model.addConstrs((u[2 * n + 1, k] >= eTW[2 * n + 1] for k in K), name="zehn-a-4")
    model.addConstrs((u[2 * n + 1, k] <= lTW[2 * n + 1] for k in K), name="zehn-b-4")

    model.addConstrs((r[i, k] >= time[i][n + i] for i in P for k in K), name="elf-a")
    model.addConstrs((r[i, k] <= L for i in P for k in K), name="elf-b")

    model.addConstrs((max(0, q[i]) <= w[i, k] for i in V for k in K), name="zwoelf-a")
    model.addConstrs((min(Q[k], Q[k] + q[i]) >= w[i, k] for i in V for k in K), name="zwoelf-b")

    # D_i+n darf nur angefahren werden, wenn das dazugehörige P_i zuvor angefahren wurde
    model.addConstrs((u[i + n, k] >= u[i, k] for i in P for k in K), name="DREIZEHN - NEU")

    # darf von 0 nirgendwo hin ausser P
    model.addConstrs((x[0, i, k] == 0 for i in set().union([0], D, [2 * n + 1]) for k in K), name="my-a")

    # darf niemals nach 0 fahren
    model.addConstrs((x[i, 0, k] == 0 for i in V for k in K), name="my-b")

    # darf nur von einem D nach 2n+1 fahren
    model.addConstrs((x[i, 2 * n + 1, k] == 0 for i in set().union([0], P, [2 * n + 1]) for k in K), name="my-b")

    # darf niemals von 2n+1 wegfahren
    model.addConstrs((x[2 * n + 1, i, k] == 0 for i in V for k in K), name="my-b")

    # darf niemals i = j fahren
    model.addConstrs((x[i, i, k] == 0 for i in V for k in K), name="my-b")

   # model.Params.timeLimit = 15
    model.optimize()

    if (model.status == GRB.OPTIMAL):
        model.printAttr('X')
        first_depot = V[0]
        second_depot = V[-1]
        route_new_v = []
        routes_stations = []
        truck = []

        timetable = []
        times_per_ferry = []

        used_seats = []
        used_seats_per_ferry = []


        for k in K:
            for i in V:
                if i != second_depot and x[first_depot, i, k].x > 0.9:
                    aux = [first_depot, V[i]]
                    times_per_ferry = [u[0, k].X, u[i, k].X]
                    used_seats_per_ferry = [w[0, k].X, w[i, k].X]
                    aux2 = [depot_p_d_depot[0], depot_p_d_depot[i]]
                    while i != second_depot:  # loopt solange bis Fahrzeug zurück am Depot ist
                        j = i
                        for h in V:
                            if j != h and x[j, h, k].x > 0.9:
                                aux.append(V[h])
                                aux2.append(depot_p_d_depot[h])
                                times_per_ferry.append(u[h, k].X)
                                used_seats_per_ferry.append(w[h, k].X)
                                i = h
                    route_new_v.append(aux)
                    routes_stations.append(aux2)
                    truck.append(k)
                    timetable.append(times_per_ferry)
                    used_seats.append(used_seats_per_ferry)

        print('\nRoute per Vehicle (in nodes):', route_new_v)
        print('Route per Vehicle (in stations):', routes_stations)
        print('Time table: ', timetable)
        print('Required capacity', used_seats)

        print("\nn = ", n)

        print('\n** nodes in model **')
        print("P = ", P)
        print("D = ", D)
        print("V = ", V)

        print('\n** actual stations **')
        print("Pickup Stations = ", pickup_stations)
        print("Delivery Stations = ", drop_stations)
        print("V = ", depot_p_d_depot)

        print("\ne_i = ", eTW)
        print("l_i = ", lTW)


        print("\n# Ferries = ", len(K))
        print("Capacity per Ferry", Q)

        print("L = ", L)

        timetable_sorted = resultingTimeToCsv(route_new_v, timetable)

        print('Timetable ordered: ', timetable_sorted)
        print('-----------')

        logGurobiResults(timetable_sorted, P, D)

        f_arr = get_f_arrival(timetable_sorted, D, t_from_station)
        add_arrival_times_to_csv(conventional_arrival, f_arr)

        print(infeasible_runs, '----- number of infeasible runs')
        print('****time table******', timetable)
        print('****time table order ascending******', timetable)
        update_feasibility('FEASIBLE')
    else:
        print("Pickup Stations = ", pickup_stations)
        print("Delivery Stations = ", drop_stations)
        print("\ne_i = ", eTW)
        print("l_i = ", lTW)
        infeasible_runs += 1
        update_feasibility('INFEASIBLE')
        print('ROUTING WITH ', len(K), ' VEHICLES NOT SOLVABLE ')


if __name__ == '__main__':
    run()
