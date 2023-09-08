import gurobipy as gp
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
import path

pseudo_depot        = path.PSEUDO_DEPOT
parameter           = path.read_parameter()
pickup_stations     = path.read_column('pickup_station', path.FINAL_PAX_REQUESTS)
dropoff_stations    = path.read_column('dropoff_station', path.FINAL_PAX_REQUESTS)

conventional_times  = path.read_column('conventional_t (in min)', path.FINAL_PAX_REQUESTS)
time_to_pickup      = path.read_column('time_to_pickup', path.FINAL_PAX_REQUESTS)
time_from_dropoff   = path.read_column('time_from_dropoff', path.FINAL_PAX_REQUESTS)
departure_time      = path.read_column('departure', path.FINAL_PAX_REQUESTS)


def get_arrival_time(departure, dest_location):
    sum_result = [a + b for a, b in zip(departure, dest_location)]
    return sum_result

#TODO: Required?
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

def sorted_delay(route, delay):
    t = {}
    for n in range(len(route)):
        ro = dict(zip(route[n], delay[n]))
        t.update(ro)
    sort = dict(sorted(t.items()))
    delay_sorted = []
    for e in sort:
        delay_sorted.append(sort[e])

    request_df = pd.DataFrame(delay_sorted)
    #request_df.to_csv(resultingTimes, index=False, header=False)
    return delay_sorted



def log_times(timetable_array, delay, P, D):
    pickup_time     = []
    dropout_time    = []
    delayed_arrival = []
    for e in P:
        pickup_time.append(timetable_array[e])

    for f in D:
        dropout_time.append(timetable_array[f])
        delayed_arrival.append(delay[f])

    vehicles    = path.K_VEHICLE_SIZE
    kn          = [vehicles] * len(P)
    kdf         = pd.DataFrame(kn, columns=['#vehicles'])

    request_file                    = pd.read_csv(path.FINAL_PAX_REQUESTS)
    request_file['dropout_time']    = dropout_time
    request_file['pickup_time']     = pickup_time
    request_file['delay']           = delayed_arrival
    request_file['#vehicles']       = kdf

    request_file.to_csv(path.PAX_REQUESTS_WITH_ROUTING, index=False)


def get_earliest_possible_arrival_at_dropoff(p, d, eTW_pickup):
    """
    Gibt die frühestmmögliche Ankunftszeit bei dropoff station an :
    Fähre verlässt bei eTW_pickup und fährt direkt und ohne Umwege zu dropoff stationFahrtzeit bei direkter Fahrt
    :param p: pickup station = [1,2,3,..]
    :param d: dropoff station = [6,7,8..]
    :param eTW: earliest departure at pickup = [10,13,...]
    :return: earliest arrival at dropoff station
    """
    sum_result = [path.TIME_MATRIX[a][b] + c for a, b, c in zip(p, d, eTW_pickup)]
    return sum_result

def get_latest_arrival_conventional_benchmark(c, d): # c = conventional_arrival, d = fahrtzeit dropoff
    lTW_drop = [i - j for i,j in zip(c, d)]
    return lTW_drop


def generate_eTW_V_var_1():
    # generate time-windows
    """
       VARIANTE 1 - 2 KLEINE ZEITFENSTER
       eTW_pickup ist die departure time des passagiers + fahrtzeit zur pickup station
       lTW_pickup ist eTW_pickup + die maximale Wartezeit

       eTW_dropoff ist eTW_pickup + direkte Fahrtzeit von pickup-station zu dropoff
       lTW_dropoff ist die conventional_arrival_time - fahrtzeit von dropoff zu destination

       möglicher effekt:
       - distanzen werden ggf gespart, da fähre gezwungen ist früh loszufahren, sodass noch andere mitfahrer eingesammelt werden können
       - Was macht die größe der Differenz zwischen lTW_pickup und eTW_dropoff aus? Die Differenz kann in jedem Fall als Fahrtzeit genutzt werden.
       Kann aber auch heißen dass die Fähre unnötig früh losfährt?
    """
    max_waiting_time    = path.wt_MAX_WAITING_TIME

    conventional_arrival        = get_arrival_time(departure_time, conventional_times)

    # TODO: gibt es einen Vorteil durch zwei verschiede Zeitfenster? Vielleicht längere Rechenzeit?

    #
    # eTW_pickup  = get_arrival_time(departure_time, time_to_pickup)
    # lTW_pickup  = [(i + max_waiting_time) for i in eTW_pickup]
    #
    # eTW_dropoff = get_eTW_dropoff(pickup_stations, dropoff_stations, eTW_pickup)
    # lTW_dropoff = get_lTW_dropoff(conventional_arrival, time_from_dropoff)


    # HIER WERDEN ZEITFENSTER VEREINHEITLICHT ALSO BEISPIELSWEISE [0,25] STATT [0, 5] UND [20, 25]

    eTW_pickup  = get_arrival_time(departure_time, time_to_pickup)
    lTW_pickup  = get_latest_arrival_conventional_benchmark(conventional_arrival, time_from_dropoff)

    eTW_dropoff = get_earliest_possible_arrival_at_dropoff(pickup_stations, dropoff_stations, eTW_pickup)
    lTW_dropoff = get_latest_arrival_conventional_benchmark(conventional_arrival, time_from_dropoff)

    # inklusive Depot eTW/lTW
    ETW = [0] + eTW_pickup + eTW_dropoff + [0]
    LTW = [max(eTW_pickup)] + lTW_pickup + lTW_dropoff + [10000]   # LTW depot ist spätestens das späteste ETW eines requests
    return ETW, LTW


#  TODO. schreibe neue matrix als dataframe zu csv datei um ergebnisse schneller validieren zu können.
def generate_corresponding_matrix(original_distance_matrix, new_V):
    """
    - generates new matrix based on V = {Depot, P, D, Depot}
    - writes new matrix to csv to better validate results
    :param original_distance_matrix:  includes every existing station
    :param new_V:  {Depot, P, D, Depot} // includes only used stations
    :return: matrix which only includes used stations
    """
    row = []
    for i in new_V:
        col = []
        for j in new_V:
            col.append(original_distance_matrix[i][j])
        row.append(col)
        result = np.array(row)

    result_df = pd.DataFrame(result, index=new_V, columns=new_V)
    result_df.to_csv(path.NEW_MATRIX_USED_STATIONS)
    print(result_df)
    return result

def log_results():

    pass

def run():
    # "P_ACTUAL" represents the actual stations whereas P represents an iteration of P_ACTUAL that is used in the model
    P_ACTUAL        = path.read_column('pickup_station', path.FINAL_PAX_REQUESTS)
    D_ACTUAL        = path.read_column('dropoff_station', path.FINAL_PAX_REQUESTS)
    V_ACTUAL        = pseudo_depot + P_ACTUAL + D_ACTUAL + pseudo_depot

    # STEP 1: generate time-windows
    E_TW, L_TW = generate_eTW_V_var_1()

    # STEP 2: set model nodes
    n = len(P_ACTUAL)
    P = [i for i in range(1, n + 1)]
    D = [i for i in range(len(P) + 1, 2 * n + 1)]
    V = [i for i in range(len(V_ACTUAL))]

    # STEP 3: set model parameters

        # time params
    d   = [0] + [path.d_SERVICE_TIMES for i in range(n * 2)] + [0]
    L   = path.L_MAX_RT_PASSENGER
    T_k = [path.TK_MAX_RT_VEHICLE for i in range(path.K_VEHICLE_SIZE)]  # max ride-time per vehicle

        # capacity params
    K   = [i for i in range(path.K_VEHICLE_SIZE)]
    Q   = [path.Q_CAPACITY for i in range(path.K_VEHICLE_SIZE)]
    q   = [0] + [path.q_LOAD_PER_REQUEST for i in range(n)] + [(-1) for i in range(n)] + [0]

        # penalty params
    P_F = path.PENALTY_FACTOR


    # STEP 4: generate time / distance matrix for V
    t_ij = generate_corresponding_matrix(path.TIME_MATRIX, V_ACTUAL)
    dist_ij = generate_corresponding_matrix(path.DISTANCE_MATRIX, V_ACTUAL)


    # STEP 5: OPTIMIZATION MODEL

    model = gp.Model("AF-DARP")

    x   = model.addVars(V, V, K, vtype=gp.GRB.BINARY, name="x_ijk")
    u   = model.addVars(V, K, vtype=gp.GRB.INTEGER, name="u_ik")
    w   = model.addVars(V, K, vtype=gp.GRB.INTEGER, name="w_ik")
    r   = model.addVars(P, K, vtype=gp.GRB.INTEGER, name="r_ik")

    max_var_p = model.addVar( vtype=gp.GRB.INTEGER, name ="max_var-p")
    max_var_d = model.addVar(vtype=gp.GRB.INTEGER, name="max_var-d")

    # penalized extra time
    pvar = model.addVars(P, K, vtype=gp.GRB.INTEGER, name="pvar")
    dvar = model.addVars(D, K, vtype=gp.GRB.INTEGER, name="dvar")


    model.setObjective(gp.quicksum(dist_ij[i][j] * x[i, j, k] for i in V for j in V for k in K)


        + gp.quicksum(dvar[i, k] * P_F for i in D for k in K)
        #+ max_var_p
        # + max_var_d
        , gp.GRB.MINIMIZE)

    # model.setObjective(gp.quicksum(u[i+n,k] - u[i, k] for i in P for k in K), GBR.minimize

    # Flow Constraints
    model.addConstrs((gp.quicksum(x[i, j, k] for j in V for k in K) == 1 for i in P), name="zwei")
    model.addConstrs((gp.quicksum(x[0, i, k] for i in V) == 1 for k in K), name="drei-a")
    model.addConstrs((1 == gp.quicksum(x[i, 2 * n + 1, k] for i in V) for k in K), name="drei-b")
    model.addConstrs((gp.quicksum(x[i, j, k] for j in V) - gp.quicksum(x[n + i, j, k] for j in V) == 0 for i in P for k in K), name="vier")
    model.addConstrs((gp.quicksum(x[j, i, k] for j in V) - gp.quicksum(x[i, j, k] for j in V) == 0 for i in P + D for k in K), name="fuenf")

    # Time and Load Constraints
    model.addConstrs((u[j, k] >= (u[i, k] + d[i] + t_ij[i][j]) * x[i, j, k] for i in V for j in V for k in K), name="sechs")

    model.addConstrs((w[j, k] >= (w[i, k] + q[j]) * x[i, j, k] for i in V for j in V for k in K), name="sieben")
    model.addConstrs((r[i, k] >= u[n + i, k] - (u[i, k] + d[i]) for i in P for k in K), name="acht")
    model.addConstrs((u[2 * n + 1, k] - u[0, k] <= T_k[k] for k in K), name="neun")
    model.addConstrs((u[i, k] >= E_TW[i] for i in P for k in K), name="zehn-a-1")
    model.addConstrs((u[i, k] <= L_TW[i] for i in P for k in K), name="zehn-b-1")
    model.addConstrs((u[i, k] - dvar[i, k] >= E_TW[i] for i in D for k in K), name="zehn-a-2")
    model.addConstrs((u[i, k] - dvar[i, k] <= L_TW[i] for i in D for k in K), name="zehn-b-2")
    #model.addConstrs((pvar[i, k] <= 10 for i in P for k in K), name="zehn-a-1")
    model.addConstrs((u[0, k] >= E_TW[0] for k in K), name="zehn-a-3")
    model.addConstrs((u[0, k] <= L_TW[0] for k in K), name="zehn-b-3")

    model.addConstrs((u[2 * n + 1, k] >= E_TW[-1] for k in K), name="zehn-a-4")
    model.addConstrs((u[2 * n + 1, k] <= L_TW[-1] for k in K), name="zehn-b-4")

    model.addConstrs((dvar[i, k] <= 10 for i in D for k in K), name="dvar-Begrenzung NEU!")

    model.addConstrs((r[i, k] >= t_ij[i][n + i] for i in P for k in K), name="elf-a")
    model.addConstrs((r[i, k] <= L for i in P for k in K), name="elf-b")
    model.addConstrs((max(0, q[i]) <= w[i, k] for i in V for k in K), name="zwoelf-a")
    model.addConstrs((min(Q[k], Q[k] + q[i]) >= w[i, k] for i in V for k in K), name="zwoelf-b")

    #model.addConstr(max_var_p == gp.max_(pvar[i, k] for i in P for k in K))
    model.addConstr(max_var_d == gp.max_(dvar[i, k] for i in D for k in K))

    # ARC STRUCTURE:

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



    if model.status == gp.GRB.OPTIMAL:
        print('START')
        model.printAttr('X')
        print('END')
        first_depot     = V[0]
        second_depot    = V[-1]

        route_new_v     = []
        routes_stations = []
        truck           = []

        timetable       = []
        times_per_ferry = []

        used_seats      = []
        used_seats_per_ferry = []

        delay_per_ferry = []
        delay_dvar = []

        for k in K:
            for i in V:
                if i != second_depot and x[first_depot, i, k].x > 0.9:
                    aux = [first_depot, V[i]]
                    times_per_ferry = [u[0, k].X, u[i, k].X]
                    #todo hier ist was falsch, das erste w hat immer die maximale Q
                    used_seats_per_ferry = [w[0, k].X, w[i, k].X]
                    try:
                        delay_per_ferry = [0, dvar[i, k].X]
                    except:
                        delay_per_ferry = [0, 0]
                    # transferred to actual stations
                    aux2 = [V_ACTUAL[0], V_ACTUAL[i]]
                    while i != second_depot:  # loopt solange bis Fahrzeug zurück am Depot ist
                        j = i
                        for h in V:
                            if j != h and x[j, h, k].x > 0.9:
                                aux.append(V[h])
                                aux2.append(V_ACTUAL[h])
                                times_per_ferry.append(u[h, k].X)
                                used_seats_per_ferry.append(w[h, k].X)
                                try:
                                    delay_per_ferry.append(dvar[h, k].X)
                                except:
                                    delay_per_ferry.append(int(0))
                                i = h

                    route_new_v.append(aux)
                    routes_stations.append(aux2)
                    truck.append(k)
                    timetable.append(times_per_ferry)
                    used_seats.append(used_seats_per_ferry)
                    delay_dvar.append(delay_per_ferry)

        print('\nRoute per Vehicle (in nodes):', route_new_v)
        print('Route per Vehicle (in stations):', routes_stations)
        print('Time table: ', timetable)
        print('Required capacity (used seats): ', used_seats)
        print('DELAY: ', delay_dvar)

        print("\nn = ", n)
        print("\ne_i = ", E_TW)
        print("l_i = ", L_TW)

        timetable_sorted = resultingTimeToCsv(route_new_v, timetable)

        data = {'e_i': E_TW, 'l_i': L_TW, 'timetable': timetable_sorted}
        df = pd.DataFrame(data)
        print(df)

        print('\n** nodes in model **')
        print("P = ", P)
        print("D = ", D)
        print("V = ", V)

        print('\n** actual stations **')
        print("Pickup Stations = ", pickup_stations)
        print("Delivery Stations = ", dropoff_stations)
        print("V = ", V_ACTUAL)
        print("\n# Ferries = ", len(K))
        print("Capacity per Ferry", Q)
        print("L = ", L)
        print("Penalty_Factor =  ", P_F)


        print('Timetable ordered: ', timetable_sorted)
        print('-----------')


        delay_sorted = sorted_delay(route_new_v, delay_dvar)
        log_times(timetable_sorted, delay_sorted, P, D)


        #f_arr = get_f_arrival(timetable_sorted, D, t_from_station)
        #add_arrival_times_to_csv(conventional_arrival, f_arr)

    else:
        print("Pickup Stations = ", pickup_stations)
        print("Delivery Stations = ", dropoff_stations)
        print("\ne_i = ", E_TW)
        print("l_i = ", L_TW)
        #infeasible_runs += 1
        #update_feasibility('INFEASIBLE')
        print('ROUTING WITH ', len(K), ' VEHICLES NOT SOLVABLE ')



if __name__ == '__main__':
    run()


