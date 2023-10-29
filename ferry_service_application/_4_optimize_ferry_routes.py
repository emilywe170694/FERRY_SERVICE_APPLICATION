import os
import gurobipy as gp
import numpy as np
import pandas as pd
import config

requests        = config.FINAL_PAX_REQUESTS
pseudo_depot    = config.PSEUDO_DEPOT

# For time windows

c_arrival       = config.read_column('conventional_arrival', requests)
departure       = config.read_column('departure', requests)
subroute_A_time = config.read_column('time_to_pickup', requests)
subroute_C_time = config.read_column('time_from_dropoff', requests)

# For output evaluation
subroute_A_dist    = config.read_column('distance_to_pickup', requests)
subroute_C_dist    = config.read_column('distance_from_dropoff', requests)
c_travel_times     = config.read_column('conventional_t (in min)', requests)
c_distance         = config.read_column('conventional_d (in km)', requests)
start_coord        = [(a, b) for a, b in zip(config.read_column('start_lat', requests), config.read_column('start_lon', requests))]
dest_coord         = [(a, b) for a, b in zip(config.read_column('destination_lat', requests), config.read_column('destination_lon', requests))]
assigned_stations  = [(a, b) for a, b in zip(config.read_column('pickup_station', requests), config.read_column('dropoff_station', requests))]


# G = (V,A), parameters and variables

def calculate_travel_times(departure, traveltime):
    """
    Calculates arrival times based on departure times and travel times.
    """
    sum_result = [a + b for a, b in zip(departure, traveltime)]
    return sum_result


def get_time_windows():
    """
    The time windows [e, l] are the same for both the pickup and drop-off stations.
    e = earliest start of service;
        results from passenger's departure + travel time to pickup station
    l = latest start of service:
        results from conventional arrival (benchmmark) - travel time from dropoff station       :return:
    """
    e = calculate_travel_times(departure, subroute_A_time)
    l = [i - j for i,j in zip(c_arrival, subroute_C_time)]

    ETW =  [0] + e + e + [0]
    LTW = [max(e)] + l + l + [10000]
    return ETW, LTW


def generate_corresponding_matrix(original_distance_matrix, new_V, name):
    """
    generates new matrix based on V = {Depot, P, D, Depot}
    and returns matrix which only includes used stations
    :param original_distance_matrix:  includes every existing station
    :param new_V:  {Depot, P, D, Depot} ; includes only used stations
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
    result_df.to_csv(config.GENERATED_MATRIX + name)
    return result


def optimize(k_fleetsize, objective):
    # "P_" represents the actual stations whereas P represents an iteration of P_ that is used in the model
    P_    = config.read_column('pickup_station', requests)
    D_    = config.read_column('dropoff_station', requests)
    V_    = pseudo_depot + P_ + D_ + pseudo_depot # V_ = [Depot, P,.. P, D,.. D, Depot]

    # STEP 1: generate time-windows
    E_TW, L_TW = get_time_windows()

    # STEP 2: set model nodes
    n = config.n_NUMBER_OF_PASSENGERS
    P = [i for i in range(1, n + 1)]
    D = [i for i in range(len(P) + 1, 2 * n + 1)]
    V = [i for i in range(len(V_))] # V = [0,1,2,3...2n+1]

    # STEP 3: set model parameters
    # time
    d = [0] + [config.d_SERVICE_TIMES for i in range(n * 2)] + [0]
    L = config.L_MAX_RT_PASSENGER
    T_k = [config.TK_MAX_RT_VEHICLE for i in range(config.K_FLEET_SIZE)]
    # capacity
    K   = [i for i in range(k_fleetsize)]
    Q   = [config.Q_CAPACITY for i in range(config.K_FLEET_SIZE)]
    q   = [0] + [config.q_LOAD_PER_REQUEST for i in range(n)] + [(-1) for i in range(n)] + [0]
    # penalty
    P_F = config.PENALTY_FACTOR

    # STEP 4: generate time / distance matrix for V
    t_ij = generate_corresponding_matrix(config.TIME_MATRIX, V_, name='_time.csv')
    dist_ij = generate_corresponding_matrix(config.DISTANCE_MATRIX, V_, name='_distance.csv')

    print('pickup:', P_)
    print('dropoff:', D_)
    data = {'e_i': E_TW, 'l_i': L_TW}
    df = pd.DataFrame(data)
    print(df)

    # STEP 5: OPTIMIZATION MODEL

    model = gp.Model("AF-DARP")

    x   = model.addVars(V, V, K, vtype=gp.GRB.BINARY, name="x_ijk")
    u   = model.addVars(V, K, vtype=gp.GRB.INTEGER, name="u_ik")
    w   = model.addVars(V, K, vtype=gp.GRB.INTEGER, name="w_ik")
    r   = model.addVars(P, K, vtype=gp.GRB.INTEGER, name="r_ik")

    # penalized extra time
    p_var = model.addVars(P, K, vtype=gp.GRB.INTEGER, name="p_var")
    d_var = model.addVars(D, K, vtype=gp.GRB.INTEGER, name="d_var")


    if (objective == 'operator'):

        # (1a) - operator's objective function
        model.setObjective(gp.quicksum(dist_ij[i][j] * x[i, j, k] for i in V for j in V for k in K)
                           + gp.quicksum(d_var[i, k] * P_F for i in D for k in K),
                           gp.GRB.MINIMIZE)

    elif (objective == 'passenger'):

    # (1b) - passenger's objective function
        model.setObjective(gp.quicksum(u[i + n, k] - u[i, k] for i in P for k in K)
                           + gp.quicksum(d_var[i, k] * P_F for i in D for k in K),
                           gp.GRB.MINIMIZE)

    # FLOW CONSTRAINTS
    model.addConstrs((gp.quicksum(x[i, j, k] for j in V for k in K) == 1 for i in P), name="(2)")
    model.addConstrs((gp.quicksum(x[0, i, k] for i in V) == 1 for k in K), name="(3a)")
    model.addConstrs((1 == gp.quicksum(x[i, 2 * n + 1, k] for i in V) for k in K), name="(3b)")
    model.addConstrs((gp.quicksum(x[i, j, k] for j in V) - gp.quicksum(x[n + i, j, k] for j in V) == 0 for i in P for k in K), name="(4)")
    model.addConstrs((gp.quicksum(x[j, i, k] for j in V) - gp.quicksum(x[i, j, k] for j in V) == 0 for i in P + D for k in K), name="(5)")

    # TIME CONSTRAINTS
    model.addConstrs((u[j, k] >= (u[i, k] + d[i] + t_ij[i][j]) * x[i, j, k] for i in V for j in V for k in K), name="(6))")
    model.addConstrs((w[j, k] >= (w[i, k] + q[j]) * x[i, j, k] for i in V for j in V for k in K), name="(7)")
    model.addConstrs((r[i, k] >= u[n + i, k] - (u[i, k] + d[i]) for i in P for k in K), name="(8)") # wenn 'r =' dann Direktfahrt
    model.addConstrs((u[2 * n + 1, k] - u[0, k] <= T_k[k] for k in K), name="(9)")
    model.addConstrs((u[i, k] >= E_TW[i] for i in V for k in K), name="(10a)")
    model.addConstrs((u[i, k] - p_var[i, k] <= L_TW[i] for i in P for k in K), name="(10b_pickup)")
    model.addConstrs((p_var[i, k] <= d_var[i + n, k] for i in P for k in K), name="(bound p)")
    model.addConstrs((u[i, k] - d_var[i, k] <= L_TW[i] for i in D for k in K), name="(10b_dropoff)")

    model.addConstrs((r[i, k] >= t_ij[i][n + i] for i in P for k in K), name="(11a)")
    model.addConstrs((r[i, k] <= L for i in P for k in K), name="(11b)")
    model.addConstrs((max(0, q[i]) <= w[i, k] for i in V for k in K), name="(12a)")
    model.addConstrs((min(Q[k], Q[k] + q[i]) >= w[i, k] for i in V for k in K), name="(12b)")

    model.addConstrs((u[i+n, k] >= u[i, k]  for i in P for k in K), name="DREIZEHN-neu")

    # manipulate penalties:
    # model.addConstrs((d_var[i, k] <= 10 for i in D for k in K), name="dvar-Begrenzung NEU!")
    # model.addConstr(max_var_d == gp.max_(dvar[i, k] for i in D for k in K))

    # ARC STRUCTURE:
    # darf von 0 nirgendwo hin ausser P, darf niemals nach 0 fahren, darf nur von einem D nach 2n+1 fahren,
    # darf niemals von 2n+1 wegfahren, darf niemals i = j fahren
    model.addConstrs((x[0, i, k] == 0 for i in set().union([0], D, [2 * n + 1]) for k in K), name="my-a")
    model.addConstrs((x[i, 0, k] == 0 for i in V for k in K), name="my-b")
    model.addConstrs((x[i, 2 * n + 1, k] == 0 for i in set().union([0], P, [2 * n + 1]) for k in K), name="my-b")
    model.addConstrs((x[2 * n + 1, i, k] == 0 for i in V for k in K), name="my-b")
    model.addConstrs((x[i, i, k] == 0 for i in V for k in K), name="my-b")

    model.setParam('TimeLimit', config.MAX_RUNTIME * 60)  # limits runtime to 15 minutes
    model.optimize()


    if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
        optimality_gap = f"{model.MIPGap * 100}%"
        print(f"Optimality Gap: {optimality_gap * 100}%")
        model.printAttr('X')
        print('END')

        print('PICKUP: ', P_)
        print('DROPOFF: ', D_)

        # make results readable
        FIRST_DEPOT     = V[0]
        SECOND_DEPOT    = V[-1]

        service_times   = []
        load            = []
        delay_dropoff   = []
        delay_pickup    = []
        assigned_ferry  = []

        route_nodes     = []
        route_stations  = []
        route_times     = []
        route_load      = []
        route_delay_p   = []
        route_delay_d   = []
        route_assigned_ferry = []

        distance_driven_per_ferry = []

        for k in K:
            for i in V:
                if i != SECOND_DEPOT and x[FIRST_DEPOT, i, k].x > 0.9:
                    nodes           = [FIRST_DEPOT, V[i]]
                    stations        = [V_[0], V_[i]]
                    service_times   = [u[0, k].X, u[i, k].X]
                    load            = [w[0, k].X, w[i, k].X]
                    assigned_ferry  = [k, k]
                    distance_driven = dist_ij[0][V[i]]
                    try:
                        delay_pickup = [0, p_var[i, k].X]
                    except:
                        delay_pickup = [0, 0]
                    try:
                        delay_dropoff = [0, d_var[i, k].X]
                    except:
                        delay_dropoff = [0, 0]

                    while i != SECOND_DEPOT:
                        j = i
                        for h in V:
                            if j != h and x[j,h,k].x > 0.9:
                                nodes.append(V[h])
                                stations.append(V_[h])
                                service_times.append(u[h, k].X)
                                load.append(w[h, k].X)
                                assigned_ferry.append(k)
                                distance_driven += dist_ij[V[j]][V[h]]
                                try:
                                    delay_pickup.append(p_var[h, k].X)
                                except:
                                    delay_pickup.append(int(0))
                                try:
                                    delay_dropoff.append(d_var[h, k].X)
                                except:
                                    delay_dropoff.append(int(0))
                                i = h
                    route_nodes.append(nodes)
                    route_stations.append(stations)
                    route_times.append(service_times)
                    route_load.append(load)
                    route_delay_p.append(delay_pickup)
                    route_delay_d.append(delay_dropoff)
                    route_assigned_ferry.append(assigned_ferry)
            distance_driven_per_ferry.append(distance_driven)
        print('\nRoute per Vehicle (in nodes):', route_nodes)
        print('Route per Vehicle (in stations):', route_stations)
        print('Time table: ', route_times)
        print('Required capacity (used seats): ', route_load)
        print('DELAY_p: ', route_delay_p)
        print('DELAY_d: ', route_delay_d)
        print('Ferry Assignment', route_assigned_ferry)
        print('Distance driven', distance_driven_per_ferry)

        # returns array with service times ordered according to routes
        timetable_ordered = decapsulate(route_nodes, route_times)

        data = {'e_i': E_TW, 'l_i': L_TW, 'timetable': timetable_ordered}
        df = pd.DataFrame(data)
        print(df)

        # assess output:
        delay_d_ordered = decapsulate(route_nodes, route_delay_d)
        ferry_assigned  = decapsulate(route_nodes, route_assigned_ferry)
        pickup_time     = [timetable_ordered[e] + d[e] for e in P]
        ferry_a         = [ferry_assigned[e] for e in P]

        dropoff_time    = [timetable_ordered[f] + d[f] for f in D]
        delay           = [delay_d_ordered[f] for f in D]
        tw              = [[E_TW[f], L_TW[f]] for f in D]

        f_travel_time   = [(a + b - c) for a, b, c in zip(dropoff_time, subroute_C_time, departure)]
        f_dist_land     = [a + b for a, b in zip(subroute_A_dist, subroute_C_dist)]

        # savings t:
        abs_savings_t = [a - b for a, b in zip(c_travel_times, f_travel_time)]
        rel_savings_t = percentual_savings(c_travel_times, f_travel_time)

        # savings d:
        abs_savings_d = [a - b for a, b in zip(c_distance, f_dist_land)]
        rel_savings_d = percentual_savings(c_distance, f_dist_land)


        # log results to _4_pax_requests_with_routing (individual instance)
        request_file = pd.read_csv(config.FINAL_PAX_REQUESTS)
        request_file['objective'] = objective
        request_file['time_window'] = tw
        request_file['pickup_time'] = pickup_time
        request_file['dropout_time'] = dropoff_time
        request_file['delay_dvar'] = delay
        request_file['assigned_ferry'] = ferry_a
        request_file['ferry_travel_time'] = f_travel_time
        add_header=not os.path.exists(config.PAX_REQUESTS_WITH_ROUTING)
        request_file.to_csv(config.PAX_REQUESTS_WITH_ROUTING, mode='a', header=add_header, index=False)

        ferry_evaluation = {'objective': objective,
                            'fleet_size': len(K),
                            'ferry_no': K,
                            'distance_traveled': distance_driven_per_ferry,
                            'route_nodes': route_nodes,
                            'route_stations': route_stations}
        df_table = pd.DataFrame(ferry_evaluation)
        add_header = not os.path.exists(config.EVALUATION_FERRY)
        df_table.to_csv(config.EVALUATION_FERRY, mode='a', header=add_header, index=False)


        # log savings to _5_evaluation_pax (average of multiple instances):
        pax_evaluation = {'start': start_coord,
                          'destination': dest_coord,
                          'stations': assigned_stations,
                          'departure': departure,
                          'objective': objective,
                          'fleet_size': len(K),
                          'conventional_tt': c_travel_times,
                          'timewindow': tw,
                          'pickup_time': pickup_time,
                          'dropoff_time': dropoff_time,
                          'assigned_ferry': ferry_a,
                          'DELAY': delay,
                          'ferry_tt': f_travel_time,
                          'net_savings_t': abs_savings_t,
                          'rel_savings_t': rel_savings_t,
                          'conventional_dist': c_distance,
                          'ferry_dist_land': f_dist_land,
                          'net_savings_d': abs_savings_d,
                          'rel_savings_d': rel_savings_d,
                          'optimality_gap': [optimality_gap] * config.n_NUMBER_OF_PASSENGERS
                          }

        df_table = pd.DataFrame(pax_evaluation)
        add_header = not os.path.exists(config.EVALUATION_PAX)
        df_table.to_csv(config.EVALUATION_PAX, mode='a', header=add_header, index=False)

    else:
        print('ROUTING WITH ', len(K), ' VEHICLES NOT SOLVABLE ')



def decapsulate(route, variable):
    """
    variables (service times, delay...) are obtained from resulting route(s) for each index in V
    :param route: [[0, 2, 12, 0], [0, 1, 11, 9, 19, 0], ..] >> route of vehicle k = 1,2 ..
    :param time:  [[0, 15:40, 17:20, 0], [0, 15:12, 15:56, 16:12, 16:01, 0], ..] >> vehicle 1 picks up pax 2 at 15:40
    :return: service times ordered by V (by passenger 0, 1, 2 ..) times matched with route >> [0, 15:12, 15:40, 16:12, ...]
    """
    t = {}
    for n in range(len(route)):
        ro = dict(zip(route[n], variable[n]))
        t.update(ro)
    sort = dict(sorted(t.items()))
    time_sorted = []
    for e in sort:
        time_sorted.append(sort[e])

    request_df = pd.DataFrame(time_sorted)
    return time_sorted


def percentual_savings(conv, ferry):
    savings = [(b - a)/a * -100 for a, b in zip(conv, ferry)]
    sr = [round(x, 2) for x in savings]
    return sr


def run_with_iterations(K_i, objective):
    for obj in objective:
        for k_i in K_i:
            optimize(k_fleetsize=k_i, objective=obj)



if __name__ == '__main__':
    run_with_iterations([10], ['operator', 'passenger'])



