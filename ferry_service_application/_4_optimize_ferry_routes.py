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


# G = (V,A), parameters and variables

def calculate_travel_times(departure, traveltime):
    """
    :param starttime: 10
    :param traveltime: 5
    :return: departure (10) + travel time (5) = arrival time (15)
    """
    sum_result = [a + b for a, b in zip(departure, traveltime)]
    return sum_result


def get_time_windows():
    """
    The time windows [e, l] are the same for both the pickup and drop-off stations.    e = earliest start of service;
        results from passenger's departure + travel time to pickup station
    l = latest start of service:
        results from conventional arrival (benchmmark) - travel time from dropoff station       :return:
    """
    e = calculate_travel_times(departure, subroute_A_time)
    l = [i - j for i,j in zip(c_arrival, subroute_C_time)]

    ETW = [0] + e + l + [0]
    LTW = [max(e)] + e + l + [10000]
    return ETW, LTW

def generate_corresponding_matrix(original_distance_matrix, new_V):
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
    #result_df.to_csv(config.NEW_MATRIX_USED_STATIONS)
    print(result_df)
    return result


def run_with_iterations(K_Fleetsize):
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
    K   = [i for i in range(config.K_FLEET_SIZE)]
    Q   = [config.Q_CAPACITY for i in range(config.K_FLEET_SIZE)]
    q   = [0] + [config.q_LOAD_PER_REQUEST for i in range(n)] + [(-1) for i in range(n)] + [0]
    # penalty
    P_F = config.PENALTY_FACTOR

    # STEP 4: generate time / distance matrix for V
    t_ij = generate_corresponding_matrix(config.TIME_MATRIX, V_)
    dist_ij = generate_corresponding_matrix(config.DISTANCE_MATRIX, V_)

    # STEP 5: OPTIMIZATION MODEL

    model = gp.Model("AF-DARP")

    x   = model.addVars(V, V, K, vtype=gp.GRB.BINARY, name="x_ijk")
    u   = model.addVars(V, K, vtype=gp.GRB.INTEGER, name="u_ik")
    w   = model.addVars(V, K, vtype=gp.GRB.INTEGER, name="w_ik")
    r   = model.addVars(P, K, vtype=gp.GRB.INTEGER, name="r_ik")

    # penalized extra time
    p_var = model.addVars(P, K, vtype=gp.GRB.INTEGER, name="p_var")
    d_var = model.addVars(D, K, vtype=gp.GRB.INTEGER, name="d_var")


    # (1a) - operator's objective function
    model.setObjective(gp.quicksum(dist_ij[i][j] * x[i, j, k] for i in V for j in V for k in K)
                       + gp.quicksum(p_var[i, k] * P_F for i in P for k in K)
                       + gp.quicksum(d_var[i, k] * P_F for i in D for k in K),
                       gp.GRB.MINIMIZE, name="(1a)")

    # (1b) - passenger's objective function
    """
    model.setObjective(gp.quicksum(u[i + n, k] - u[i, k] for i in P for k in K)
                       + gp.quicksum(p_var[i, k] * P_F for i in P for k in K)
                       + gp.quicksum(d_var[i, k] * P_F for i in D for k in K),
                       gp.GRB.MINIMIZE, name="(1b)")
    """

    # FLOW CONSTRAINTS
    model.addConstrs((gp.quicksum(x[i, j, k] for j in V for k in K) == 1 for i in P), name="(2)")
    model.addConstrs((gp.quicksum(x[0, i, k] for i in V) == 1 for k in K), name="(3a)")
    model.addConstrs((1 == gp.quicksum(x[i, 2 * n + 1, k] for i in V) for k in K), name="(3b)")
    model.addConstrs((gp.quicksum(x[i, j, k] for j in V) - gp.quicksum(x[n + i, j, k] for j in V) == 0 for i in P for k in K), name="(4)")
    model.addConstrs((gp.quicksum(x[j, i, k] for j in V) - gp.quicksum(x[i, j, k] for j in V) == 0 for i in (set(P) | set(D)) for k in K), name="(5)")

    # TIME CONSTRAINTS
    model.addConstrs((u[j, k] >= (u[i, k] + d[i] + t_ij[i][j]) * x[i, j, k] for i in V for j in V for k in K), name="sechs")




