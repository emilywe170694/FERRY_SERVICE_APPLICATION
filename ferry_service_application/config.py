import pandas as pd

# Input Data

root = '/Users/emilyjlw/PycharmProjects/DARP05/Ferry_Application_'

DEMAND_POLYGON_LEFT     = root + "/data/input/demand_polygon_left.csv"
DEMAND_POLYGON_RIGHT    = root + "/data/input/demand_polygon_right.csv"
PARAMETER_SETTINGS      = root + '/data/input/parameter_settings.csv'
STATIONS_COORD_LEFT     = root + '/data/input/stations_coordinates_left.csv'
STATIONS_COORD_RIGHT    = root + '/data/input/stations_coordinates_right.csv'
STATIONS_DIST_MATRIX    = root + '/data/input/stations_distance_matrix.csv'


# Output Data
INITIAL_PAX_REQUESTS        = root + '/data/output/_1_initial_pax_requests.csv'
ASSIGNED_PAX_REQUESTS       = root + '/data/output/_2_assigned_pax_requests.csv'
ASSIGNED_PAX_REQUESTS_TEST  = root + '/data/output/_2_assigned_pax_requests_test.csv'
ACCEPTED_PAX_REQUESTS       = root + '/data/output/_3_accepted_assigned_pax_requests.csv'
FINAL_PAX_REQUESTS          = root + '/data/output/_3_final_pax_requests.csv'
PAX_REQUESTS_WITH_ROUTING   = root + '/data/output/_4_pax_requests_with_routing.csv'
OPTIMIZATION_ITER           = root + '/data/output/_5_OPTIMIZATION_ITERATION.csv'
OPTIMIZATION_ITER_AVG       = root + '/data/output/_5_OPTIMIZATION_ITERATION_AVG.csv'

# SPECIFYING STATIONS (for indexing)
NUMBER_OF_STATIONS_RIGHT_SIDE = 17
NUMBER_OF_STATIONS_LEFT_SIDE  = 17
PSEUDO_DEPOT                  = [34]
BIKING_VELOCITY_IN_METER      = 15000

# CREATE TIME MATRIX FROM PROVIDED DISTANCE MATRIX
FERRY_VELOCITY  = 3.33  # 12 km/h in m/s = 3.33
TIME_MATRIX     = pd.read_csv(STATIONS_DIST_MATRIX, header=None, delimiter=';').div(FERRY_VELOCITY).div(60).round().astype(int)
DISTANCE_MATRIX = pd.read_csv(STATIONS_DIST_MATRIX, header=None, delimiter=';')

# SETTINGS FOR GENERATING REQUESTS
n_NUMBER_OF_PASSENGERS = 10
dt_DEPARTURE_TIME_SPAN = 30
INSTANCES_PER_ITERATION= 5

#  MODEL PARAMETERS
d_SERVICE_TIMES         = 1
K_FLEET_SIZE            = 10
Q_CAPACITY              = 10
q_LOAD_PER_REQUEST      = 1
PENALTY_FACTOR          = 10
TK_MAX_RT_VEHICLE       = 500   # max ride-time per ferry
L_MAX_RT_PASSENGER      = 100   # max ride-time per passenger



# RE-USED METHODS
def generate_time_matrix(distance_matrix):
    dataframe = pd.read_csv(distance_matrix, header=None, delimiter=';').div(FERRY_VELOCITY).div(60).round().astype(int)
    return dataframe


def read_parameter():
    params_df = pd.read_csv(PARAMETER_SETTINGS, header=0, index_col=None)
    params = params_df.set_index('param').to_dict()['value']
    return params


def return_parameter():
    params_df = pd.read_csv(PARAMETER_SETTINGS, header=0, index_col=None)
    return params_df


def read_column(column_name, file):
    result = []
    result_df = pd.read_csv(file, header=0)
    for index, row in result_df.iterrows():
        result.append(
            row[column_name])
    return result


