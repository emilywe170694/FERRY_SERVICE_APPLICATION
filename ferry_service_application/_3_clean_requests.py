import pandas as pd
import config
import _1_generate_demand
import _2_assign_stations
import random

INITIAL_REQUESTS = "/Users/emilyjlw/PycharmProjects/DARP05/Application3/Data/output/PassengerRequests_departureTime.csv"
CLEANED_REQUESTS = "/Users/emilyjlw/PycharmProjects/DARP05/Application3/Data/output/PassengerRequests_infeasibleRemoved.csv"
PARAMS = '/Users/emilyjlw/PycharmProjects/DARP05/Application3/Data/input/Parameter.csv'

# 12 km/h = 3.33 meter/second
parameter = config.read_parameter()

def read_column(file, column_name):
    result = []
    result_df = pd.read_csv(file, header=0)
    for index, row in result_df.iterrows():
        result.append(
            row[column_name])
    return result

def get_conventional_time():
    times = read_column(config.ASSIGNED_PAX_REQUESTS, 'conventional_t (in min)')
    return times

def get_min_ferry_time():
    times = read_column(config.ASSIGNED_PAX_REQUESTS, 'min_ferry_time')
    return times

def identify_dispensable_requests(con, ferry):
    """

    Compares the conventional travel time to the minimum ferry time (no waiting times, direct travel)
    A request is 'dispensable' when 'min ferry-route' > 'conventional travel time'.

    :param con: conventional times per passenger in array = [30, 54, 20...]
    :param ferry: min ferry-routes = [34, 5, 45...]
    :return: index of requests to exclude
    """
    index_to_exclude = []

    for i in range(len(ferry)):
        if ferry[i] >= con[i]:
            index_to_exclude.append(i)

    return index_to_exclude


def reject_dispensable_requests(index_to_drop):
    """
    Die in Methode identify_dispensable_requests identifizierten rows gelten als nicht akzeptable Requests.
    Es wird ein neuer Requests_dataframe generiert, der nur die akzeptieren Requests enthält.
    Anschließend werden die gedroppten Requests durch neue Requests in der Methode replace_rejected_requests ersetzt.
    Die neuen Requests werden wiederum auf Zulässkigkeit geprüft bis das initiale n erreicht ist. 
    :param index_to_drop: Array of indices that are to be droppen from request csv = [1,2,3..]
    :return: 
    """
    requests_df                 = pd.read_csv(config.ASSIGNED_PAX_REQUESTS)
    accepted_requests_df        = pd.DataFrame(requests_df.drop(index_to_drop))
    number_of_accepted_requests = len(accepted_requests_df.index)

    if number_of_accepted_requests < config.n_NUMBER_OF_PASSENGERS:
        new_requests_df = replace_rejected_requests(amount=len(index_to_drop))
        final_df = pd.concat([accepted_requests_df, new_requests_df])
    else:
        final_df = requests_df

    final_df.to_csv(config.FINAL_PAX_REQUESTS, index=None)


def update_n(new_n):
    params = config.return_parameter()
    params.loc[len(params.index)] = ['n_accepted', new_n]
    params.to_csv(config.PARAMETER_SETTINGS, index=None)


def replace_rejected_requests(amount):
    # same as def run, but appending new requests to existing csv instead of creating a new set of requests.
    # requests are build one by one, but methods _1_generate_demand.generate_requests / _2_assign_stations.assign_stations
    # usually deal with more than one request. Therefore  request[0] is the only one required in this method.
    accepted = 0
    accepted_requests = []

    while accepted < amount:
        print('loops starting')
        right_coordinates   = _1_generate_demand.generate_locations_in_polygone(config.DEMAND_POLYGON_RIGHT, 1)
        left_coordinates    = _1_generate_demand.generate_locations_in_polygone(config.DEMAND_POLYGON_LEFT, 1)

        request             = _1_generate_demand.generate_requests(right_coordinates, left_coordinates)

        """request = [[start_lat, start_lon, destination_lat, destination_lon, departure_time, conv_time, conv_distance)]]"""
        request_formatted2  = [[[request[0][0], request[0][1]],
                               [request[0][2], request[0][3]]]]


        assigned_request    = _2_assign_stations.assign_stations(request_formatted2)
        """ assigned_request = [[pickup_station, dropoff_station, assigned_pickup_coord, assigned_dropoff_coord,
                                 distance_to_pickup, distance_from_dropoff, time_to_pickup, time_from_dropoff, min_ferry_time]]"""

        conventional        = request[0][5]
        min_ferry           = assigned_request[0][8]
        if conventional > min_ferry:
            print('accepted')
            accepted += 1
            # jetzt muss ich irgendwie alles in einem array zusammensetzen
            accepted_requests.extend([request[0]+assigned_request[0]])
        else:
            print('rejected, try again')
    print(accepted, ' REQUESTS REPLACED')
    accepted_requests_df = pd.DataFrame(accepted_requests,
                                          columns=['start_lat', 'start_lon',
                                                   'destination_lat', 'destination_lon',
                                                   'departure',
                                                   'conventional_t (in min)', 'conventional_d (in km)',
                                                   'pickup_station', 'dropoff_station',
                                                   'pickup_coord', 'dropoff_coord',
                                                   'distance_to_pickup', 'distance_from_dropoff',
                                                   'time_to_pickup','time_from_dropoff',
                                                   'min_ferry_time'])
    print(accepted_requests_df)
    return accepted_requests_df


def run():
    conventional_travel_time    = get_conventional_time()
    ferry_travel_time           = get_min_ferry_time()

    index_to_drop = identify_dispensable_requests(conventional_travel_time, ferry_travel_time)
    print('REQUEST BY INDEX TO REJECT: ', index_to_drop)

    if not index_to_drop:
        print(''
              'no requests rejected'
              '')
    else:
        reject_dispensable_requests(index_to_drop)
    print("requests CLEANED and written to ", config.FINAL_PAX_REQUESTS)


if __name__ == '__main__':
    run()
