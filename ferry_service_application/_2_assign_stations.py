import csv
import random

import pandas as pd
from geopy.distance import geodesic as GD
from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np
from shapely.geometry import Point, Polygon
import config

def get_stations(file_path):
    data_stations = pd.read_csv(file_path, header=0)
    stations = []
    for index, row in data_stations.iterrows():
        stations.append((row['lat'], row['lon']))
    return stations


def get_initial_requests(path):
    requests = []
    request_df = pd.read_csv(path, header=0)
    for index, row in request_df.iterrows():
        request_start = [float(row['start_lat']), float(row['start_lon'])]
        request_dest = [float(row['destination_lat']), float(row['destination_lon'])]
        requests.append([request_start, request_dest])
    return requests


def get_polygone(path):
    points = []
    with open(path, 'r') as station_locations:
        reader = csv.reader(station_locations, delimiter=',')
        for row in reader:
            points.append((float(row[0]), float(row[1])))
    print(points)
    return points


def concat_assigned_stations_to_requests_csv(results_df):
    initial_requests    = pd.read_csv(config.INITIAL_PAX_REQUESTS)
    assigned_requests   = pd.concat([initial_requests, results_df], axis=1)
    assigned_requests.to_csv(config.ASSIGNED_PAX_REQUESTS, index=None)
    print('**** assigned requests written to ', config.ASSIGNED_PAX_REQUESTS)

def assign_stations(requests):
    """
            The nearest stations are selected for the assignment of start and destination points of the initial requests,
            with distances being calculated using the Euclidean metric.
            A location's closest station may be on the opposite side of the river.
            Consequently, it is initially determined in which polygon (on which side of the river)
            the respective location is situated. If the starting point is situated in the left polygon,
            then only a station on the left side can be considered as the starting station.
            Simultaneously, the destination station must be on the right side, along with the destination point.

           Stations [0:16] are located on the left side, stations [17:33] are located on the right side.

            :parameter:
            requests = [[start, destination], [start, destination], ..., [start, destination]]
                     = [[51.2349, 6.7794], [51.2421, 6.7052]], ...]

            :return: results =  ([pickup_station, dropoff_station, assigned_pickup_coord, assigned_dropoff_coord,
                        distance_to_pickup, distance_from_dropoff, time_to_pickup, time_from_dropoff, min_ferry_time])
    """

    stations_left  = get_stations(config.STATIONS_COORD_LEFT)
    stations_right = get_stations(config.STATIONS_COORD_RIGHT)

    poly_left  = Polygon(get_polygone(config.DEMAND_POLYGON_LEFT))
    poly_right = Polygon(get_polygone(config.DEMAND_POLYGON_RIGHT))

    results = []

    for request in requests:
        start       = request[0]
        destination = request[1]

        if Point(start).within(poly_left):
            # assign pickup station
            available_pickup    = stations_left
            available_dropoff   = stations_right

            # stations separated into two lists (stations_left / stations_right) - index of stations needs to be adapted.
            add_pickup          = 0
            add_dropoff         = config.NUMBER_OF_STATIONS_LEFT_SIDE

        elif Point(start).within(poly_right):
            # assign pickup station
            available_pickup    = stations_right
            available_dropoff   = stations_left
            add_pickup          = config.NUMBER_OF_STATIONS_RIGHT_SIDE
            add_dropoff         = 0


        # VARIANT A - MINIMIZE ROUTE ON LAND
        # closest_pickup_index    = int(distance.cdist([start], available_pickup, 'euclidean').argmin(axis=1))
        # assigned_pickup_coord   = available_pickup[closest_pickup_index]
        # pickup_station          = closest_pickup_index + add_pickup
        # closest_dropoff_index   = int(distance.cdist([start], available_dropoff, 'euclidean').argmin(axis=1))
        # assigned_dropoff_coord  = available_dropoff[closest_dropoff_index]
        # dropoff_station         = closest_dropoff_index + add_dropoff

        # VARIANT B - MINIMIZE ROUTE ON WATER
        # generate distance matrix from request start to available pickup_stations
        start_to_pickup_dm = distance.cdist([start], available_pickup, 'euclidean')
        # sort size distances by index (ascending)
        start_to_pickup_sorted = np.argsort(start_to_pickup_dm)
        # take the first three indices of the sorted distances --> three closest stations to start
        closest_three_pickup = start_to_pickup_sorted[0][0:3]
        # get coordinates of three closest stations
        choice_of_pickup = [available_pickup[index] for index in closest_three_pickup]
        # generate distance matrix from destination to the choice of pickups
        closest_pickup_index \
            = closest_three_pickup[int(distance.cdist([destination], choice_of_pickup, 'euclidean').argmin(axis=1))]
        assigned_pickup_coord = available_pickup[closest_pickup_index]
        pickup_station = closest_pickup_index + add_pickup

        # ANALOGOUS PROCESS FOR DROPOFF-STATIONS
        pickup_to_dropoff_dm    = distance.cdist([available_pickup[closest_pickup_index]], available_dropoff, 'euclidean')
        pickup_to_dropoff_dist  = np.argsort(pickup_to_dropoff_dm)
        closest_three           = pickup_to_dropoff_dist[0][0:3]
        choice_of_dropoff       = [available_dropoff[index] for index in closest_three]
        dropoff_index  \
            = closest_three[int(distance.cdist([destination], choice_of_dropoff, 'euclidean').argmin(axis=1))]
        assigned_dropoff_coord  = available_dropoff[dropoff_index]
        dropoff_station         = dropoff_index + add_dropoff

        # GET VALUES
        distance_to_pickup = round(GD(start, assigned_pickup_coord).meters, 2)
        time_to_pickup = int((distance_to_pickup / config.BIKING_VELOCITY_IN_METER) * 60)
        distance_from_dropoff = round(GD(destination, assigned_dropoff_coord).meters, 2)
        time_from_dropoff = int((distance_from_dropoff / config.BIKING_VELOCITY_IN_METER) * 60)  # time = distance/velocity

        """
        The minimum ferry time is the travel time required when traveling by ferry without any waiting times or stops.
        The ferry-route consists of three sub-routes:
            (initial start to assigned pickup-station)
            + (direct travel time from assigned pickup-station to assigned dropoff-station)
            + (assigned dropoff-station to destination)
            = min. ferry time
        """
        min_ferry_time = time_to_pickup + time_from_dropoff + config.TIME_MATRIX[pickup_station][dropoff_station] + config.d_SERVICE_TIMES

        results.append([pickup_station, dropoff_station, assigned_pickup_coord, assigned_dropoff_coord,
                        distance_to_pickup, distance_from_dropoff, time_to_pickup, time_from_dropoff, min_ferry_time])

    return results

def requests_to_dataframe(results):
        results_df = pd.DataFrame(results,
                                  columns=['pickup_station', 'dropoff_station', 'pickup_coord', 'dropoff_coord',
                                           'distance_to_pickup', 'distance_from_dropoff', 'time_to_pickup',
                                           'time_from_dropoff', 'min_ferry_time'])
        return results_df


def run():
    initial_requests        = get_initial_requests(config.INITIAL_PAX_REQUESTS)
    assigned_requests       = assign_stations(initial_requests)  # 2.  weißt jedem Request stationen zu: r = [[1,4], [6,3]..]
    assigned_requests_df    = requests_to_dataframe(assigned_requests)
    concat_assigned_stations_to_requests_csv(assigned_requests_df)

    # todo draw_points_request(r1, 'green')



if __name__ == '__main__':
    run()
