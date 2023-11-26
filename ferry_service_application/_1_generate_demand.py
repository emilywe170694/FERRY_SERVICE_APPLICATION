import csv
import random
import pandas as pd
from shapely.geometry import Polygon, Point
from helper import google_maps_operations
import config

# CONSTANTS
API_KEY = config.GOOGLE_MAPS_API_KEY
GOOGLE_URL = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
call_gmaps = google_maps_operations.GoogleMaps(API_KEY, GOOGLE_URL)

n = config.n_NUMBER_OF_PASSENGERS

def get_polygone(path):
    """
    Method to read polygon points from a CSV file
    :param path: file path to polygone coordinates
    :return: array of vertices
    """
    points = []
    with open(path, 'r') as station_locations:
        reader = csv.reader(station_locations, delimiter=',')
        for row in reader:
            points.append((float(row[0]), float(row[1])))
    return points


def generate_coordinates(poly, number_of_requests):
    """
    Method to generate random coordinates within a polygon
    :param poly: shapely Polygone object created previously read polygone vertices
    :param number_of_requests: number of random points per polygone
    :return: array of random points
    """
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    while len(points) < number_of_requests:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if random_point.within(poly):
            points.append((random_point.x, random_point.y))
    return points


def generate_locations_in_polygone(file_path, number_of_requests):
    poly = Polygon(get_polygone(file_path))
    coord = generate_coordinates(poly, number_of_requests)
    return coord


def generate_requests(coordinates_right, coordinates_left):
    """
    1. Matches 1 coordinate (left) with 1 coordinate (right) and returns list paired coordinates  = PC.
        -> Each pair of coordinates in PC represents a start  and a destination of a request.
    2. Elements in PC are randomly shuffled to prevent requests from all going in the same direction
    3. Conventional Route for each request is obtained via Google Maps API
    4. Departure time for each request is randomly generated within time span (0, dt_DEPARTURE_TIME_SPAN)
    5. Requests are written to CSV (_1_initial_pax_requests.csv)

    :param coordinates_right: n random points in right polygone
    :param coordinates_left: n random points in left polygone
    :return: array of passenger_requests = [[start_lat, start_lon, dest_lat, dest_lon, departure, conventional_time, conventionel_distance], ...]
    """
    requests = []
    for start, destination in list(zip(coordinates_right, coordinates_left)):
        requests.append([start, destination])
    """
    Shuffle requests to prevent requests from all going in the same direction
    """
    for i in range(len(requests)):
        random.shuffle(requests[i])

    pax_requests = []
    for start, destination in requests:
        conv_t_d        = compute_conventional_route(start, destination)  # returns array (time, distance)
        departure   = random.randint(0, config.dt_DEPARTURE_TIME_SPAN)
        conv_arrival = departure + conv_t_d[0]
        pax_requests.append(
            [start[0], start[1], destination[0], destination[1], departure, conv_t_d[0], conv_arrival, conv_t_d[1]])

    return pax_requests

def compute_conventional_route(start, destination):
    print('computing conventional routes.. ')

    origin = "%s %s" % (start)
    dest = "%s %s" % (destination)
    mode = 'bicycling'

    time_distance = call_gmaps.route_time_distance(origin, dest, mode)
    return time_distance

def get_arrival_time(start, traveltime):
    sum_result = [a + b for a, b in zip(start, traveltime)]
    return sum_result

def requests_to_dataframe(final_requests):
    request_df = pd.DataFrame(final_requests,
                              columns=['start_lat', 'start_lon', 'destination_lat', 'destination_lon',  'departure',
                                       'conventional_t (in min)', 'conventional_arrival',
                                       'conventional_d (in km)'])
    return request_df


def requests_to_csv(request_df):
    request_df.to_csv(config.INITIAL_PAX_REQUESTS, index=None)
    print('*** ', n, ' initial requests  written to ', config.INITIAL_PAX_REQUESTS, ' ***')




def run():
    poly_right_side             = Polygon(get_polygone(config.DEMAND_POLYGON_RIGHT))
    coordinates_right_poly      = generate_coordinates(poly_right_side, number_of_requests=n)

    poly_left_side              = Polygon(get_polygone(config.DEMAND_POLYGON_LEFT))
    coordinates_left_poly       = generate_coordinates(poly_left_side, number_of_requests=n)

    requests                    = generate_requests(coordinates_right_poly, coordinates_left_poly)
    requests_df                 = requests_to_dataframe(requests)

    requests_to_csv(requests_df)


if __name__ == '__main__':
    run()
