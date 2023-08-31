import csv
import random
import pandas as pd
from shapely.geometry import Polygon, Point
from Ferry_Application_.helper import google_maps_operations
import path

# CONSTANTS
API_KEY = 'AIzaSyCRFoABLyG0iRNdu66Um06U1mh4R_ARqfQ'
GOOGLE_URL = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
call_gmaps = google_maps_operations.GoogleMaps(API_KEY, GOOGLE_URL)

parameter = path.read_parameter()
n = path.n_NUMBER_OF_PASSENGERS


# METHODS
def get_polygone(path):
    points = []
    with open(path, 'r') as station_locations:
        reader = csv.reader(station_locations, delimiter=',')
        for row in reader:
            points.append((float(row[0]), float(row[1])))
    return points


def generate_coordinates(poly, number_of_requests):
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
    input-parameter: 2 Listen, die Koordinaten von jeweils einer Seite des Flusses enthält.
    So wird sicher gestellt, dass jede Anfrage eine Flussquerung inkludiert.
    """
    requests = []
    for start, destination in list(zip(coordinates_right, coordinates_left)):
        requests.append([start, destination])
    """
    Shuffle requests, damit nicht alle Anfragen in die selbe Richtung gehen:
    """
    for i in range(len(requests)):
        random.shuffle(requests[i])


    """
    create request data frame 
    """

    """
      Für jedes Paar Koordinaten ( = request) wird die konventionelle Route berechnet und eine Abfahrtszeit
      innerhalb der Zeitfensterns (0 und der in den settings festgelegten maximalen departure time) generiert.
      Erst dann wird die Anfrage als neue Reihe auf die CSV-Datei geschrieben.
    """
    final_request = []
    for start, destination in requests:
        conv = compute_conventional_route(start, destination)  # returns array (time, distance)
        final_request.append(
            [start[0], start[1], destination[0], destination[1], random.randint(0, parameter['departure']), conv[0], conv[1]])

    return final_request



def compute_conventional_route(start, destination):
    print('computing conventional routes.. ')

    origin = "%s %s" % (start)
    dest = "%s %s" % (destination)
    mode = 'bicycling'

    yo = call_gmaps.latlon_to_address(start[0], start[1])
    time_distance = call_gmaps.route_time_distance(origin, dest, mode)

    return time_distance

def requests_to_dataframe(final_requests):
    request_df = pd.DataFrame(final_requests,
                              columns=['start_lat', 'start_lon', 'destination_lat', 'destination_lon',  'departure',
                                       'conventional_t (in min)',
                                       'conventional_d (in km)'])
    return request_df


def requests_to_csv(request_df):
    request_df.to_csv(path.INITIAL_PAX_REQUESTS, index=None)
    print('*** ', n, ' initial requests  written to ', path.INITIAL_PAX_REQUESTS, ' ***')




def run():
    right_coordinates   = generate_locations_in_polygone(path.DEMAND_POLYGON_RIGHT, number_of_requests=n)
    left_coordinates    = generate_locations_in_polygone(path.DEMAND_POLYGON_LEFT, number_of_requests=n)
    requests            = generate_requests(right_coordinates, left_coordinates)
    requests_df         = requests_to_dataframe(requests)

    requests_to_csv(requests_df)

if __name__ == '__main__':
    run()
