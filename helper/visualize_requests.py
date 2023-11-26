import sqlite3
import sys

import pandas as pd
sys.path.append("ferry_service_application")
from ferry_service_application import config
import matplotlib.pyplot as plt
import descartes
import geopandas as gpd
from branca.element import IFrame
from shapely.geometry import Point, Polygon
import folium as fl
import webbrowser
from IPython.display import IFrame
from flask import Flask, render_template
import sqlite3 as sl

requests = config.FINAL_PAX_REQUESTS

def get_stations(file_path):
    data_stations = pd.read_csv(file_path, header=0)
    stations = []
    for index, row in data_stations.iterrows():
        stations.append((row['lat'], row['lon']))
    return stations

def read_column(file, column_name):
    result = []
    result_df = pd.read_csv(file, header=0)
    for index, row in result_df.iterrows():
        result.append(
            row[column_name])
    return result

def format_column(column_list):
    result = []
    for col in column_list:
        p = col.strip("()").split(', ')
        conv = (float(p[0]), float(p[1]))
        result.append(conv)
    return result

app = Flask('ferry_app')

# Erzeugen Sie eine Liste von Koordinaten (Beispiel)
gmaps_api = config.GOOGLE_MAPS_API_KEY


station_coordinates = get_stations(config.STATIONS_COORD)

request_start       = list(zip(read_column(requests, 'start_lat'),
                               read_column(requests, 'start_lon')))
request_dest        = list(zip(read_column(requests, 'destination_lat'),
                               read_column(requests, 'destination_lon')))
request_pickup      = format_column(read_column(requests, 'pickup_coord'))
request_dropoff     = format_column(read_column(requests, 'dropoff_coord'))

#------workaround: # SINGLE INSTANCE ANALYSIS ---------
start = [(51.26999717159475, 6.709110597661836),
(51.2550047279871, 6.75348971507969),
(51.2647314391159, 6.6869342724008805),
(51.23984402756247, 6.707946212791647),
(51.26621960413196, 6.677457614432723),
(51.24892904830692, 6.729220438557718),
(51.26967327815839, 6.678443264998846),
(51.24540089782636, 6.773376917978012),
(51.25643354274168, 6.758545285397966),
(51.25761911216152, 6.761765022730833)]

destination = [(51.23662691128537, 6.73381485348564),
(51.26570544862253, 6.681464101694137),
(51.25124534289744, 6.767211443103097),
(51.26322547096908, 6.726391823252583),
(51.25935670860031, 6.751813642180755),
(51.25495625799842, 6.767175117876903),
(51.265530748179344, 6.716365585039993),
(51.26745634040095, 6.675276148810144),
(51.23861088954723, 6.69328671633926),
(51.24457031999798, 6.690487609424903)]

stations = [(20, 4),
(27, 9),
(3, 21),
(6, 23),
(3, 21),
(9, 27),
(2, 20),
(30, 13),
(27, 9),
(28, 10)]

PW = []
DW = []

for item in stations:
    PW.append(station_coordinates[item[0]])
    DW.append(station_coordinates[item[1]])

#------END workaround ---------

@app.route('/')

# SINGLE INSTANCE ANALYSIS
# def map():
#     return render_template('map.html',
#                            station_coordinates = station_coordinates,
#                            gmaps_api = gmaps_api,
#                            request_start = start,
#                            request_dest = destination,
#                            request_pickup = PW,
#                            request_dropoff = DW)

def map():
    return render_template('map.html',
                           station_coordinates = station_coordinates,
                           gmaps_api = gmaps_api,
                           request_start = request_start,
                           request_dest = request_dest,
                           request_pickup = request_pickup,
                           request_dropoff = request_dropoff)

if __name__ == '__main__':
    app.run(port=8080, debug=True)
