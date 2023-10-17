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
coordinates = [(51.22786415395434, 6.772850463594631)]
gmaps_api = config.GOOGLE_MAPS_API_KEY

station_coordinates = get_stations(config.STATIONS_COORD)
request_start       = list(zip(read_column(requests, 'start_lat'),
                               read_column(requests, 'start_lon')))
request_dest        = list(zip(read_column(requests, 'destination_lat'),
                               read_column(requests, 'destination_lon')))
request_pickup      = format_column(read_column(requests, 'pickup_coord'))
request_dropoff     = format_column(read_column(requests, 'dropoff_coord'))


print(station_coordinates)
print(request_pickup)
@app.route('/')
def map():
    return render_template('map.html',
                           coordinates=coordinates,
                           station_coordinates = station_coordinates,
                           gmaps_api = gmaps_api,
                           request_start = request_start,
                           request_dest = request_dest,
                           request_pickup = request_pickup,
                           request_dropoff = request_dropoff)

if __name__ == '__main__':
    app.run(port=8080, debug=True)




# area_of_operation = gpd.read_file('/Users/emilyjlw/PycharmProjects/DARP05/Ferry_Application_/helper/visualize/Shapefile-polygon.shp')
# area_of_operation.plot()
# area_of_operation.to_crs(epsg=4326).plot()
# plt.show()
