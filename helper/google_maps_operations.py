import googlemaps
import requests


class GoogleMaps:
    # def __init__(self,
    #              api_key='AIzaSyCRFoABLyG0iRNdu66Um06U1mh4R_ARqfQ',
    #              url='https://maps.googleapis.com/maps/api/distancematrix/json?'):
    #     self.api_key = api_key
    #     self.url = url

    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def latlon_to_address(self, lat, lon):
        geocode_result = googlemaps.Client(key=self.api_key).reverse_geocode((float(lat), float(lon)))
        return geocode_result[1]["formatted_address"]

    # returns total distance in km when entering two formatted addresses:
    # mode must be d = driving, b = bicycling, w = walking
    def route_distance_km(self, origin, destination, mode):  # mode = WALKING / DRIVING / BICYCLING / TRANSIT
        r = requests.get(
            self.url + "origins=" + origin + "&destinations=" + destination + "&mode=" + mode + "&key=" + self.api_key)
        # hier müsste ich noch mode of transport eingeben können --> &mode=bicycling --> &mode=walking
        distance = r.json()["rows"][0]["elements"][0]["distance"]["text"]
        print(f'total distance from {origin} to {destination} when {mode} is {distance}')
        return distance

    # returns total time in minutes when entering two formatted addresses:
    def route_time_min(self, origin, destination,
                       mode):  # returns total time in minutes when entering to formatted addresses
        r = requests.get(
            self.url + "origins=" + origin + "&destinations=" + destination + "&mode=" + mode + "&key=" + self.api_key)
        # hier müsste ich noch mode of transport eingeben können --> &mode=bicycling --> &mode=walking
        time = r.json()["rows"][0]["elements"][0]["duration"]["value"]
        # print(f'total travel time from {origin} to {destination} when {mode} is {time}')
        return time

    def route_time_distance(self, origin, destination, mode):  # returns total time in minutes when entering to formatted addresses

        r = requests.get(
            self.url + "origins=" + origin + "&destinations=" + destination + "&mode=" + mode + "&key=" + self.api_key)
        print('json-----', r,
              self.url + "origins=" + origin + "&destinations=" + destination + "&mode=" + mode + "&departure_time=343641500" + "&key=" + self.api_key)
        # hier müsste ich noch mode of transport eingeben können --> &mode=bicycling --> &mode=walking
        time = r.json()["rows"][0]["elements"][0]["duration"]["value"]
        distance = r.json()["rows"][0]["elements"][0]["distance"]["value"]
        return int(round(time / 60)), distance
