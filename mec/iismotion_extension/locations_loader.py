# FROM mec
from src.common.Location import Location
from src.placeable.Placeable import Placeable
from src.common.CommonFunctions import CommonFunctions
from src.city.Map import Map
import os
import json
from typing import List


class LocationsLoader:
    """respnsible for loading/saving locations from/to file"""

    def __init__(self, path: str = 'cellCache/smallcellCache/') -> None:
        self.com = CommonFunctions()
        self.path = path

    def generate_locations(
        self, *,
        map: Map,
        count: int,
        min_radius
    ) -> List[Location]:
        locations = []
        for _ in range(0, count):
            location = map.getRandomNode()
            while (
                self.com.getShortestDistanceFromLocations(locations, location)
                < min_radius
            ):
                location = map.getRandomNode()

            x, y = map.mapGrid.getGridCoordinates(location)
            location.setGridCoordinates(x, y)
            locations.append(location)

        return locations

    def generate_random_locations(
        self, *,
        map: Map,
        count: int,
        min_radius
    ) -> List[Location]:
        locations = []
        for _ in range(0, count):
            location = map.getRandomPoint()
            while (
                self.com.getShortestDistanceFromLocations(locations, location)
                < min_radius
            ):
                location = map.getRandomPoint()

            x, y = map.mapGrid.getGridCoordinates(location)
            location.setGridCoordinates(x, y)
            locations.append(location)

        return locations

    def load_locations_from_file(
        self,
        map: Map,
        filename: str
    ) -> List[Location]:
        locations = []
        if os.path.exists(self.path + filename):
            with open(self.path + filename) as cachedData:
                cached = json.load(cachedData)

            for item in cached:
                location = Location()
                location.longitude = item['longitude']
                location.latitude = item['latitude']

                x, y = map.mapGrid.getGridCoordinates(location)
                location.setGridCoordinates(x, y)
                locations.append(location)
        else:
            raise ValueError(
                'Failed to load Locations from given file', filename)

        return locations

    def store_placeables_locations_into_file(
        self,
        list: List[Placeable],
        filename: str
    ):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        data = []
        for placeable in list:
            item = {}
            location = placeable.getLocation()
            item['latitude'] = location.getLatitude()
            item['longitude'] = location.getLongitude()
            data.append(item)

        with open(self.path + filename, 'w+') as outfile:
            json.dump(data, outfile)
