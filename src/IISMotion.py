from functools import cached_property
from src.city.Map import Map
from src.city.MapZone import MapZone
from src.city.ZoneType import ZoneType
from src.city.grid.MapGrid import MapGrid
from src.movement.ActorCollection import ActorCollection

from src.common.CommonFunctions import CommonFunctions
from src.common.FrontendServer import FrontendServer
from src.common.tools.ImagePrinter import ImagePrinter

from src.common.SimulationClock import *
import json


class IISMotion:

    def __init__(self, map: Map, guiEnabled=False, gridRows=10,
                 secondsPerTick=1,
                 locationLoggingEnabled=False,
                 ):
        '''
        Highest level class of simulated model
        @param guiEnabled: True/False enable GUI updates to JS frontend
        @param radius: import map in radius from location
        @param location: center location of map that will be imported
        '''

        self.com = CommonFunctions()
        self.movableCollectionsSet = {}
        self.guiEnabled = guiEnabled
        self.imagePrinter = ImagePrinter()
        self.secondsPerTick = secondsPerTick
        self.locationLoggingEnabled = locationLoggingEnabled

        self.map = map
        self.mapGrid = map.mapGrid
        if self.mapGrid is None:
            self.mapGrid = MapGrid.fromMap(map, gridRows, self.movableCollectionsSet)
            self.map.setMapGrid(self.mapGrid)
        else:
            self.mapGrid.actorCollectionsSet = self.movableCollectionsSet

    @cached_property
    def frontend(self):
        if self.guiEnabled:
            return FrontendServer()
        return None

    def setSecondsPerTick(self, secondsPerTick):
        self.secondsPerTick = secondsPerTick

    def addActorCollection(self, collection: ActorCollection) -> None:
        self.movableCollectionsSet[collection.name] = collection

    def createActorCollection(self, name, ableOfMovement, movementStrategy) -> ActorCollection:
        collection = ActorCollection(name, self.map, ableOfMovement, movementStrategy, self.mapGrid,
                                     self.secondsPerTick)
        self.movableCollectionsSet[name] = collection
        return collection

    def getActorCollection(self, name) -> ActorCollection:
        return self.movableCollectionsSet[name]

    # def stepCollections(self, names):
    #     newDay = updateSimulationClock(self.secondsPerTick)
    #     for name in names:
    #         collection = self.movableCollectionsSet[name]
    #         if (collection.ableOfMovement):
    #             collection.step()
    #             collection.logMovement(newDay)
    #     self.sendUpdateToFrontend()

    def stepAllCollections(self, newDay):
        for key, collection in self.movableCollectionsSet.items():
            if (collection.ableOfMovement):
                # print("Performing step on collection: ", key)
                collection.step(newDay)
            if (self.locationLoggingEnabled):
                collection.logMovement(newDay)
        self.sendUpdateToFrontend()

    def addMapZone(self, name, zoneType: ZoneType, probability, lacationsPolygon: []):
        self.map.addMapZone(MapZone(name, zoneType, probability, lacationsPolygon, self.map))

    def sendUpdateToFrontend(self):
        if (self.guiEnabled):
            data = {}
            data["type"] = "FeatureCollection"

            features = []
            for key, collection in self.movableCollectionsSet.items():
                features = features + collection.getFeaturesGeoJson()

            data["features"] = features

            if (len(features) > 0):
                json_data = json.dumps(data)
                self.frontend.addMessageToQueue(json_data)
                # self.com.appendToFile("json", json_data)

    def printHeatmapOfAllCollections(self, filename):
        self.imagePrinter.save2DArrayAsPicture(filename, self.mapGrid.getCountPerCell())

    def printHeatmapOfCollections(self, filename, collectionNames):
        self.imagePrinter.save2DArrayAsPicture(filename, self.mapGrid.getCountPerCell(collectionNames))


    def logActorsToNearestActorOfCollection(self, movableCollectionNames, toObjectsOfCollectionNamed, distThreshold):
        movables= []
        for name in movableCollectionNames:
            for key, movable in self.getActorCollection(name).actorSet.items():
                movables.append(movable)


        for actor in movables:
            location = actor.getLocation()
            closest, distance = self.mapGrid.getClosestActorAndDistanceFrom(2, [toObjectsOfCollectionNamed], location)
            if(distance<=distThreshold):
                closest.incrementNearPlaceablesCounter()

        for key, actor in self.getActorCollection(toObjectsOfCollectionNamed).actorSet.items():
            actor.savePlaceableCounterToLog()