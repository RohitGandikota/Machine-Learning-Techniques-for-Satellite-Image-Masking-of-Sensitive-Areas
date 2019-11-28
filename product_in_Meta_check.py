# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:49:26 2019

@author: Rohit Gandikota
"""
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

big = [[0, 100], [90,100], [100, 30], [0,0], [0,100]]
small = [[0, 100], [90,100], [100, 30], [0,0], [0,100]]
#small = [[50,50], [100,100], [60, 40], [0,0]]

def checkProductInMeta(meta, workorder):
    meta = Polygon(meta)
    for point in workorder:
        p = Point(point[0], point[1])
        if meta.contains(p):
            pass
        else:
            print('Workorder not lying inside meta !!!')
            return 0
checkProductInMeta(big, small)