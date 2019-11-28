# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:29:21 2019

@author: Rohit Gandikota
"""
import ogr
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from PIL import Image, ImageDraw
import gdal
import numpy as np
import fiona
from sklearn.preprocessing import PolynomialFeatures


#%% Linear regression with poly variable◘s


def ReadGridMask(poly_x, poly_y, shp_file_path):
    x = fiona.open(shp_file_path)
    for y in x[0]['geometry']['coordinates']:
        y = np.array(y)
        bub = y.copy()
        y[:,0] = bub[:,1]
        y[:,1] = bub[:,0]
        del(bub)
#        print(y)
        poly = PolynomialFeatures(20,include_bias=False)
        X_poly_test= poly.fit_transform(y)
        x_poly = (poly_x.predict(X_poly_test))
        y_poly = (poly_y.predict(X_poly_test))
        
    return np.vstack((x_poly,y_poly))

def get_poly_model(file_path):
    
    with open(file_path,'r') as f:
        lines = f.readlines()
    nrows = int(lines[0].split(':')[-1])
    ncols = int(lines[1].split(':')[-1])
    irows = int(lines[2].split(':')[-1])
    icols = int(lines[3].split(':')[-1])
    
    if nrows*ncols == len(lines)-5:
        scanpix = []
        latlon  = []
        for i in range(5, len(lines)):
            scanpix.append([((i-5)%ncols)*icols, ((i-5)//ncols)*irows])
            lat = float(lines[i].split(' ')[1])
            lon = float(lines[i].split(' ')[3])
            latlon.append([lat,lon])
        poly = PolynomialFeatures(20,include_bias=False)
        X_poly = poly.fit_transform(latlon)
        y = np.array(scanpix)
        poly_x = linear_model.LinearRegression(normalize=True)
        poly_y = linear_model.LinearRegression(normalize=True)
        model_poly_x=poly_x.fit(X_poly, y[:,0])
        model_poly_y=poly_y.fit(X_poly, y[:,1])
        return poly_x, poly_y
    
    else:
        print('Error in reading grid file: ' + file_path)
        raise Exception(' Error in Reading SLC Grid File')
        #return 0
    
def RPCMaskRasterWithShape(Raster, shape_file, file_path):
    poly_x, poly_y = get_poly_model(file_path) 
    
    Polygon1 = np.einsum('ij->ij', ReadGridMask(poly_x, poly_y, shp_file_path))
    Polygon = []
    for i in range(len(Polygon1.T)):
        Polygon.append((int(Polygon1[0,i]), int(Polygon1[1,i])))
#    Polygon.append((Polygon1[0,0], int(Polygon1[1,0])))
       
#    Polygon = np.array([[4000,0], [4000,4000], [0,4000], [0,0], [4000,0]])
    dataset=gdal.Open(Raster,gdal.GA_Update)
    for band in range(1,dataset.RasterCount+1):
        bandpointer=dataset.GetRasterBand(band)
        #print(bandpointer)
        data=bandpointer.ReadAsArray(0,0,dataset.RasterXSize,dataset.RasterYSize).astype(np.float)
        datay=Image.new('L',(dataset.RasterXSize,dataset.RasterYSize),color=1)
        ImageDraw.Draw(datay).polygon(Polygon, outline=0, fill = 0)
        data=np.array(datay)*data
        bandpointer.WriteArray(data)
        print('Done')

    dataset.FlushCache()
    bandpointer=None
    data=None
    dataset=None
    
#%%
file_path = 'E:\\Share\\19112\\19112_L1_SlantRange_grid.txt'
shp_file_path = 'E:\\Share\\19112\\test.shp'
Raster = 'E:\\Share\\19112\\scene_VV\\imagery_VV.tif'
RPCMaskRasterWithShape(Raster, shp_file_path, file_path)
