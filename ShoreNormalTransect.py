#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:34:11 2023

@author: wrenteria
To read data from:
https://catalog.data.gov/dataset/tiger-line-shapefile-2019-nation-u-s-coastline-national-shapefile
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.io.shapereader import Reader
from pyproj import Geod
import skimage.transform as transform
from sklearn.cluster import MeanShift
from sklearn.linear_model import LinearRegression
import os

path =os.getcwd()
fn = path + '/tl_2019_us_coastline'+'/'+'tl_2019_us_coastline.shp'

waterbody = "Pacific"

# Read a point in the area of interest - aoi
# also couald read the mean coordinate from the database/grd of the aoi
Pm = (-124.191985,41.748202)


Radius = 0.01125          # RBF for clustering i.e 0.015
length_transect = 1000  # meters
step = 100              # step between point in the transect


################### DO NOT CHANGE #########################################
# Read the shapefile
reader = Reader(fn)
#acces to records at Pacific, create a list of records with that attribute
pacific = [seg for seg in reader.records() if seg.attributes["NAME"]==waterbody]
#Check the segment of the coastline that contains the mean coordindates of the aoi
for i in range(len(pacific)):
    b = pacific[i].bounds
    if b[0]<Pm[0]<b[2]:
        if b[1]<Pm[1]<b[3]:
            aoi = pacific[i].geometry.xy

coastline = np.column_stack((aoi[0],aoi[1]))

# To compute distace, azimut over the geoide
geod = Geod(ellps='GRS80')

# To cluster similar segments in the coastline
coastline_clustering = MeanShift(bandwidth=Radius).fit(coastline)
coastline = np.column_stack((coastline,coastline_clustering.labels_))
Nseg = int(coastline[:,2].max()) # Number max of segments

SNT = dict()    # To save the transects

for s in range(Nseg):
    c = coastline[coastline[:,2]==float(s)]
    N = int(len(c)/2) # middle point
    # To get the orientation of the segment of shoreline -slope/regression
    model = LinearRegression()
    model.fit(c[:,0].reshape((-1,1)), c[:,1])
    nlat = model.predict(c[:,0].reshape((-1,1)))
    #azimuth = orientation
    az,za,d = geod.inv(c[0,0],nlat[0],c[-1,0],nlat[-1])
       
    # To get the origen point for the Shore Normal Transect
    x0 = c[N,0]
    y0 = c[N,1]
    
    # length dimension of transect
    L = np.arange(step,length_transect,step)
    x0 = x0*np.ones_like(L)
    y0 = y0*np.ones_like(L)
    az = az*np.ones_like(L)
    
    # To get coordinates of extreme point in the transect normal
    snt_x, snt_y, backaz = geod.fwd(x0,y0, az+90, L)
    
    # Save transect to a dictionary
    SNT[s]=np.column_stack((snt_x,snt_y))

np.save('ShoreNormalTransects.npy',SNT)

# To plot
# Load the transects
snt = np.load('ShoreNormalTransects.npy',allow_pickle=True).item()

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.plot(coastline[:,0],coastline[:,1],'k')

for i in range(len(snt)):
    trs = snt[i]
    ax.plot(trs[:,0],trs[:,1],'b')
ax.gridlines(draw_labels=True)
plt.savefig('shorenormaltransect.png')
