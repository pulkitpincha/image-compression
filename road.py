# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:25:45 2023

@author: stimp
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
import math
from sklearn.cluster import KMeans

#loading the image
image_1 = iio.imread(uri="C:/Users/ravte/Downloads/OPSM322_Homework-1/OPSM322_Homework-1/road2.jpg")
plt.imshow(image_1)

image_1.shape
height = image_1.shape[0]
width = image_1.shape[1]
rgb_val = image_1.shape[2]

x0 = np.array((image_1[:, :, :]).reshape((height*width, rgb_val))) 
x = np.array((image_1[:, :, :]).reshape((height*width, rgb_val)), dtype=float)
x = x.T
m = x.shape[1]

#plotting the image
def plot_cluster_1(clust_num, kmeans_list, x0_arg, cno_arr_arg):
    i =  clust_num
    ff = kmeans_list
    uu = ff
    if uu.shape == 0:
        cno_arr_arg[i] = 0
        return
    else:
       xxx = x0_arg[uu, :]
       K = xxx.shape[0]
       COLS = round( math.sqrt(K) )
       ROWS = math.ceil(K / COLS)
       image_1 = np.ones((ROWS, COLS, 3))
       for j in range(0, xxx.shape[0]):
           r = math.floor(j / COLS)
           c = np.mod(j , COLS)
           image_1[r,c,:] = xxx[j,:].reshape(1,1,3)
       return image_1
   

#finding the number of unique colors
unique_colors = np.unique(image_1.reshape(-1, image_1.shape[2]), axis=0)
num_colors = unique_colors.shape[0]
print(f"Number of unique colors in the image: {num_colors}")


#running K-Means
cno = 70
cno_arr = np.ones((cno))
kmeans = KMeans(init="random", n_clusters=cno, n_init=20, random_state=0).fit(x0)
l_lab = kmeans.labels_
centroids = kmeans.cluster_centers_

image_1_comp = np.ones((height, width, 3))
pixel_id = 0

for r in range(height):
    for c in range(width):
        clust_id = l_lab[pixel_id]
        cent_val = centroids[clust_id]
        image_1_comp[r,c,:] = cent_val.reshape(1,1,3)
        pixel_id = pixel_id + 1
plt.imshow(image_1_comp.astype('uint8'))

for i in range(0, cno):
    cl_i = np.where(l_lab==i)
    cl_i = np.asarray(cl_i)
    cl_i = cl_i[0]
    i_image = plot_cluster_1(i, cl_i, x0, cno_arr)
    if cno_arr[i] == 0:
        print("The cluster id {} is empty".format(i+1))
    else:
        fig = plt.figure()
        plt.imshow(i_image.astype('uint8'))
        

distortions = []


cluster_range = range(1, 70)  

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, init="random", n_init=20, random_state=0).fit(x0)
    distortions.append(kmeans.inertia_)

#K-means with 75 clusters
cno_1 = 75
cno_1_arr = np.ones((cno_1))
kmeans = KMeans(init="random", n_clusters=cno_1, n_init=20, random_state=0).fit(x0)
l_lab = kmeans.labels_
centroids = kmeans.cluster_centers_

image_1_comp = np.ones((height, width, 3))
pixel_id = 0

for r in range(height):
    for c in range(width):
        clust_id = l_lab[pixel_id]
        cent_val = centroids[clust_id]
        image_1_comp[r,c,:] = cent_val.reshape(1,1,3)
        pixel_id = pixel_id + 1
plt.imshow(image_1_comp.astype('uint8'))

for i in range(0, cno_1):
    cl_i = np.where(l_lab==i)
    cl_i = np.asarray(cl_i)
    cl_i = cl_i[0]
    i_image = plot_cluster_1(i, cl_i, x0, cno_1_arr)
    if cno_1_arr[i] == 0:
        print("The cluster id {} is empty".format(i+1))
    else:
        fig = plt.figure()
        plt.imshow(i_image.astype('uint8'))



#Taking out compression ratio
original_file_size = m*24 #Number of pixels
print("Original File Size:",original_file_size,"bytes")

compressed_file_size = (24*cno_1) + ((math.log2(cno_1))*m)
print("Compressed File Size:",compressed_file_size,"bytes")

compression_ratio = original_file_size / compressed_file_size
print("Compression Ratio:",compression_ratio)


