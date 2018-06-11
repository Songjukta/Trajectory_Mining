import random
import copy
import nltk
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from math import sin, cos, sqrt, atan2, radians
from random import randint
from pymining import itemmining
from pymining import seqmining
#import seaborn as sns

# approximate radius of earth in km
R = 6373.0

def lat_long_dist(X, Y):
    lat1 = radians(X[0])
    lon1 = radians(X[1])
    lat2 = radians(Y[0])
    lon2 = radians(Y[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2+cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R*c*1000
    return distance

def lat_long_to_x(lt, ln):
    dx = (ln-point_matrix[0][1])*40000*math.cos((point_matrix[0][0]+lt)*math.pi/360)/360
    if dx < 1:
        dx = dx*1000
    else:
        dx = 0
    return dx

def lat_long_to_y(lt, ln):
    dy = (point_matrix[0][0]-lt)*40000/360
    if dy < 1:
        dy = dy*1000
    else:
        dy = 0
    return dy

trajectory_df = pd.read_csv('tracks.csv', names=['id', 'id_android', 'speed', 'time', 'distance', 'rating', 'rating_bus', 'rating_weather', 'car_or_bus', 'linha'])

#print trajectory_df['car_or_bus']

point_df = pd.read_csv('trackspoints.csv', names=['id', 'latitude', 'longitude', 'track_id', 'time'])

point_lat_long_df = point_df[['latitude', 'longitude']]

#print point_df['time']

point_matrix = point_lat_long_df.as_matrix()

#print lat_long_dist(point_matrix[0], point_matrix[2])

point_matrix_xy = [ [lat_long_to_x(pt[0], pt[1]), lat_long_to_y(pt[0], pt[1])] for pt in point_matrix]

#print point_matrix_xy[0],  point_matrix_xy[2]



for n_clusters in range(50, 51):
    kclusterer = nltk.cluster.kmeans.KMeansClusterer(n_clusters, distance = lat_long_dist, repeats=2, avoid_empty_clusters=True)
    cluster_labels = kclusterer.cluster(point_matrix, assign_clusters=True)
    print 'Done with kmeans clustering'

cluster_sample_dict = {}

for cluster_label in range(n_clusters):
       indices = [i for i, val in enumerate(cluster_labels) if val == cluster_label]
       random_index = random.choice(indices)
       cluster_sample_dict[str(cluster_label)] = random_index
       print point_matrix_xy[random_index]

print '$$$'
for pt in cluster_sample_dict:
    for pt2 in cluster_sample_dict:
        #print pt, pt2
        #print cluster_sample_dict[pt], cluster_sample_dict[pt2]
        #print point_matrix_xy[cluster_sample_dict[pt]]
        #print tuple(point_matrix_xy[cluster_sample_dict[pt]]), tuple(point_matrix_xy[cluster_sample_dict[pt2]])
        a = np.array(point_matrix_xy[cluster_sample_dict[pt]])
        b = np.array(point_matrix_xy[cluster_sample_dict[pt2]])
        dist = np.linalg.norm(a-b)
        
        


x = [pt[0] for pt in point_matrix_xy]
y = [pt[1] for pt in point_matrix_xy]

#print  cluster_labels[0], cluster_labels[1], cluster_labels[2]

x_sample = []
y_sample = []
z_sample = []

for custer_label in range(n_clusters):
    indices = [i for i, val in enumerate(cluster_labels) if val == custer_label]
    print 'Cluster', str(cluster_label), ':', len(indices)
    for n in range(min(20, len(indices))):
        random_index = random.choice(indices)
        x_sample.append(x[random_index])
        y_sample.append(y[random_index ])
        z_sample.append(cluster_labels[random_index ])

plt.scatter(x_sample, y_sample, c= z_sample, s = 30, edgecolors='none')
plt.show()

transactions = []
for track in point_df.track_id.unique():
    index_list = point_df .index[point_df['track_id'] == track].tolist()
    label_sequence = ['C '+str(cluster_labels[sp]) for sp in index_list]
    transactions.append(label_sequence)
    
relim_input = itemmining.get_relim_input(transactions)
report = itemmining.freq_seq_enum(relim_input, min_support=len(trajectory_df)*0.05)
print report
#transactions = (('a', 'b', 'c'), ('b'), ('a'), ('a', 'c', 'd'), ('b', 'c'), ('b', 'c'))
#relim_input = itemmining.get_relim_input(transactions)
#report = itemmining.relim(relim_input, min_support=2)
#print report





