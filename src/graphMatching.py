#!/usr/bin/env python3
# -- coding: utf-8 --
import sys
import rospy
import numpy as np
import open3d as o3d
import math
import random
from scipy.spatial.distance import cdist, pdist, squareform
import pygmtools as pygm
import matplotlib
matplotlib.use('Agg')  # Use Agg backend (non-interactive)
import matplotlib.pyplot as plt # for plotting

from matplotlib.patches import ConnectionPatch # for plotting matching result
import networkx as nx # for plotting graphs

from open3d_ros_helper import open3d_ros_helper as orh
from ros_np_multiarray import ros_np_multiarray as rnm
from sklearn.decomposition import PCA

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray 
from aloam_velodyne.msg import *
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import threading

def nanargsort(arr):
    mask = ~np.isnan(arr)
    return np.argsort(np.where(mask, arr, np.inf), axis=1, kind='quicksort')

class GraphMatchingLoopCloser():
    def __init__(self) -> None:
        rospy.Subscriber("/keypointGlobalGraph", MapAndDescriptors, self.globalHandler)
        rospy.Subscriber("/keypointLocalGraph", MapAndDescriptors, self.localHandler)

        self.marker_pub = rospy.Publisher('/keypointMatchingDisplay', MarkerArray, queue_size=10)

        self.globalKeypointsPC = o3d.geometry.PointCloud
        self.localKeypointsPC = o3d.geometry.PointCloud
        self.globalDescriptor = np.zeros(shape=(1,135))
        self.localDescriptor = np.zeros(shape=(1,135))

        self.closest_indices = [0]
        self.closest_indices_3rd = [[0, 0, 0]]
    
        self.mutexGraph = threading.Lock()
        self.mutexClosest = threading.Lock()

        self.threshold = 15.0

        self.Gglobal = nx.Graph()
        self.Glocal = nx.Graph()

        # self.pygm.set_backend('numpy')

    def globalHandler(self, globalMapMsg):
        with self.mutexGraph:
            self.globalKeypointsPC = orh.rospc_to_o3dpc(globalMapMsg.keypoints)
            self.globalDescriptor = np.zeros(shape=(1,135))
            for i in range(globalMapMsg.size):
                self.globalDescriptor = np.append(self.globalDescriptor, rnm.to_numpy_f64(globalMapMsg.descriptors.descriptor[i]).reshape(1, 135), axis=0)
            self.globalDescriptor = np.delete(self.globalDescriptor, 0, axis=0)
            Keypoint = np.asarray(self.globalKeypointsPC.points)
        
        # Calculate pairwise Euclidean distances
        distances = squareform(pdist(Keypoint))

        # Create a graph with NetworkX
        self.Gglobal.clear()

        # Add nodes to the graph
        for i in range(Keypoint.shape[0]):
            self.Gglobal.add_node(i, pos=Keypoint[i])

        # Add edges based on distance threshold
        for i in range(Keypoint.shape[0]):
            for j in range(i + 1, Keypoint.shape[0]):
                if distances[i, j] < self.threshold:
                    self.Gglobal.add_edge(i, j, weight=distances[i, j])

    def localHandler(self, localMapMsg):
        with self.mutexGraph:
            self.localDescriptor = np.zeros(shape=(1,135))
            self.localKeypointsPC = orh.rospc_to_o3dpc(localMapMsg.keypoints)
            for i in range(localMapMsg.size):
                self.localDescriptor = np.append(self.localDescriptor, rnm.to_numpy_f64(localMapMsg.descriptors.descriptor[i]).reshape(1, 135), axis=0)
            self.localDescriptor = np.delete(self.localDescriptor, 0, axis=0)
            Keypoint = np.asarray(self.localKeypointsPC.points)
        
        # Calculate pairwise Euclidean distances
        distances = squareform(pdist(Keypoint))

        # Create a graph with NetworkX
        self.Glocal.clear()

        # Add nodes to the graph
        for i in range(Keypoint.shape[0]):
            self.Glocal.add_node(i, pos=Keypoint[i])

        # Add edges based on distance threshold
        for i in range(Keypoint.shape[0]):
            for j in range(i + 1, Keypoint.shape[0]):
                if distances[i, j] < self.threshold:
                    self.Glocal.add_edge(i, j, weight=distances[i, j])

    def matchingLocalGloabl(self):
        print("[matchingLocalGloabl] global map: ", self.globalKeypointsPC, np.shape(self.globalDescriptor), " | local map: ", self.localKeypointsPC, np.shape(self.localDescriptor))
        with self.mutexGraph:
            distances = cdist(self.localDescriptor, self.globalDescriptor)
        distances = np.where(~np.isnan(distances), distances, np.inf)
        with self.mutexGraph:
            self.closest_indices = np.argmin(distances, axis=1)
            self.closest_indices_3rd = np.argsort(distances, axis=1, kind='quicksort')[:, :3]

    def pubMatchingDisplay(self):
        if len(self.closest_indices) > 2:
            with self.mutexGraph:
                A2 = nx.to_numpy_array(self.Gglobal)
                A1 = nx.to_numpy_array(self.Glocal)

                n1 = np.array([np.shape(A1)[0]])
                n2 = np.array([np.shape(A2)[0]])

                num_nodes2 = np.shape(A2)[0]
                num_nodes1 = np.shape(A1)[0]

                G2 = nx.from_numpy_array(A2)
                pos2 = nx.spring_layout(self.Gglobal)
                pos1 = nx.spring_layout(self.Glocal)

                G1 = nx.from_numpy_array(A1)
                color1 = ['#FF5733' for _ in range(num_nodes1)]
                color2 = ['#FF5733' for _ in range(num_nodes2)]

                conn1, edge1 = pygm.utils.dense_to_sparse(A1)
                conn2, edge2 = pygm.utils.dense_to_sparse(A2)
                import functools
                gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001) # set affinity function
                K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

                X = pygm.sm(K, n1, n2)
                X = pygm.hungarian(X)
                
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.title('RRWM Soft Matching Matrix')
                plt.imshow(X, cmap='Blues')
                plt.subplot(1, 2, 2)
                plt.title('Ground Truth Matching Matrix')
                # plt.imshow(X_gt, cmap='Blues')
                plt.savefig('/home/vision/catkin_ws/pyg1.png')

                plt.figure(figsize=(8, 4))
                # plt.suptitle(f'IPFP Matching Result (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
                ax1 = plt.subplot(1, 2, 1)
                plt.title('Subgraph 1')
                plt.gca().margins(0.4)
                nx.draw_networkx(G1, pos=pos1, node_color=color1)
                ax2 = plt.subplot(1, 2, 2)
                plt.title('Graph 2')
                nx.draw_networkx(G2, pos=pos2, node_color=color2)
                for i in range(num_nodes1):
                    j = np.argmax(X[i]).item()
                    con = ConnectionPatch(xyA=pos1[i], xyB=pos2[j], coordsA="data", coordsB="data",
                                        axesA=ax1, axesB=ax2, color="green")
                    plt.gca().add_artist(con)
                plt.savefig('/home/vision/catkin_ws/pyg2.png')
                print(np.shape(X))

                marker_array = MarkerArray()

                localKeypoint = np.asarray(self.localKeypointsPC.points)
                globalKeypoint = np.asarray(self.globalKeypointsPC.points)
                print(np.shape(globalKeypoint))

                # for idx, nidx in enumerate(self.closest_indices):
                for idx in range(num_nodes1):
                    nidx = np.argmax(X[idx]).item()
                    line_marker = Marker()
                    line_marker.header.frame_id = "/camera_init"  # Change 'base_link' to your desired frame
                    line_marker.id = idx
                    line_marker.type = Marker.LINE_STRIP
                    line_marker.action = Marker.ADD
                    line_marker.pose.orientation.w = 1.0

                    x1, y1, z1 = localKeypoint[idx]
                    x2, y2, z2 = globalKeypoint[nidx]

                    if math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) < 5.0:
                        line_marker.scale.x = 0.5
                        line_marker.color.r = 0.0 #random.random()
                        line_marker.color.g = 0.0
                        line_marker.color.b = 1.0
                        line_marker.color.a = 1.0
                    else:
                        line_marker.scale.x = 0.1
                        line_marker.color.r = 1.0 #random.random()
                        line_marker.color.g = 0.0
                        line_marker.color.b = 0.0
                        line_marker.color.a = 0.8
                    
                    point1 = Point(x=x1, y=y1, z=z1)
                    point2 = Point(x=x2, y=y2, z=z2)

                    # Add the points to the marker
                    line_marker.points.append(point1)
                    line_marker.points.append(point2)

                    # Add the line marker to the array
                    marker_array.markers.append(line_marker)
                
                    self.marker_pub.publish(marker_array)
                    print("pub")

    def pubMatchingDisplay2(self):
        if len(self.closest_indices) > 2:
            with self.mutexGraph:
                
                marker_array = MarkerArray()

                localKeypoint = np.asarray(self.localKeypointsPC.points)
                globalKeypoint = np.asarray(self.globalKeypointsPC.points)

                i = 0
                for idx, nnidx in enumerate(self.closest_indices_3rd):
                    for nidx in nnidx:
                        line_marker = Marker()
                        line_marker.header.frame_id = "/camera_init"  # Change 'base_link' to your desired frame
                        line_marker.id = i
                        i = 1 + i
                        line_marker.type = Marker.LINE_STRIP
                        line_marker.action = Marker.ADD
                        line_marker.pose.orientation.w = 1.0

                        x1, y1, z1 = localKeypoint[idx]
                        x2, y2, z2 = globalKeypoint[nidx]

                        point1 = Point(x=x1, y=y1, z=z1)
                        point2 = Point(x=x2, y=y2, z=z2)

                        if math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) < 5.0:
                            line_marker.scale.x = 0.5
                            line_marker.color.r = 0.0 #random.random()
                            line_marker.color.g = 0.0
                            line_marker.color.b = 1.0
                            line_marker.color.a = 1.0
                        else:
                            line_marker.scale.x = 0.02
                            line_marker.color.r = 1.0 #random.random()
                            line_marker.color.g = 0.0
                            line_marker.color.b = 0.0
                            line_marker.color.a = 0.4

                        # Add the points to the marker
                        line_marker.points.append(point1)
                        line_marker.points.append(point2)

                        # Add the line marker to the array
                        marker_array.markers.append(line_marker)
                        self.marker_pub.publish(marker_array)
                        # print(marker_array)


def main():
    graphLC = GraphMatchingLoopCloser()
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        rate.sleep()
        graphLC.matchingLocalGloabl()
        graphLC.pubMatchingDisplay2()

if __name__ == "__main__":
    rospy.init_node('graphMatching',anonymous=True)
    main()