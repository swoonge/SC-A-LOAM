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

        self.marker_pub = rospy.Publisher('/keypointMatchingDisplay', MarkerArray, queue_size=1)
        self.Graph_pub = rospy.Publisher('/keypointMatchingGraphDisplay', MarkerArray, queue_size=1)

        self.globalKeypointsPC = o3d.geometry.PointCloud
        self.localKeypointsPC = o3d.geometry.PointCloud
        self.globalDescriptor = np.zeros(shape=(1,135))
        self.localDescriptor = np.zeros(shape=(1,135))

        self.closest_indices = [0]
        self.closest_indices_3rd = [[0, 0, 0]]
        self.distances = []
        self.row_indices = []
        self.col_indices = []
    
        self.mutexGraph = threading.Lock()
        self.mutexClosest = threading.Lock()

        self.threshold = 20.0

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
            # distances = squareform(pdist(Keypoint))

            # # Create a graph with NetworkX
            # self.Gglobal.clear()

            # # Add nodes to the graph
            # for i in range(Keypoint.shape[0]):
            #     self.Gglobal.add_node(i, pos=Keypoint[i])

            # # Add edges based on distance threshold
            # for i in range(Keypoint.shape[0]):
            #     for j in range(i + 1, Keypoint.shape[0]):
            #         if distances[i, j] < self.threshold:
            #             self.Gglobal.add_edge(i, j, weight=distances[i, j])

    def localHandler(self, localMapMsg):
        with self.mutexGraph:
            self.localDescriptor = np.zeros(shape=(1,135))
            self.localKeypointsPC = orh.rospc_to_o3dpc(localMapMsg.keypoints)
            for i in range(localMapMsg.size):
                self.localDescriptor = np.append(self.localDescriptor, rnm.to_numpy_f64(localMapMsg.descriptors.descriptor[i]).reshape(1, 135), axis=0)
            self.localDescriptor = np.delete(self.localDescriptor, 0, axis=0)
            Keypoint = np.asarray(self.localKeypointsPC.points)
        
            # # Calculate pairwise Euclidean distances
            # distances = squareform(pdist(Keypoint))

            # # Create a graph with NetworkX
            # self.Glocal.clear()

            # # Add nodes to the graph
            # for i in range(Keypoint.shape[0]):
            #     self.Glocal.add_node(i, pos=Keypoint[i])

            # # Add edges based on distance threshold
            # for i in range(Keypoint.shape[0]):
            #     for j in range(i + 1, Keypoint.shape[0]):
            #         if distances[i, j] < self.threshold:
            #             self.Glocal.add_edge(i, j, weight=distances[i, j])

    def matchingKeypoints(self):
        print("[matchingLocalGloabl] global map: ", self.globalKeypointsPC, np.shape(self.globalDescriptor), " | local map: ", self.localKeypointsPC, np.shape(self.localDescriptor))
        with self.mutexGraph:
            distances = cdist(self.localDescriptor, self.globalDescriptor)
        distances = np.where(~np.isnan(distances), distances, np.inf)
        with self.mutexGraph:
            self.distances = distances
            row_indices, col_indices = np.where(distances <= 0.023)
            # print(self.row_indices, self.col_indices)
            self.closest_indices = np.argmin(distances, axis=1)
            self.closest_indices_3rd = np.argsort(distances, axis=1, kind='quicksort')[:, :1]
            self.matchingGraph(row_indices, col_indices)

    def matchingGraph(self, row_indices, col_indices):
        if len(row_indices) > 2:
            matchedGlobalPC = np.asarray(self.globalKeypointsPC.points)#[col_indices]
            matchedLocalPC = np.asarray(self.localKeypointsPC.points)#[row_indices]
            # matchedGolbalDec = np.asarray(self.globalDescriptor[col_indices].points)
            # matchedLocalDec = np.asarray(self.localDescriptor[row_indices].points)

            # Calculate pairwise Euclidean distances
            distancesG = squareform(pdist(matchedGlobalPC))

            # Create a graph with NetworkX
            self.Gglobal.clear()

            # Add nodes to the graph
            for i in range(matchedGlobalPC.shape[0]):
                self.Gglobal.add_node(i, pos=matchedGlobalPC[i])

            # Add edges based on distance threshold
            for i in range(matchedGlobalPC.shape[0]):
                for j in range(i + 1, matchedGlobalPC.shape[0]):
                    if distancesG[i, j] < self.threshold:
                        self.Gglobal.add_edge(i, j, weight=distancesG[i, j])

            # isolated_nodesG = [node for node in self.Gglobal.nodes if self.Gglobal.degree(node) == 0]
            # self.Gglobal.remove_nodes_from(isolated_nodesG)

            distancesL = squareform(pdist(matchedLocalPC))

            # Create a graph with NetworkX
            self.Glocal.clear()

            # Add nodes to the graph
            for i in range(matchedLocalPC.shape[0]):
                self.Glocal.add_node(i, pos=matchedLocalPC[i])

            # Add edges based on distance threshold
            for i in range(matchedLocalPC.shape[0]):
                for j in range(i + 1, matchedLocalPC.shape[0]):
                    if distancesL[i, j] < self.threshold:
                        self.Glocal.add_edge(i, j, weight=distancesL[i, j])

            # isolated_nodesL = [node for node in self.Glocal.nodes if self.Glocal.degree(node) == 0]
            # self.Glocal.remove_nodes_from(isolated_nodesL)

            AG = nx.to_numpy_array(self.Gglobal)
            AL = nx.to_numpy_array(self.Glocal)

            nG = np.array([np.shape(AG)[0]])
            nL = np.array([np.shape(AL)[0]])

            num_nodesL = np.shape(AL)[0]
            num_nodesG = np.shape(AG)[0]

            GL = nx.from_numpy_array(AL)
            GG = nx.from_numpy_array(AG)

            connG, edgeG = pygm.utils.dense_to_sparse(AG)
            connL, edgeL = pygm.utils.dense_to_sparse(AL)
            import functools
            gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1) # set affinity function
            K = pygm.utils.build_aff_mat(None, edgeG, connG, None, edgeL, connL, nG, None, nL, None, edge_aff_fn=gaussian_aff)

            # X = pygm.rrwm(K, nG, nL)
            # X = pygm.hungarian(X)

            marker_array = MarkerArray()

            iiii = 0

            for edge in GG.edges():
                marker = Marker()
                marker.header.frame_id = "/camera_init"
                marker.type = Marker.LINE_STRIP
                marker.id = iiii
                iiii = 1 + iiii
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05  # 엣지의 두께 조절
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0

                x1, y1, z1 = matchedGlobalPC[edge[0]]
                x2, y2, z2 = matchedGlobalPC[edge[1]]
                point1 = geometry_msgs.msg.Point(x=x1, y=y1, z=z1)
                point2 = geometry_msgs.msg.Point(x=x2, y=y2, z=z2)

                marker.points.append(point1)
                marker.points.append(point2)
                marker_array.markers.append(marker)
            print(iiii)
            
            
            for edge in GL.edges():
                marker = Marker()
                marker.header.frame_id = "/camera_init"
                marker.type = Marker.LINE_STRIP
                marker.id = iiii
                iiii = 1 + iiii
                marker.pose.orientation.w = 1.0
                marker.scale.x = 0.05  # 엣지의 두께 조절
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0

                x1, y1, z1 = matchedLocalPC[edge[0]]
                x2, y2, z2 = matchedLocalPC[edge[1]]
                point1 = geometry_msgs.msg.Point(x=x1, y=y1, z=z1)
                point2 = geometry_msgs.msg.Point(x=x2, y=y2, z=z2)

                marker.points.append(point1)
                marker.points.append(point2)
                marker_array.markers.append(marker)
            print(iiii)
            
            # for i in range(num_nodesL):
            #     j = np.argmax(X[i]).item()
            #     marker = Marker()
            #     marker.header.frame_id = "/camera_init"
            #     marker.type = Marker.LINE_STRIP
            #     marker.id = iiii
            #     iiii = 1 + iiii
            #     marker.pose.orientation.w = 1.0
            #     marker.scale.x = 0.05  # 엣지의 두께 조절
            #     marker.color.a = 1.0
            #     marker.color.r = 0.0
            #     marker.color.g = 0.0
            #     marker.color.b = 1.0

            #     x1, y1, z1 = matchedGlobalPC[i]
            #     x2, y2, z2 = matchedLocalPC[j]
            #     point1 = geometry_msgs.msg.Point(x=x1, y=y1, z=z1)
            #     point2 = geometry_msgs.msg.Point(x=x2, y=y2, z=z2)
            #     # print(marker)

            #     marker.points.append(point1)
            #     marker.points.append(point2)
                # marker_array.markers.append(marker)
            # print(iiii)
            self.Graph_pub.publish(marker_array)

    def pubMatchingDisplay(self):
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

                        
                        #     else:
                        #         line_marker.scale.x = 0.05
                        #         line_marker.color.r = 1.0 #random.random()
                        #         line_marker.color.g = 0.0
                        #         line_marker.color.b = 0.0
                        #         line_marker.color.a = 0.3
                        # else:
                        #     continue
                        if self.distances[idx, nidx] > 0.025:
                            continue
                            line_marker.scale.x = 0.05
                            line_marker.color.r = 1.0 #random.random()
                            line_marker.color.g = 0.2
                            line_marker.color.b = 0.2
                            line_marker.color.a = 0.9

                        elif self.distances[idx, nidx] < 0.024:
                            # print(idx, nidx)
                            if math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) < 8.0:
                                line_marker.scale.x = 0.5
                                line_marker.color.r = 0.0 #random.random()
                                line_marker.color.g = 0.0
                                line_marker.color.b = 1.0
                                line_marker.color.a = 1.0
                            else:
                                if random.random() < 0.8:
                                    continue
                                line_marker.scale.x = 0.05
                                line_marker.color.r = 0.0 #random.random()
                                line_marker.color.g = 0.0
                                line_marker.color.b = 0.0
                                line_marker.color.a = 1.0                              
                        else:
                            continue
                            line_marker.scale.x = 0.2
                            line_marker.color.r = 0.0 #random.random()
                            line_marker.color.g = 1.0
                            line_marker.color.b = 1.0
                            line_marker.color.a = 0.9

                        # Add the points to the marker
                        line_marker.points.append(point1)
                        line_marker.points.append(point2)

                        # Add the line marker to the array
                        # marker_array.markers.append(line_marker)
                self.marker_pub.publish(marker_array)
                        # print(marker_array)


def main():
    graphLC = GraphMatchingLoopCloser()
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        rate.sleep()
        graphLC.matchingKeypoints()
        graphLC.pubMatchingDisplay()

if __name__ == "__main__":
    rospy.init_node('graphMatching',anonymous=True)
    main()