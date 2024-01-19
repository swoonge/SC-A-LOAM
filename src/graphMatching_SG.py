#!/usr/bin/env python3
# -- coding: utf-8 --
import sys
import rospy
import numpy as np
import open3d as o3d
import math
import random
from scipy.spatial.distance import cdist, pdist, squareform

import networkx as nx # for plotting graphs

# from open3d_ros_helper import open3d_ros_helper as orh
from sklearn.decomposition import PCA

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray 
from aloam_velodyne.msg import *
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import sensor_msgs.point_cloud2 as pc2

from models.superglue import SuperGlue
from models.utils import (AverageTimer, PickleStreamer, printImageSize,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

import threading

def nanargsort(arr):
    mask = ~np.isnan(arr)
    return np.argsort(np.where(mask, arr, np.inf), axis=1, kind='quicksort')

def ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message

        Returns:
            pcl.PointCloud_PointXYZRGB: PCL XYZI point cloud
            
    """
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        # points_list.append([data[0], data[1], data[2], data[3]])
        points_list.append([data[0], data[1], data[2]])

    # pcl_data = o3d.geometry.PointCloud()
    pcl_data = o3d.utility.Vector3dVector(points_list)

    return pcl_data 

def ros_to_np(ros_cloud):
    """ Converts a ROS PointCloud2 message to a np_data PointXYZ

        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message

        Returns:
            np_data: XYZ point cloud
            
    """
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        # points_list.append([data[0], data[1], data[2], data[3]])
        points_list.append([data[0], data[1], data[2], data[3]])

    # np_data = o3d.geometry.PointCloud()
    np_data = np.array(points_list)

    return np_data

class GraphMatchingLoopCloser():
    def __init__(self, opt) -> None:
        self.subTriger = 0
        self.RawDataLock = threading.Lock()

        rospy.Subscriber("/globalGraph", MapAndDescriptors, self.globalHandler)
        rospy.Subscriber("/localGraph", MapAndDescriptors, self.localHandler)

        self.marker_pub = rospy.Publisher('/keypointMatchingDisplay', MarkerArray, queue_size=1)
        self.Graph_pub = rospy.Publisher('/keypointMatchingGraphDisplay', MarkerArray, queue_size=1)

        self.globalKeypointsPC = o3d.geometry.PointCloud()
        self.localKeypointsPC = o3d.geometry.PointCloud()
        self.globalDescriptor = np.zeros(shape=(1,135))
        self.localDescriptor = np.zeros(shape=(1,135))

        self.device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
        superglueConfig = {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
        self.superglue = SuperGlue(superglueConfig).eval().to(self.device)
    #     {'keypoints': [tensor([[3517., 1245.], [ 991., 2575.], [2559.,  416.],..., [ 693.,  657.], [2652., 2098.], [1551., 2739.]], device='cuda:0')], 
    #      'scores': (tensor([0.5256, 0.5188, 0.4826,  ..., 0.0613, 0.0613, 0.0613], device='cuda:0'),), 
    #      'descriptors': [tensor([[ 0.0189,  0.0394,  0.0267,  ...,  0.0298,  0.0132,  0.0453], [-0.0463, -0.0875, -0.0942,  ..., -0.0002, -0.0103, -0.0149],
    #     [-0.0755,  0.0888, -0.0227,  ...,  0.0307, -0.0917,  0.0163],
    #     ...,
    #     [-0.0087,  0.0114,  0.1206,  ..., -0.0063, -0.1124, -0.0167],
    #     [ 0.0907, -0.1026,  0.0447,  ..., -0.0218,  0.0086, -0.0261],
    #     [ 0.1247, -0.0190,  0.0700,  ...,  0.0505,  0.0042,  0.0161]],
    #    device='cuda:0')], 
    #      'frame': array([[248, 248, 247, ..., 151, 151, 151],
    #    [248, 248, 247, ..., 151, 151, 151],
    #    [248, 248, 247, ..., 151, 151, 151],
    #    ...,
    #    [216, 216, 216, ..., 204, 204, 204],
    #    [216, 216, 215, ..., 204, 204, 204],
    #    [216, 216, 215, ..., 204, 204, 204]], dtype=uint8)}
        # 이렇게 딕셔너리로 만들어서 넣어주어야 함. 물론 device='cuda:0'에 attach 한 상태로
        # desc0, desc1 = data['descriptors0'], data['descriptors1']
        # kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        self.threshold = 20.0

    def globalHandler(self, globalMapMsg):
        self.subTriger += 1
        with self.RawDataLock:
            self.globalKeypointsPC = ros_to_np(globalMapMsg.keypoints)
            globalDescriptorCache = np.zeros(shape=(1,135))
            
            for i in range(globalMapMsg.size):
                globalDescriptorCache = np.append(globalDescriptorCache, np.array(globalMapMsg.descriptors.descriptor[i].data).reshape(1, 135), axis=0)
            globalDescriptorCache = np.delete(globalDescriptorCache, 0, axis=0)

            globalDescriptorCache2 = np.zeros((globalDescriptorCache.shape[0], 256))
            globalDescriptorCache2[:,:135] = globalDescriptorCache

            print(globalDescriptorCache2.shape)
            self.globalDescriptor = globalDescriptorCache2

    def localHandler(self, localMapMsg):
        self.subTriger += 1
        with self.RawDataLock:
            self.localKeypointsPC = ros_to_np(localMapMsg.keypoints)
            localDescriptorCache = np.zeros(shape=(1,135))

            for i in range(localMapMsg.size):
                localDescriptorCache = np.append(localDescriptorCache, np.array(localMapMsg.descriptors.descriptor[i].data).reshape(1, 135), axis=0)
            localDescriptorCache = np.delete(localDescriptorCache, 0, axis=0)

            localDescriptorCache2 = np.zeros((localDescriptorCache.shape[0], 256))
            localDescriptorCache2[:,:135] = localDescriptorCache
            self.localDescriptor = localDescriptorCache2

    def matchingKeypoints(self):
        if ((np.shape(self.globalKeypointsPC)[0]) < 1 or (np.shape(self.localKeypointsPC)[0] < 1)):
            return
        print("[matchingLocalGloabl] global map: ", len(self.globalKeypointsPC), np.shape(self.globalDescriptor), " | local map: ", len(self.localKeypointsPC), np.shape(self.localDescriptor))
        with self.RawDataLock:
            descriptorsTarget = torch.tensor(self.globalDescriptor).to(self.device)
            descriptorsLocal = torch.tensor(self.localDescriptor).to(self.device)
            keypointsTarget = torch.Tensor(self.globalKeypointsPC[:,:2]).to(self.device)
            keypointsLocal = torch.Tensor(self.localKeypointsPC[:,:2]).to(self.device)
            scoresTarget = torch.Tensor(self.globalKeypointsPC[:,3:]).to(self.device).reshape(-1).float()
            scoresLocal = torch.Tensor(self.localKeypointsPC[:,3:]).to(self.device).reshape(-1).float()
            
        data = {}
        data['keypoints0'] = keypointsTarget.float()#.reshape()
        data['keypoints1'] = keypointsLocal.float()
        data['descriptors0'] = descriptorsTarget.unsqueeze(0).permute(0, 2, 1).float()
        data['descriptors1'] = descriptorsLocal.unsqueeze(0).permute(0, 2, 1).float()
        data['scores0'] = scoresTarget/torch.max(scoresTarget)
        data['scores1'] = scoresLocal/torch.max(scoresTarget)
        
        xMin = 10000000.0
        xMax = -10000000.0
        yMin = 10000000.0
        yMax = -10000000.0
        for point in keypointsTarget:
            if point[0] < xMin: xMin = point[0]
            if point[0] > xMax: xMax = point[0]
            if point[1] < yMin: yMin = point[1]
            if point[1] > yMax: yMax = point[1]
        data['shape0'] = (xMax-xMin, yMax-yMin)

        xMin = 10000000.0
        xMax = -10000000.0
        yMin = 10000000.0
        yMax = -10000000.0
        for point in keypointsLocal:
            if point[0] < xMin: xMin = point[0]
            if point[0] > xMax: xMax = point[0]
            if point[1] < yMin: yMin = point[1]
            if point[1] > yMax: yMax = point[1]
        data['shape1'] = (xMax-xMin, yMax-yMin)

        print("---------[output of superglue]---------")
        pred = self.superglue(data)
        print(pred)
        print("---------------------------------------")

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

            # AG = nx.to_numpy_array(self.Gglobal)
            # AL = nx.to_numpy_array(self.Glocal)

            # nG = np.array([np.shape(AG)[0]])
            # nL = np.array([np.shape(AL)[0]])

            # num_nodesL = np.shape(AL)[0]
            # num_nodesG = np.shape(AG)[0]

            # GL = nx.from_numpy_array(AL)
            # GG = nx.from_numpy_array(AG)

            # connG, edgeG = pygm.utils.dense_to_sparse(AG)
            # connL, edgeL = pygm.utils.dense_to_sparse(AL)
            # import functools
            # gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1) # set affinity function
            # K = pygm.utils.build_aff_mat(None, edgeG, connG, None, edgeL, connL, nG, None, nL, None, edge_aff_fn=gaussian_aff)

            # X = pygm.rrwm(K, nG, nL)
            # X = pygm.hungarian(X)

            marker_array = MarkerArray()

            iiii = 0

            for edge in self.Gglobal.edges():
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
            
            
            for edge in self.Glocal.edges():
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
        if len(self.closest_indices_3rd) > 2:
            with self.mutexGraph:
                
                marker_array = MarkerArray()

                localKeypoint = np.asarray(self.localKeypointsPC.points)
                globalKeypoint = np.asarray(self.globalKeypointsPC.points)

                i = 0
                for idx, nnidx in enumerate(self.closest_indices_3rd):
                    # print("[nnidx]", nnidx)
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
                        # 0.025보다 거리가 멀면 빨강
                        if self.distances[idx, nidx] > 0.025:
                            line_marker.scale.x = 0.05
                            line_marker.color.r = 1.0 #random.random()
                            line_marker.color.g = 0.2
                            line_marker.color.b = 0.2
                            line_marker.color.a = 0.9

                        # 0.024보다 거리가 가까우면
                        elif self.distances[idx, nidx] < 0.024:
                            # print(idx, nidx)
                            # 그 중에서 실제 거리가 10m보다 가까우면 두꺼운 파랑
                            if math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) < 50.0:
                                line_marker.scale.x = 0.5
                                line_marker.color.r = 0.0 #random.random()
                                line_marker.color.g = 0.0
                                line_marker.color.b = 1.0
                                line_marker.color.a = 1.0
                            # 거리는 매칭되었지만 10m보다 먼 점이면 검정
                            else:
                                if random.random() < 0.8:
                                    continue
                                line_marker.scale.x = 0.05
                                line_marker.color.r = 0.0 #random.random()
                                line_marker.color.g = 0.0
                                line_marker.color.b = 0.0
                                line_marker.color.a = 1.0                              
                        else: # 0.24~0.25사이는 청록
                            # continue
                            line_marker.scale.x = 0.2
                            line_marker.color.r = 0.0 #random.random()
                            line_marker.color.g = 1.0
                            line_marker.color.b = 1.0
                            line_marker.color.a = 0.9

                        # Add the points to the marker
                        line_marker.points.append(point1)
                        line_marker.points.append(point2)

                        # Add the line marker to the array
                        marker_array.markers.append(line_marker)
                self.marker_pub.publish(marker_array)
                        # print(marker_array)


def main(opt):
    graphLC = GraphMatchingLoopCloser(opt)
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        rate.sleep()
        if (graphLC.subTriger >= 2):
            graphLC.matchingKeypoints()
        # graphLC.pubMatchingDisplay()

if __name__ == '__main__':
    rospy.init_node('graphMatching',anonymous=True)
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.pickle'],
        # '--image_glob', type=str, nargs='+', default=['*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.01, # 0.005
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20, # 20
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2, # 0.2
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    main(opt)