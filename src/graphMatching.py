#!/usr/bin/env python3
# -- coding: utf-8 --
import sys
import rospy
import numpy as np
import open3d as o3d

from open3d_ros_helper import open3d_ros_helper as orh
from ros_np_multiarray import ros_np_multiarray as rnm

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64MultiArray 
from aloam_velodyne.msg import *

class GraphMatchingLoopCloser():
    def __init__(self) -> None:
        rospy.Subscriber("/keypointGlobalGraph", MapAndDescriptors, self.globalHandler)
        rospy.Subscriber("/keypointLocalGraph", MapAndDescriptors, self.localHandler)

        self.globalKeypointsPC = o3d.geometry.PointCloud
        self.localKeypointsPC = o3d.geometry.PointCloud
        self.globalDescriptor = np.zeros(shape=(1,135))
        self.localDescriptor = np.zeros(shape=(1,135))
        # print(np.shape(self.globalDescriptor))


    def globalHandler(self, globalMapMsg):
        self.globalKeypointsPC = orh.rospc_to_o3dpc(globalMapMsg.keypoints)
        self.globalDescriptor = np.zeros(shape=(1,135))
        print("[------------------------------------------------------]")
        for i in range(globalMapMsg.size):
            self.globalDescriptor = np.append(self.globalDescriptor, rnm.to_numpy_f64(globalMapMsg.descriptors.descriptor[i]).reshape(1, 135), axis=0)
        self.globalDescriptor = np.delete(self.globalDescriptor, 0, axis=0)
        print("recive global map size: ", globalMapMsg.size)
        print("recive global map: ", self.globalKeypointsPC)
        print("recive global descriptor: ", np.shape(self.globalDescriptor))

    def localHandler(self, localMapMsg):
        self.localDescriptor = np.zeros(shape=(1,135))
        self.localKeypointsPC = orh.rospc_to_o3dpc(localMapMsg.keypoints)
        for i in range(localMapMsg.size):
            self.localDescriptor = np.append(self.localDescriptor, rnm.to_numpy_f64(localMapMsg.descriptors.descriptor[i]).reshape(1, 135), axis=0)
        self.localDescriptor = np.delete(self.localDescriptor, 0, axis=0)

def main():
    graphLC = GraphMatchingLoopCloser()
    rate = rospy.Rate(10)
    i = 0
    rospy.spin()

    while not rospy.is_shutdown():
        # print("test pass", i)
        # i += 1
        rospy.spin()
        # rate.sleep()

if __name__ == "__main__":
    rospy.init_node('graphMatching',anonymous=True)
    main()