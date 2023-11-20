#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <std_msgs/Int64.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <aloam_velodyne/LCPair.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "scancontext/Scancontext.h"

std::mutex mKF;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
SCManager scManager;
double scDistThres, scMaximumRadius;

ros::Publisher pubLCdetectResult;

void KeyFrameDSHandler(const sensor_msgs::PointCloud2::ConstPtr &_thisKeyFrame)
{
    // std::cout << "[-] get new keyframe" << std::endl;
    pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*_thisKeyFrame, *thisKeyFrameDS);
    mKF.lock();
    scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);
    mKF.unlock();
}

void DetectTriggerforLCHandler(const std_msgs::Int64::ConstPtr &_data)
{
    
    int keyframePosesSize = _data->data;
    // std::cout << "[DetectTriggerforLCHandler] keyframePosesSize: " << keyframePosesSize << std::endl;
    mKF.lock();
    if( int(keyframePosesSize) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
    {   
        mKF.unlock();
        return;
    }
    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
    mKF.unlock();
    // std::cout << detectResult.first << std::endl;
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 ) { 
        const int prev_node_idx = SCclosestHistoryFrameID;
        const int curr_node_idx = keyframePosesSize - 1; // because cpp starts 0 and ends n-1
        // cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

        aloam_velodyne::LCPair pair;
        pair.a_int = prev_node_idx;
        pair.b_int = curr_node_idx;
        pubLCdetectResult.publish(pair);
    }
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "alaserScDetector");
	ros::NodeHandle nh;

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.3;
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);

	ros::Subscriber subKeyFrameDS = nh.subscribe<sensor_msgs::PointCloud2>("/KeyFrameDSforLC", 100, KeyFrameDSHandler);
	ros::Subscriber subTrigger = nh.subscribe<std_msgs::Int64>("/DetectTriggerforLC", 100, DetectTriggerforLCHandler);

	pubLCdetectResult = nh.advertise<aloam_velodyne::LCPair>("/LCdetectResult", 100);

 	ros::spin();

	return 0;
}
