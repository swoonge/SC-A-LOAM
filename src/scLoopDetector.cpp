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

std::mutex mKeyFrameBuf;
std::mutex mLocalMapBuf;

std::queue<pcl::PointCloud<PointType>::Ptr> keyFrameQue;
std::queue<pcl::PointCloud<PointType>::Ptr> localMapPCQue;
std::vector<pcl::PointCloud<PointType>::Ptr> keypointsVector;
pcl::VoxelGrid<PointType> downSizeFilterScancontext;

SCManager scManager;
double scDistThres, scMaximumRadius;

ros::Publisher pubLCdetectResult;

int KeyFrameNum = 0;
int LocalMapNum = 0;
int LocalMapIdxRange = 6;

void KeyFrameDSHandler(const sensor_msgs::PointCloud2::ConstPtr &_thisKeyFrame) {
    // ROSmsg 타입의 pointcloud를 pcl::PointCloud 로 변환
    pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*_thisKeyFrame, *thisKeyFrameDS);

    // 들어온 keyFrame을 keyFrameQue에 push
    mKeyFrameBuf.lock();
    keyFrameQue.push(thisKeyFrameDS);
    mKeyFrameBuf.unlock();
    KeyFrameNum++;
}

// void DetectTriggerforLCHandler(const std_msgs::Int64::ConstPtr &_data) {
//     int keyframePosesSize = _data->data;
//     // std::cout << "[DetectTriggerforLCHandler] keyframePosesSize: " << keyframePosesSize << std::endl;
//     mKeyFrameBuf.lock();
//     if( int(keyframePosesSize) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
//     {   
//         mKeyFrameBuf.unlock();
//         return;
//     }
//     auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
//     mKeyFrameBuf.unlock();
//     // std::cout << detectResult.first << std::endl;
//     int SCclosestHistoryFrameID = detectResult.first;
//     if( SCclosestHistoryFrameID != -1 ) { 
//         const int prev_node_idx = SCclosestHistoryFrameID;
//         const int curr_node_idx = keyframePosesSize - 1; // because cpp starts 0 and ends n-1
//         // cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

//         aloam_velodyne::LCPair pair;
//         pair.a_int = prev_node_idx;
//         pair.b_int = curr_node_idx;
//         pubLCdetectResult.publish(pair);
//     }
// }

void LocalMapHandler(const sensor_msgs::PointCloud2::ConstPtr &_thisKeyFrame){
    // ROSmsg 타입의 pointcloud를 pcl::PointCloud 로 변환
    pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*_thisKeyFrame, *thisKeyFrameDS);

    // 들어온 keyFrame을 keyFrameQue에 push
    mLocalMapBuf.lock();
    localMapPCQue.push(thisKeyFrameDS);
    mLocalMapBuf.unlock();
}


void ScancontextProcess(void) {
    float loopClosureFrequency = 30.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok()) {
        rate.sleep();
        if (keyFrameQue.size() > 0) {
            mKeyFrameBuf.lock();
            auto frontData = keyFrameQue.front();
            keyFrameQue.pop();
            mKeyFrameBuf.unlock();

            scManager.makeAndSaveScancontextAndKeys(*frontData);
            auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
            int SCclosestHistoryFrameID = detectResult.first;
            if( SCclosestHistoryFrameID != -1 ) { 
                const int prev_node_idx = SCclosestHistoryFrameID;
                const int curr_node_idx = KeyFrameNum - 1; // because cpp starts 0 and ends n-1
                // cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

                aloam_velodyne::LCPair pair;
                pair.a_int = prev_node_idx;
                pair.b_int = curr_node_idx;
                pubLCdetectResult.publish(pair);
            }
        }
    }
}

void KeypointDetectionProcess(void) {
    return;
    // scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);
}

float ISS_SalientRadius = 10;
float ISS_NonMaxRadius = 6;
float ISS_Gamma21 = 0.9;
float ISS_Gamma23 = 0.9;
int ISS_MinNeighbors = 10;
int Local_map_idx = 6;
int recentIdxprocessed = Local_map_idx;
float Local_map_boundary = 25.0;
pcl::PointXYZ lastCenterPoint = pcl::PointXYZ(0, 0, 0);

// void keyPointDetection(void) {
//     float loopClosureFrequency = 10.0; // can change 
//     ros::Rate rate(loopClosureFrequency);
//     while (ros::ok()){
//         rate.sleep();
//         if (recentIdxprocessed < KeyFrameNum) {
//             // cout << recentIdxprocessed << endl;
//             surround_keypoint_detection(recentIdxprocessed);
//             recentIdxprocessed++;
//         }
//         else {
//             pcl::PointCloud<PointType>::Ptr nonkeypoints(new pcl::PointCloud<PointType>);
//         }
//     }
//     sensor_msgs::PointCloud2 KeypointMapMsg;
//     pcl::toROSMsg(*laserCloudLocal, KeypointMapMsg);
//     KeypointMapMsg.header.frame_id = "/camera_init";
//     pubKeyPointMap.publish(KeypointMapMsg);
// }

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
    ros::Subscriber subKeyLocalMap = nh.subscribe<sensor_msgs::PointCloud2>("/LGMLocalMap", 100, LocalMapHandler);
	// ros::Subscriber subTrigger = nh.subscribe<std_msgs::Int64>("/DetectTriggerforLC", 100, DetectTriggerforLCHandler);

	pubLCdetectResult = nh.advertise<aloam_velodyne::LCPair>("/LCdetectResult", 100);

    // std::thread keyPointDetection {keyPointDetection};
    //     // 첫 번째 함수를 실행하는 스레드 생성
    std::thread threadSC(ScancontextProcess);

    // // 두 번째 함수를 실행하는 스레드 생성
    // std::thread threadKP(KeypointDetectionProcess);

    // // 메인 스레드에서 대기
    // thread1.join();
    // thread2.join();

 	ros::spin();

	return 0;
}
