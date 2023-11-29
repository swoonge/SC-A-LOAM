#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

#include <tf/transform_datatypes.h>
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

#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>

#include <ros/ros.h>
#include <std_msgs/Int64.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <aloam_velodyne/LCPair.h>
#include <aloam_velodyne/LocalMapAndPose.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "scancontext/Scancontext.h"

using namespace gtsam;

std::mutex mKeyFrameBuf;
std::mutex mLocalMapBuf;
std::vector<std::pair<int, int>> ICPPassedPair;
std::queue<pcl::PointCloud<PointType>::Ptr> keyFrameQue;
std::queue<pcl::PointCloud<PointType>::Ptr> localMapPclQue;
std::queue<geometry_msgs::Pose> localMapPoseQue;
std::vector<std::pair<pcl::PointCloud<PointType>::Ptr, geometry_msgs::Pose>> keypointsVector;
pcl::VoxelGrid<PointType> downSizeFilterScancontext;

SCManager scManager;
double scDistThres, scMaximumRadius;

ros::Publisher pubLCdetectResult, pubKeyPointResult;

int KeyFrameNum = 0;
int LocalMapNum = 0;
int LocalMapIdxRange = 6;

float LocalMapBoundary = 30.0;
float ISS_SalientRadius = 10;
float ISS_NonMaxRadius = 6;
float ISS_Gamma21 = 0.9;
float ISS_Gamma23 = 0.9;
int ISS_MinNeighbors = 10;

int Local_map_idx = 6;
int recentIdxprocessed = Local_map_idx;
float Local_map_boundary = 25.0;
pcl::PointXYZ lastCenterPoint = pcl::PointXYZ(0, 0, 0);

Eigen::Affine3f rosPoseToEigenAffine(const geometry_msgs::Pose& rosPose) {
    Eigen::Affine3f eigenAffine;

    // Translation
    eigenAffine.translation() << rosPose.position.x, rosPose.position.y, rosPose.position.z;

    // Quaternion rotation
    Eigen::Quaternionf quaternion(
        rosPose.orientation.w,
        rosPose.orientation.x,
        rosPose.orientation.y,
        rosPose.orientation.z
    );
    eigenAffine.linear() = quaternion.toRotationMatrix();

    return eigenAffine;
}

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const geometry_msgs::Pose& rosPose)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = rosPoseToEigenAffine(rosPose);

    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

pcl::PointCloud<PointType>::Ptr global2local(const pcl::PointCloud<PointType>::Ptr &cloudIn, const geometry_msgs::Pose& rosPose)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = rosPoseToEigenAffine(rosPose);
    transCur = transCur.inverse();

    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }
    return cloudOut;
}

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

void LocalMapHandler(const aloam_velodyne::LocalMapAndPose::ConstPtr &_LocalMapAndPose){
    // ROSmsg 타입의 pointcloud를 pcl::PointCloud 로 변환
    pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(_LocalMapAndPose->point_cloud, *thisKeyFrameDS);

    mLocalMapBuf.lock();
    localMapPoseQue.push(_LocalMapAndPose->pose);
    // 들어온 keyFrame을 keyFrameQue에 push
    localMapPclQue.push(thisKeyFrameDS);
    mLocalMapBuf.unlock();
}

void ScancontextProcess(void) {
    float loopClosureFrequency = 30.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok()) {
        // If keyFrameQue have keyFrame data, pop out keyFrame.
        mKeyFrameBuf.lock();
        if (keyFrameQue.size() > 0) {
            auto frontData = keyFrameQue.front();
            keyFrameQue.pop();
            mKeyFrameBuf.unlock();

            // Make SC.
            scManager.makeAndSaveScancontextAndKeys(*frontData);

            // Search Loop by SC.
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
        else{
            mKeyFrameBuf.unlock();
        }
        rate.sleep();
    }
}

void keyPointDetection(void) {
    mLocalMapBuf.lock();
    auto localMapPcl = localMapPclQue.front();
    auto localMapPose = localMapPoseQue.front();
    localMapPclQue.pop();
    localMapPoseQue.pop();
    mLocalMapBuf.unlock();

    // 키포인트 추출
    pcl::PointCloud<PointType>::Ptr currkeypoints(new pcl::PointCloud<PointType>);

    // [] ISS keypoint 추출
    pcl::ISSKeypoint3D<PointType, PointType> detector;
    detector.setInputCloud(localMapPcl);
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
    detector.setSearchMethod(kdtree);
    // double resolution = computeCloudResolution(cloud);
    // Set the radius of the spherical neighborhood used to compute the scatter matrix.
    detector.setSalientRadius(ISS_SalientRadius * 0.2);
    // Set the radius for the application of the non maxima supression algorithm.
    detector.setNonMaxRadius(ISS_NonMaxRadius * 0.2);
    // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
    detector.setMinNeighbors(ISS_MinNeighbors);
    // Set the upper bound on the ratio between the second and the first eigenvalue.
    detector.setThreshold21(ISS_Gamma21);
    // Set the upper bound on the ratio between the third and the second eigenvalue.
    detector.setThreshold32(ISS_Gamma23);
    // Set the number of prpcessing threads to use. 0 sets it to automatic.
    detector.setNumberOfThreads(4);
    detector.compute(*currkeypoints);

    // // 바운더리 근처의 키포인트는 제거
    // pcl::PassThrough<PointType> pass;
    // pass.setInputCloud (currkeypoints);
    // pass.setFilterFieldName ("x");
    // pass.setFilterLimits (pclCenterPoint.x - (2*Local_map_boundary/3), pclCenterPoint.x + (2*Local_map_boundary/3));
    // pass.filter (*currkeypoints);

    // pass.setFilterFieldName ("y");
    // pass.setFilterLimits (pclCenterPoint.y - (2*Local_map_boundary/3), pclCenterPoint.y + (2*Local_map_boundary/3));
    // pass.filter (*currkeypoints);

    // pass.setFilterFieldName ("z");
    // pass.setFilterLimits (pclCenterPoint.z - 10.0, pclCenterPoint.z + 10.0);
    // pass.filter (*currkeypoints);

    cout << "CurrentKeyFrameNum:" << KeyFrameNum << " || KeypointsVector's size: " << keypointsVector.size() << " || Keypoint's num : " << currkeypoints->points.size() << endl;

    *currkeypoints = *global2local(currkeypoints, localMapPose);
    
    keypointsVector.push_back(std::make_pair(currkeypoints, localMapPose));

    sensor_msgs::PointCloud2 keyPointMsg;
    pcl::toROSMsg(*currkeypoints, keyPointMsg);
    keyPointMsg.header.frame_id = "/camera_init";
    pubKeyPointResult.publish(keyPointMsg);
}

void KeypointDetectionProcess(void) {
    float loopClosureFrequency = 30.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok()){
        mLocalMapBuf.lock();
        if (localMapPoseQue.size() > 0) {
            mLocalMapBuf.unlock();
            keyPointDetection();
        }
        else{
            mLocalMapBuf.unlock();
        }
    rate.sleep();
    }
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "alaserScDetector");
	ros::NodeHandle nh;

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor

    nh.param<float>("LocalMapBoundary", LocalMapBoundary, 30.0);
    nh.param<float>("ISS_SalientRadius", ISS_SalientRadius, 10.0);
	nh.param<float>("ISS_NonMaxRadius", ISS_NonMaxRadius, 6.0); 
    nh.param<float>("ISS_Gamma21", ISS_Gamma21, 0.9);
	nh.param<float>("ISS_Gamma23", ISS_Gamma23, 0.9);
    nh.param<int>("ISS_MinNeighbors", ISS_MinNeighbors, 50);
 
    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.3;
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);

	ros::Subscriber subKeyFrameDS = nh.subscribe<sensor_msgs::PointCloud2>("/KeyFrameDSforLC", 100, KeyFrameDSHandler);
    ros::Subscriber subKeyLocalMap = nh.subscribe<aloam_velodyne::LocalMapAndPose>("/LGMLocalMap", 100, LocalMapHandler);

	pubLCdetectResult = nh.advertise<aloam_velodyne::LCPair>("/LCdetectResult", 100);
    pubKeyPointResult = nh.advertise<sensor_msgs::PointCloud2>("/keyPointResult", 100);

    std::thread threadSC(ScancontextProcess);
    std::thread threadKP(KeypointDetectionProcess);

 	ros::spin();

	return 0;
}
