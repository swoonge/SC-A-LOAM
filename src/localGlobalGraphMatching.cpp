#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <tuple>
#include <pcl/io/vtk_lib_io.h>
#include <chrono>

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
#include <pcl/keypoints/harris_3d.h>
#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/normal_3d.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/console/time.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/features/shot.h>
#include <pcl/features/fpfh.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>

#include <pcl/point_types_conversion.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/rops_estimation.h>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>

#include <ros/ros.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <aloam_velodyne/LCPair.h>
#include <aloam_velodyne/LocalMapAndPose.h>
#include <aloam_velodyne/PointCloud2List.h>
#include <aloam_velodyne/Float64MultiArrayArray.h>
#include <aloam_velodyne/KPAndSurroundPC.h>
#include <aloam_velodyne/KPAndDescriptor.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

using namespace gtsam;

// Publisher 선언
ros::Publisher pubLCdetectResult, pubKeyPointResult, pubKeyPointDisplay, pubKeyPointSurround, pubKeyregionDisplay;

// 기본 파라미터 선언
float LocalMapBoundary = 30.0;
float HarrisThreshold = 0.000001;
float rops_RadiusSearch = 1.0;
int rops_NumberOfPartitionBins = 5;
int rops_NumberOfRotations = 3;
float rops_SupportRadius = 1.0;

// 변수 선언
int keyPointExtractionCount = 0;
int keyPointDetectionCount = 0;
int KeyPoseKeyPointsGFVectorSize = 0;
int keyPoseIdxPointCloudGFVectorSize = 0;
int keyPointIdx = 0;
int ProcessedDescriptorCount = 0;
int KeypointMergingProcessRange[2] = {0, 0};

// 뮤텍스 선언
std::mutex mtxkeyPoseIdxPointCloudGFVector;
std::mutex mtxPoseArray;

pcl::VoxelGrid<PointType> downSizeFilter;

// 기타 array, pointcloud, vector, queue 선언
geometry_msgs::PoseArray allPoseArray;
pcl::PointCloud<PointType>::Ptr clusteredGlobalKP(new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> KeyPoseKeyPointsGFVector;
std::vector<pcl::PointCloud<PointType>::Ptr> localMapVector;
std::vector<std::tuple<int, pcl::PointCloud<PointType>::Ptr, geometry_msgs::Pose>> keyPoseIdxPointCloudGFVector;
// std::vector<std::tuple<int, PointType, pcl::PointCloud<pcl::Histogram<135>>::Ptr>> globalKPandDCVector;
std::vector<std::tuple<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr, int, pcl::Histogram<135>>> KPwithInfo;

template <typename PointT>
pcl::PointCloud<PointType> saveClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in, PointType centerPoint, float thres)
{
    pcl::PointCloud<PointType> cloud_out;
    cloud_out.header = cloud_in.header;
    cloud_out.points.resize(cloud_in.points.size());

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (std::pow(cloud_in.points[i].x - centerPoint.x, 2) + std::pow(cloud_in.points[i].y - centerPoint.y, 2) + std::pow(cloud_in.points[i].z - centerPoint.z, 2) < thres * thres){
            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }
    }
    if (j != cloud_in.points.size()) { cloud_out.points.resize(j); }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
    return cloud_out;
}


/**
 * @brief /LGMLocalMap의 토픽명, aloam_velodyne::LocalMapAndPose의 형식의 메시지에 대한 callback함수. keyPoseIdxPointCloudGFQue에 idx, pointcloud, pose 순으로 pushback 된다.
 * @param _LocalMapAndPose aloam_velodyne::LocalMapAndPose::ConstPtr
*/
void LocalMapHandler(const aloam_velodyne::LocalMapAndPose::ConstPtr &_LocalMapAndPose){
    pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(_LocalMapAndPose->point_cloud, *thisKeyFrameDS);

    // 들어온 LocalMapAndPose데이터를 keyPoseIdxPointCloudGFQue에 모두 순서대로 저장.
    mtxkeyPoseIdxPointCloudGFVector.lock();
    keyPoseIdxPointCloudGFVector.push_back(std::make_tuple(_LocalMapAndPose->idx, thisKeyFrameDS, _LocalMapAndPose->pose));
    keyPoseIdxPointCloudGFVectorSize = keyPoseIdxPointCloudGFVector.size() - 1;
    mtxkeyPoseIdxPointCloudGFVector.unlock();
}

void KPAndDescriptorHandler(const aloam_velodyne::KPAndDescriptor::ConstPtr &_KPAndDescriptor){

    pcl::Histogram<135> Descriptor;
    for (size_t i = 0; i < 135; i++) {
        Descriptor.histogram[i] = _KPAndDescriptor->descriptor.data[i];
    }

    get<3>(KPwithInfo[ProcessedDescriptorCount]) = Descriptor;
    ProcessedDescriptorCount++;

    cout << "{KPAndDescriptorHandler} " << ProcessedDescriptorCount << endl;
}


/**
 * @brief cloudIn에 대해서 euclideanClustering을 수행하여, 클러스터링 된 point들의 평균점을 반환한다.
 * @param cloudIn pcl::PointCloud<PointType>::Ptr
 * @param l2dist pcl::EuclideanClusterExtraction<PointType>.setClusterTolerance
 * @param clusterSize pcl::EuclideanClusterExtraction<PointType>.setMinClusterSize
 * @return pcl::PointCloud<PointType>::Ptr
*/
pcl::PointCloud<PointType>::Ptr euclideanClusteringOnlyClusters( const pcl::PointCloud<PointType>::Ptr &cloudIn, float l2dist, int clusterSize ) {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    tree->setInputCloud(cloudIn);

    std::vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance (l2dist);
    ec.setMinClusterSize (clusterSize);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloudIn);
    ec.extract (cluster_indices);

    for (const auto& cluster : cluster_indices) {
        // 클러스터 내의 각 인덱스에 대해 작업
        float n = 0.0;
        float sumX = 0.0;
        float sumY = 0.0;
        float sumZ = 0.0;
        int intensity = 0;

        for (const auto& idx : cluster.indices) {
            n++;
            PointType pointI = cloudIn->points[idx];
            
            // 인텐시티 값을 0-255 범위로 매핑
            intensity += static_cast<int>(pointI.intensity * 255);
            sumX += pointI.x;
            sumY += pointI.y;
            sumZ += pointI.z;
        }
        PointType pointavg;
        pointavg.x = sumX / n;
        pointavg.y = sumY / n;
        pointavg.z = sumZ / n;
        pointavg.intensity = intensity / n;

        cloudOut->push_back(pointavg);
    }
    return cloudOut;
}


//process methods  
void KeypointDescriptorProcess( void ) {
}

void keypointMergingProcessrangeCal( void ) {
    KeypointMergingProcessRange[1] = KeyPoseKeyPointsGFVectorSize;
    KeypointMergingProcessRange[0] = 0;
    geometry_msgs::Pose currentPose = get<2>(keyPoseIdxPointCloudGFVector[KeypointMergingProcessRange[1]]);
    for (int i = KeypointMergingProcessRange[1] - 1; i >= 0; i--) {
        geometry_msgs::Pose poseCache = get<2>(keyPoseIdxPointCloudGFVector[i]);
        double distance = std::sqrt(std::pow(currentPose.position.x - poseCache.position.x, 2) +
                            std::pow(currentPose.position.y - poseCache.position.y, 2) +
                            std::pow(currentPose.position.z - poseCache.position.z, 2));
        if (distance > LocalMapBoundary * 1.5) {
            KeypointMergingProcessRange[0] = i;
            break;
        }
    }
}

void KeypointMergingProcess( void ) {
    float frequency = 10.0; // can change 
    ros::Rate rate(frequency);
    int processtimeN = 0;
    double processtimetotal = 0;
    while (ros::ok()){
        if (keyPointExtractionCount == 0) {
            rate.sleep();
            continue;
        }
        
        geometry_msgs::Pose poseCache = get<2>(keyPoseIdxPointCloudGFVector[KeypointMergingProcessRange[1]]);
        geometry_msgs::Pose currentPose = get<2>(keyPoseIdxPointCloudGFVector[KeyPoseKeyPointsGFVectorSize]);
        double distance = std::sqrt(std::pow(currentPose.position.x - poseCache.position.x, 2) +
                    std::pow(currentPose.position.y - poseCache.position.y, 2) +
                    std::pow(currentPose.position.z - poseCache.position.z, 2));
        if ( distance < LocalMapBoundary * 0.7 ){
            rate.sleep();
            continue;
        }
        auto start_time0 = std::chrono::high_resolution_clock::now();
        // KeypointMergingProcessRange -> Merging할 keypose범위 설정
        keypointMergingProcessrangeCal();

        
        pcl::PointCloud<PointType>::Ptr globalKPCache(new pcl::PointCloud<PointType>); // globalKPCache: KeypointMergingProcessRange를 합친 PC
        pcl::PointCloud<PointType>::Ptr globalPCCache(new pcl::PointCloud<PointType>); // globalKPCache: KeypointMergingProcessRange를 합친 PC
        globalKPCache->clear();
        // pcl::PointCloud<PointType>::Ptr globalPCCache(new pcl::PointCloud<PointType>); // globalKPCache: KeypointMergingProcessRange를 합친 PC
        for (int node_idx = KeypointMergingProcessRange[0]; node_idx < KeypointMergingProcessRange[1]; node_idx++) {
            *globalKPCache += *KeyPoseKeyPointsGFVector[node_idx];
            *globalPCCache += *get<1>(keyPoseIdxPointCloudGFVector[node_idx]);
        }
        downSizeFilter.setInputCloud(globalPCCache);
        downSizeFilter.filter(*globalPCCache);
        auto clusteredKPCache = euclideanClusteringOnlyClusters(globalKPCache, 0.1, 4);
        
        
        for (int i = 0; i < clusteredKPCache->size(); i++) {
            
            auto cpoint = clusteredKPCache->points[i];
            int minIdx = 0;
            double minDistanceCache = 100000.0;
            
            for (int node_idx = KeypointMergingProcessRange[0]; node_idx < KeypointMergingProcessRange[1]; node_idx++) {
                    geometry_msgs::Pose PoseCache = get<2>(keyPoseIdxPointCloudGFVector[node_idx]);
                    double distance = std::sqrt(std::pow(PoseCache.position.x - cpoint.x, 2) +
                                                std::pow(PoseCache.position.y - cpoint.y, 2) +
                                                std::pow(PoseCache.position.z - cpoint.z, 2));
                    if (distance < minDistanceCache) {
                        minDistanceCache = distance;
                        minIdx = node_idx;
                    }
            }
            
            pcl::PointCloud<PointType>::Ptr surroundPC(new pcl::PointCloud<PointType>());
            *surroundPC = saveClosedPointCloud(*globalPCCache, cpoint, 4.0);
            if (surroundPC->points.size() < 10) {
                continue;
            }

            pcl::PointCloud<PointType>::Ptr KPCache(new pcl::PointCloud<PointType>);
            KPCache->points.push_back(cpoint);
            pcl::PointXYZI cpose;
            cpose.x = get<2>(keyPoseIdxPointCloudGFVector[minIdx]).position.x;
            cpose.y = get<2>(keyPoseIdxPointCloudGFVector[minIdx]).position.y;
            cpose.z = get<2>(keyPoseIdxPointCloudGFVector[minIdx]).position.z;
            KPCache->points.push_back(cpose);           
            
            pcl::Histogram<135> dummyDescriptor;
            KPwithInfo.push_back(std::make_tuple(KPCache, surroundPC, minIdx, dummyDescriptor));
        }
        

        auto end_time0 = std::chrono::high_resolution_clock::now();
        auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(end_time0 - start_time0);
        processtimetotal += duration0.count()/1000.0;
        processtimeN++;
        cout << "    [MergingProcess] coverage: " << KeypointMergingProcessRange[0] << "~" << KeypointMergingProcessRange[1] << " || clusterPointNum: " << clusteredKPCache->size() << " || All Keypoints: " << KPwithInfo.size() << " || process time: " << processtimetotal/processtimeN << "(ms)" << endl;
        
        rate.sleep();
    }
}

void KeypointDetectionProcess( void ) {
    float frequency = 10.0; // can change 
    ros::Rate rate(frequency);
    int processtimeN = 0;
    double processtimetotal = 0;
    while (ros::ok()){
        mtxkeyPoseIdxPointCloudGFVector.lock();
        // keyPoseIdxPointCloudGFQue에 데이터가 있다면 실행
        if (keyPoseIdxPointCloudGFVectorSize > keyPointExtractionCount) {
            auto start_time = std::chrono::high_resolution_clock::now();
            // cout << "[keyPointDetection] for " << globalKeypointsVectorSize + 1 << " odom node" << endl;
            auto localCache = keyPoseIdxPointCloudGFVector[keyPointExtractionCount];
            mtxkeyPoseIdxPointCloudGFVector.unlock();

            int localMapIdx = get<0>(localCache);
            pcl::PointCloud<PointType>::Ptr localMapPcl = get<1>(localCache);
            geometry_msgs::Pose localMapPose = get<2>(localCache);
            // localMapVector.push_back(localMapPcl);

            // NormalEstimation
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
            pcl::search::KdTree<pcl::PointXYZI>::Ptr treeNe(new pcl::search::KdTree<pcl::PointXYZI> ());
            ne.setSearchMethod(treeNe);
            ne.setKSearch(0);
            ne.setViewPoint(localMapPose.position.x, localMapPose.position.y, localMapPose.position.z);
            ne.setRadiusSearch(1.5f); // 1
            ne.setInputCloud(localMapPcl);
            ne.compute(*normals);

            // HarrisKeypoint3D 추출
            pcl::PointCloud<pcl::PointXYZI>::Ptr currkeypoints(new pcl::PointCloud<PointType>);
            pcl::HarrisKeypoint3D <pcl::PointXYZI, pcl::PointXYZI> detector;
            detector.setNonMaxSupression (true);
            detector.setInputCloud (localMapPcl);
            detector.setRadiusSearch(100);
            detector.setRadius (2.0f);
            detector.setThreshold (HarrisThreshold);
            detector.setNormals(normals);
            detector.compute (*currkeypoints);

            KeyPoseKeyPointsGFVector.push_back(currkeypoints); // 포인트 클라우드 벡터
            KeyPoseKeyPointsGFVectorSize = KeyPoseKeyPointsGFVector.size() - 1;
            keyPointExtractionCount++;
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            // cout << "point size : " << localMapPcl->size() << endl;

            processtimetotal += duration.count()/1000.0;
            processtimeN++;
            cout << "[DetectionProcess] processed/Que " << keyPointExtractionCount << "/" << keyPoseIdxPointCloudGFVectorSize << " || keypoint Num: " << currkeypoints->points.size() << " || process time: " << processtimetotal/processtimeN << "(ms)" << endl;
        }
        else{
            mtxkeyPoseIdxPointCloudGFVector.unlock();
        }
        rate.sleep();
    }
}

void PubKeypointProcess( void ) { 
    float frequency = 1.0;
    ros::Rate rate(frequency);
    long unsigned int keypointCount = 0;
    while ( ros::ok() ) {
        while ( keypointCount < KPwithInfo.size() ) {
            if (KPwithInfo.size() == 0){
                rate.sleep();
                break;
            }

            aloam_velodyne::KPAndSurroundPC KPSurrmsg;
            sensor_msgs::PointCloud2 KPmsg;
            sensor_msgs::PointCloud2 Smsg;

            pcl::toROSMsg(*get<0>(KPwithInfo[keypointCount]), KPmsg);
            pcl::toROSMsg(*get<1>(KPwithInfo[keypointCount]), Smsg);
            KPSurrmsg.keypoint_point_cloud = KPmsg;
            KPSurrmsg.surround_point_cloud = Smsg;

            pubKeyPointSurround.publish(KPSurrmsg);
            keypointCount++;
        }
        if ((keypointCount == KPwithInfo.size()) && ((KPwithInfo.size() != 0))) {
            cout << "        [PubKeypointProcess] pub AllPoint. " << keypointCount << " / " << KPwithInfo.size() << endl;
        }
        
        rate.sleep();
    }
}

void KeypointDisplayProcess( void ) {
    float frequency = 0.5; // can change 
    ros::Rate rate(frequency);
    while (ros::ok()){
        pcl::PointCloud<PointType>::Ptr globalKPMapforDisplay(new pcl::PointCloud<PointType>());
        for (size_t i = 0; i < KPwithInfo.size(); i++) {
            globalKPMapforDisplay->points.push_back(get<0>(KPwithInfo[i])->points[0]);
        }
        sensor_msgs::PointCloud2 KeypointMsg;
        pcl::toROSMsg(*globalKPMapforDisplay, KeypointMsg);
        KeypointMsg.header.frame_id = "/camera_init";
        pubKeyPointDisplay.publish(KeypointMsg);
        rate.sleep();
    }
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "alaserMatcher");
	ros::NodeHandle nh;

    nh.param<float>("LocalMapBoundary", LocalMapBoundary, 30.0);
    nh.param<float>("HarrisThreshold", HarrisThreshold, 1e-10);
    nh.param<float>("rops_RadiusSearch", rops_RadiusSearch, 1.0);
    nh.param<int>("rops_NumberOfPartitionBins", rops_NumberOfPartitionBins, 5);
    nh.param<int>("rops_NumberOfRotations", rops_NumberOfRotations, 3); 
    nh.param<float>("rops_SupportRadius", rops_SupportRadius, 1.0);

    float FilterSize = 0.1;
    downSizeFilter.setLeafSize(FilterSize, FilterSize, FilterSize);

    ros::Subscriber subKeyLocalMap = nh.subscribe<aloam_velodyne::LocalMapAndPose>("/LGMLocalMap", 100, LocalMapHandler);
    ros::Subscriber subDescriptor = nh.subscribe<aloam_velodyne::KPAndDescriptor>("/KPAndDescriptor", 100, KPAndDescriptorHandler);
    // ros::Subscriber subAllPose = nh.subscribe<geometry_msgs::PoseArray>("/LGMAllPose", 100, PoseHandler);

    pubKeyPointSurround = nh.advertise<aloam_velodyne::KPAndSurroundPC>("/KPSurroundPC", 100);

	// pubLCdetectResult = nh.advertise<aloam_velodyne::LCPair>("/LCdetectResult", 100);
    // pubKeyPointResult = nh.advertise<aloam_velodyne::PointCloud2List>("/keyPointResult", 100);
    pubKeyPointDisplay = nh.advertise<sensor_msgs::PointCloud2>("/keyPointDisplay", 100);
    // pubKeyregionDisplay = nh.advertise<sensor_msgs::PointCloud2>("/keyregionDisplay", 100);

    std::thread threadKPDetect(KeypointDetectionProcess);
    std::thread threadKPMerge(KeypointMergingProcess);
    std::thread threadKPDisplay(KeypointDisplayProcess);
    std::thread threadKPDescriptor(PubKeypointProcess);
    // std::thread threadKPDescriptor(KeypointDescriptorProcess);

 	ros::spin();

	return 0;
}
