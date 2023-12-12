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
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <aloam_velodyne/LCPair.h>
#include <aloam_velodyne/LocalMapAndPose.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "scancontext/Scancontext.h"

using namespace gtsam;

typedef pcl::SHOT352 ShotFeature;

std::mutex mKeyFrameBuf;
std::mutex mLocalMapBuf;
std::mutex globalKeypointBuf;
std::vector<std::pair<int, int>> ICPPassedPair;
std::queue<pcl::PointCloud<PointType>::Ptr> keyFrameQue;
std::queue<pcl::PointCloud<PointType>::Ptr> localMapPclQue;
std::queue<geometry_msgs::Pose> localMapPoseQue;
std::vector<std::tuple<pcl::PointCloud<PointType>::Ptr, geometry_msgs::Pose, pcl::PointCloud<pcl::Histogram<135>>::Ptr>> localkeypointsVector;
std::vector<std::tuple<pcl::PointCloud<PointType>::Ptr, geometry_msgs::Pose, pcl::PointCloud<pcl::Histogram<135>>::Ptr>> globalKeypointsVector;
std::vector<std::tuple<pcl::PointCloud<PointType>::Ptr, geometry_msgs::Pose, pcl::PointCloud<pcl::Histogram<135>>::Ptr>> recentKeypointsVector; 
// vector<tuple<int, string, int>> v2;
// v2.push_back(make_tuple(2, "cbg", 7));
// std::vector<std::pair<PointType, geometry_msgs::Pose>> keypointsFeatureVector;
pcl::VoxelGrid<PointType> downSizeFilterScancontext;
pcl::PointCloud<PointType>::Ptr KeypointsMergeMapCache(new pcl::PointCloud<PointType>());
pcl::PointCloud<pcl::Histogram<135>>::Ptr DescriptorsMergeMapCache(new pcl::PointCloud<pcl::Histogram<135>>());
pcl::PointCloud<PointType>::Ptr globalKeypointsMergeMap(new pcl::PointCloud<PointType>());

SCManager scManager;
double scDistThres, scMaximumRadius;

ros::Publisher pubLCdetectResult, pubKeyPointResult, pubKeyPointDisplay;

int KeyFrameNum = 0;
int LocalMapNum = 0;
int LocalMapIdxRange = 6;
int globalKeypointsVectorSize = 0;

float LocalMapBoundary = 30.0;
float ISS_SalientRadius = 10;
float ISS_NonMaxRadius = 6;
float ISS_Gamma21 = 0.9;
float ISS_Gamma23 = 0.9;
int ISS_MinNeighbors = 10;
std::string Detector = "ISS";
float HarrisThreshold = 0.000001;

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

pcl::PointCloud<pcl::PointXYZI>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const geometry_msgs::Pose& rosPose)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudOut(new pcl::PointCloud<PointType>());

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

// void keyPointDetectionISS(void) {
//     mLocalMapBuf.lock();
//     auto localMapPcl = localMapPclQue.front();
//     auto localMapPose = localMapPoseQue.front();
//     localMapPclQue.pop();
//     localMapPoseQue.pop();
//     mLocalMapBuf.unlock();

//     // 키포인트 추출
//     pcl::PointCloud<PointType>::Ptr currkeypoints(new pcl::PointCloud<PointType>);

//     // [] ISS keypoint 추출
//     pcl::ISSKeypoint3D<PointType, PointType> detector;
//     detector.setInputCloud(localMapPcl);
//     pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
//     detector.setSearchMethod(kdtree);
//     // double resolution = computeCloudResolution(cloud);
//     // Set the radius of the spherical neighborhood used to compute the scatter matrix.
//     detector.setSalientRadius(ISS_SalientRadius * 0.2);
//     // Set the radius for the application of the non maxima supression algorithm.
//     detector.setNonMaxRadius(ISS_NonMaxRadius * 0.2);
//     // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
//     detector.setMinNeighbors(ISS_MinNeighbors);
//     // Set the upper bound on the ratio between the second and the first eigenvalue.
//     detector.setThreshold21(ISS_Gamma21);
//     // Set the upper bound on the ratio between the third and the second eigenvalue.
//     detector.setThreshold32(ISS_Gamma23);
//     // Set the number of prpcessing threads to use. 0 sets it to automatic.
//     detector.setNumberOfThreads(4);
//     detector.compute(*currkeypoints);

//     // // 바운더리 근처의 키포인트는 제거
//     // pcl::PassThrough<PointType> pass;
//     // pass.setInputCloud (currkeypoints);
//     // pass.setFilterFieldName ("x");
//     // pass.setFilterLimits (pclCenterPoint.x - (2*Local_map_boundary/3), pclCenterPoint.x + (2*Local_map_boundary/3));
//     // pass.filter (*currkeypoints);

//     // pass.setFilterFieldName ("y");
//     // pass.setFilterLimits (pclCenterPoint.y - (2*Local_map_boundary/3), pclCenterPoint.y + (2*Local_map_boundary/3));
//     // pass.filter (*currkeypoints);

//     // pass.setFilterFieldName ("z");
//     // pass.setFilterLimits (pclCenterPoint.z - 10.0, pclCenterPoint.z + 10.0);
//     // pass.filter (*currkeypoints);
//     cout << "ISS : CurrentKeyFrameNum:" << KeyFrameNum << " || KeypointsVector's size: " << localkeypointsVector.size() << " || Keypoint's num : " << currkeypoints->points.size() << endl;
    
//     pcl::PointCloud<ShotFeature>::Ptr ShotFeatures;
//     // globalKeypointsVector.push_back(std::make_tuple(currkeypoints, localMapPose, ShotFeatures));
//     *currkeypoints = *global2local(currkeypoints, localMapPose);
//     // localkeypointsVector.push_back(std::make_pair(currkeypoints, localMapPose));

//     sensor_msgs::PointCloud2 keyPointMsg;
//     pcl::toROSMsg(*currkeypoints, keyPointMsg);
//     keyPointMsg.header.frame_id = "/camera_init";
//     pubKeyPointResult.publish(keyPointMsg);
// }

pcl::PointCloud<pcl::PointXYZI>::Ptr convertPointCloud(const pcl::PointCloud<PointType>& input_cloud) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    for (const auto& point : input_cloud.points) {
        pcl::PointXYZI new_point;
        new_point.x = point.x;
        new_point.y = point.y;
        new_point.z = point.z;
        // CustomPoint에 추가된 멤버 변수가 있을 경우, 해당 변수도 복사해줌
        new_point.intensity = point.intensity;

        output_cloud->push_back(new_point);
    }

    return output_cloud;
}

// void EdgeDetection(void) {
//     mLocalMapBuf.lock();
//     auto localMapPcl = localMapPclQue.front();
//     auto localMapPose = localMapPoseQue.front();
//     localMapPclQue.pop();
//     localMapPoseQue.pop();
//     mLocalMapBuf.unlock();

//     pcl::PointCloud<pcl::PointXYZI>::Ptr converted_cloud = convertPointCloud(*localMapPcl);
//     pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>);
//     pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
//     pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI> ());
//     ne.setSearchMethod (tree);
//     ne.setRadiusSearch (1.0);
//     ne.setInputCloud(converted_cloud);
//     ne.compute(*normal);

//     pcl::PointCloud<PointType>::Ptr currEdges(new pcl::PointCloud<PointType>);

//     // Organized Edge Detection 객체 생성
//     pcl::OrganizedEdgeFromNormals<pcl::PointXYZI, pcl::Normal, pcl::Label> oed;
//     oed.setInputCloud(converted_cloud);  // 입력 PointCloud 설정
//     oed.setInputNormals(normal);
//     oed.setDepthDisconThreshold(0.0002f);
//     oed.setHCCannyHighThreshold(0.5f);
//     oed.setHCCannyLowThreshold(0.2f);
//     oed.setMaxSearchNeighbors(50);

//     // oed.setDepthDisconThreshold(0.000001);
//     // oed.setMaxSearchNeighbors(50);
//     oed.setEdgeType(oed.EDGELABEL_NAN_BOUNDARY | oed.EDGELABEL_OCCLUDING |
//                     oed.EDGELABEL_OCCLUDED | oed.EDGELABEL_HIGH_CURVATURE);

//     // Edge Detection 수행
//     pcl::PointCloud<pcl::Label> labels;
//     std::vector<pcl::PointIndices> label_indices;
//     oed.compute(labels, label_indices);

//     cout << "--" << labels[0] << "--" << label_indices[0] << label_indices[1]<< label_indices[2]<<label_indices[3]<<label_indices[4] << endl;
//     cout << "--------------------------------------------------------------------------------" << endl;

//     pcl::PointCloud<pcl::PointXYZI>::Ptr occluding_edges(new pcl::PointCloud<pcl::PointXYZI>),
//                                         occluded_edges(new pcl::PointCloud<pcl::PointXYZI>),
//                                         nan_boundary_edges(new pcl::PointCloud<pcl::PointXYZI>),
//                                         high_curvature_edges(new pcl::PointCloud<pcl::PointXYZI>),
//                                         rgb_edges(new pcl::PointCloud<pcl::PointXYZI>),
//                                         final_edges(new pcl::PointCloud<pcl::PointXYZI>);

//     pcl::copyPointCloud(*converted_cloud, label_indices[0].indices, *nan_boundary_edges);
//     pcl::copyPointCloud(*converted_cloud, label_indices[1].indices, *occluding_edges);
//     pcl::copyPointCloud(*converted_cloud, label_indices[2].indices, *occluded_edges);
//     pcl::copyPointCloud(*converted_cloud, label_indices[3].indices, *high_curvature_edges);
//     // pcl::copyPointCloud(*converted_cloud, label_indices[4].indices, *rgb_edges);
//     *final_edges += *occluding_edges;
//     *final_edges += *occluded_edges;
//     *final_edges += *nan_boundary_edges;
//     *final_edges += *high_curvature_edges;
//     *final_edges += *rgb_edges;
// }

void keyPointDetectionHarris(void) {
    mLocalMapBuf.lock();
    auto localMapPcl = localMapPclQue.front();
    auto localMapPose = localMapPoseQue.front();
    localMapPclQue.pop();
    localMapPoseQue.pop();
    mLocalMapBuf.unlock();

    // 이상치 제거 (Statistical Outlier Removal)
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(localMapPcl);
    sor.setMeanK(50); // 주변 이웃의 수
    sor.setStddevMulThresh(1.0); // 표준 편차의 배수
    sor.filter(*localMapPcl);

    // NormalEstimation
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr treeNe(new pcl::search::KdTree<pcl::PointXYZI> ());
    ne.setSearchMethod(treeNe);
    ne.setKSearch(0);
    ne.setRadiusSearch(1.0f);
    ne.setInputCloud(localMapPcl);
    ne.compute(*normals);

    // SHOTEstimation
    // pcl::PointCloud<ShotFeature>::Ptr shotFeatures(new pcl::PointCloud<ShotFeature>);
    // pcl::SHOTEstimation<PointType, pcl::Normal, ShotFeature> shotEstimation;
    // shotEstimation.setInputCloud(localMapPcl);
    // pcl::search::KdTree<PointType>::Ptr treeDescriptor(new pcl::search::KdTree<PointType> ());
    // shotEstimation.setSearchMethod(treeDescriptor);
    // shotEstimation.setInputNormals(normal);
    // shotEstimation.setKSearch(0);
    // shotEstimation.setRadiusSearch(1.0f);
    // shotEstimation.compute(*shotFeatures);

    // HarrisKeypoint3D 추출
    pcl::PointCloud<pcl::PointXYZI>::Ptr currkeypoints(new pcl::PointCloud<PointType>);
    pcl::HarrisKeypoint3D <pcl::PointXYZI, pcl::PointXYZI> detector;
    detector.setNonMaxSupression (true);
    detector.setInputCloud (localMapPcl);
    detector.setRadiusSearch(50);
    detector.setRadius (1.0f);
    detector.setThreshold (HarrisThreshold);
    detector.setNormals(normals);
    detector.compute (*currkeypoints);

    // 바운더리 근처의 키포인트는 제거
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud (currkeypoints);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (localMapPose.position.x - (2*Local_map_boundary/3), localMapPose.position.x + (2*Local_map_boundary/3));
    pass.filter (*currkeypoints);

    pass.setFilterFieldName ("y");
    pass.setFilterLimits (localMapPose.position.y - (2*Local_map_boundary/3), localMapPose.position.y + (2*Local_map_boundary/3));
    pass.filter (*currkeypoints);

    pass.setFilterFieldName ("z");
    pass.setFilterLimits (localMapPose.position.z - 10.0, localMapPose.position.z + 10.0);
    pass.filter (*currkeypoints);

    // RoPS 디스크립터
    // Perform triangulation.
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudNormals(new pcl::PointCloud<pcl::PointXYZINormal>);
	pcl::concatenateFields(*localMapPcl, *normals, *cloudNormals);
	pcl::search::KdTree<pcl::PointXYZINormal>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointXYZINormal>);
	kdtree2->setInputCloud(cloudNormals);
	pcl::GreedyProjectionTriangulation<pcl::PointXYZINormal> triangulation;
	pcl::PolygonMesh triangles;
	triangulation.setSearchRadius(1.0f);
	triangulation.setMu(2.5);
	triangulation.setMaximumNearestNeighbors(50);
	triangulation.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees.
	triangulation.setNormalConsistency(false);
	triangulation.setMinimumAngle(M_PI / 18); // 10 degrees.
	triangulation.setMaximumAngle(2 * M_PI / 3); // 120 degrees.
	triangulation.setInputCloud(cloudNormals);
	triangulation.setSearchMethod(kdtree2);
	triangulation.reconstruct(triangles);

    pcl::ROPSEstimation<pcl::PointXYZI, pcl::Histogram<135>> rops;
	rops.setInputCloud(currkeypoints); 
	rops.setSearchMethod(treeNe);
    rops.setSearchSurface (localMapPcl);
	rops.setRadiusSearch(3.0f);
	rops.setTriangles(triangles.polygons);
	rops.setNumberOfPartitionBins(5);
	rops.setNumberOfRotations(3);
	rops.setSupportRadius(0.3f);
    pcl::PointCloud<pcl::Histogram<135>>::Ptr descriptors(new pcl::PointCloud<pcl::Histogram<135>>());
	rops.compute(*descriptors);

    cout << "|| Harris : CurrentKeyFrameNum:" << KeyFrameNum << " || KeypointsVector's size: " << globalKeypointsVector.size() << " || Keypoint's num : " << currkeypoints->points.size() << "|| descriptor : currkeypoint's size:" << currkeypoints->size() << " || Features num: " << descriptors->size() << " || Features's size: " << descriptors->points[0].descriptorSize() << endl;// << shotFeatures->points[0] << endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr KeyPointCashe(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*currkeypoints, *KeyPointCashe);
    globalKeypointBuf.lock();
    globalKeypointsVector.push_back(std::make_tuple(KeyPointCashe, localMapPose, descriptors));
    globalKeypointsVectorSize = globalKeypointsVector.size();
    globalKeypointBuf.unlock();

    // localkeypointsVector에 빈 요소들 추가
    pcl::PointCloud<pcl::PointXYZI>::Ptr localkeypointCashe(new pcl::PointCloud<pcl::PointXYZI>);
    geometry_msgs::Pose* localkeypointPoseCashe = new geometry_msgs::Pose();
    pcl::PointCloud<pcl::Histogram<135>>::Ptr localkeypointDescriptorsCashe(new pcl::PointCloud<pcl::Histogram<135>>());
    localkeypointsVector.push_back(std::make_tuple(localkeypointCashe, *localkeypointPoseCashe, localkeypointDescriptorsCashe)); // -> keypointsVector에 tuple 헤더 : #include <tuple> 로 디스크립터도 묶어야할듯

    // sensor_msgs::PointCloud2 keyPointMsg;
    // pcl::toROSMsg(*currkeypoints, keyPointMsg);
    // keyPointMsg.header.frame_id = "/camera_init";
    // pubKeyPointResult.publish(keyPointMsg);
}

std::tuple<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<pcl::Histogram<135>>::Ptr> euclideanClustering( const pcl::PointCloud<PointType>::Ptr &cloudIn, const pcl::PointCloud<pcl::Histogram<135>>::Ptr &descriptorIn, float l2dist, int clusterSize ) {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    pcl::PointCloud<pcl::Histogram<135>>::Ptr descriptorOut(new pcl::PointCloud<pcl::Histogram<135>>());

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
        pcl::Histogram<135> descriptorI = descriptorIn->points[cluster.indices[0]];
        cloudOut->push_back(pointavg);
        descriptorOut->push_back(descriptorI);
    }
    return {cloudOut, descriptorOut};
}

void keypointClusterAndMerge( int startIdx ) {
    // alaserScDetector: /usr/local/include/boost/smart_ptr/shared_ptr.hpp:734: typename boost::detail::sp_member_access<T>::type boost::shared_ptr<T>::operator->() const [with T = pcl::PointCloud<pcl::PointXYZI>; typename boost::detail::sp_member_access<T>::type = pcl::PointCloud<pcl::PointXYZI>*]: Assertion `px != 0' failed.
    // [ WARN] [1702302643.268717901]: TF_REPEATED_DATA ignoring data with redundant timestamp for frame aft_pgo (parent camera_init) at time 7.671396 according to authority unknown_publisher
    // [alaserScDetector-6] process has died [pid 3290571, exit code -6, cmd /home/vision/catkin_ws/devel/lib/aloam_velodyne/alaserScDetector /velodyne_points:=/velodyne_points __name:=alaserScDetector __log:=/home/vision/.ros/log/406bd966-982c-11ee-aa85-b98370643f1a/alaserScDetector-6.log].
    // log file: /home/vision/.ros/log/406bd966-982c-11ee-aa85-b98370643f1a/alaserScDetector-6*.log
    
    KeypointsMergeMapCache->clear();
    DescriptorsMergeMapCache->clear();
    for(int i = (startIdx); i < (startIdx + 10); i++) {
        *KeypointsMergeMapCache += *get<0>(globalKeypointsVector[i]);
        *DescriptorsMergeMapCache += *get<2>(globalKeypointsVector[i]);
    }
    // pcl::PointCloud<PointType>::Ptr localKeypointsMap(new pcl::PointCloud<PointType>::Ptr);
    // pcl::PointCloud<pcl::Histogram<135>>::Ptr localDescriptorsMap(new pcl::PointCloud<pcl::Histogram<135>>::Ptr);
    auto [localKeypointsMap, localDescriptorsMap] = euclideanClustering(KeypointsMergeMapCache, DescriptorsMergeMapCache, 0.2, 4);

    std::vector<std::tuple<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<pcl::Histogram<135>>::Ptr>> KeypointsLocalMergeMapCacheVector;
    for (int i = 0; i <= 10; i++) {
        pcl::PointCloud<PointType>::Ptr pointCloudPtr(new pcl::PointCloud<PointType>);
        pcl::PointCloud<pcl::Histogram<135>>::Ptr histogramPtr(new pcl::PointCloud<pcl::Histogram<135>>);
        KeypointsLocalMergeMapCacheVector.push_back(std::make_tuple(pointCloudPtr, histogramPtr));
    }

    for (int i = 0; i < localKeypointsMap->size(); i++) {
        cout << "i : " << i << endl;
        float minDist = 1000000.0;
        int minIdx = 0;
        PointType currentPoint = localKeypointsMap->points[i];
        pcl::Histogram<135> currentDescriptor = localDescriptorsMap->points[i];
        
        for(int j = (startIdx); j < (startIdx + 10); j++) {
            
            geometry_msgs::Pose currentPose = get<1>(globalKeypointsVector[j]); // get<1>(globalKeypointsVector[j]) -> j번째 keynode의 pose

            double distance = std::sqrt(std::pow(currentPoint.x - currentPose.position.x, 2) +
                                        std::pow(currentPoint.y - currentPose.position.y, 2) +
                                        std::pow(currentPoint.z - currentPose.position.z, 2));
            if (distance < minDist) {
                minIdx = j;
                minDist = distance;
            }
        }
        cout << "min keypoes: " << minIdx << endl;
        auto& keypointsPtr = std::get<0>(KeypointsLocalMergeMapCacheVector[minIdx - startIdx]);
        auto& descriptorsPtr = std::get<1>(KeypointsLocalMergeMapCacheVector[minIdx - startIdx]);
        keypointsPtr->push_back(currentPoint);
        descriptorsPtr->push_back(currentDescriptor);
    }

    for ( int i = (startIdx); i < (startIdx + 10); i++ ) {
        *get<0>(localkeypointsVector[i]) += *global2local(get<0>(KeypointsLocalMergeMapCacheVector[i - startIdx]), get<1>(globalKeypointsVector[i]));
        get<1>(localkeypointsVector[i]) = get<1>(globalKeypointsVector[i]);
        *get<2>(localkeypointsVector[i]) += *get<1>(KeypointsLocalMergeMapCacheVector[i - startIdx]);
    }
}

void PubKeypoint( void ) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr PubKeypointCache(new pcl::PointCloud<pcl::PointXYZI>());
    for(int i = 0; i < (localkeypointsVector.size() - 5); i++) {
        *PubKeypointCache += *local2global(get<0>(localkeypointsVector[i]), get<1>(localkeypointsVector[i]));
    }
    sensor_msgs::PointCloud2 globalKeypointsMergeMapmsg;
    pcl::toROSMsg(*PubKeypointCache, globalKeypointsMergeMapmsg);
    globalKeypointsMergeMapmsg.header.frame_id = "/camera_init";
    pubKeyPointDisplay.publish(globalKeypointsMergeMapmsg);
}

int keypointClusterAndMergeCount = 10;

void KeypointDetectionProcess(void) {
    float loopClosureFrequency = 30.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok()){
        mLocalMapBuf.lock();
        if (localMapPoseQue.size() > 0) {
            mLocalMapBuf.unlock();
            cout << "[keyPointDetection] for " << globalKeypointsVectorSize + 1 << " odom node" << endl;
            if (Detector == "ISS") {
                // keyPointDetectionISS();
            }
            else if (Detector == "Harris") {
                keyPointDetectionHarris();
            }
        }
        else{
            mLocalMapBuf.unlock();
        }
        if (globalKeypointsVectorSize >= keypointClusterAndMergeCount + 6) {
            cout << "[keypointClusterAndMerge] " << globalKeypointsVectorSize << "'s process." << endl;
            keypointClusterAndMerge(keypointClusterAndMergeCount - 10);
            PubKeypoint();
            globalKeypointBuf.lock();
            keypointClusterAndMergeCount += 6;
            globalKeypointBuf.unlock();
            // EdgeDetection();
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
    nh.param<std::string>("Detector", Detector, "SIFT"); 
    nh.param<float>("HarrisThreshold", HarrisThreshold, 1e-10);
    
    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.1;
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);

	ros::Subscriber subKeyFrameDS = nh.subscribe<sensor_msgs::PointCloud2>("/KeyFrameDSforLC", 100, KeyFrameDSHandler);
    ros::Subscriber subKeyLocalMap = nh.subscribe<aloam_velodyne::LocalMapAndPose>("/LGMLocalMap", 100, LocalMapHandler);

	pubLCdetectResult = nh.advertise<aloam_velodyne::LCPair>("/LCdetectResult", 100);
    pubKeyPointResult = nh.advertise<sensor_msgs::PointCloud2>("/keyPointResult", 100);
    pubKeyPointDisplay = nh.advertise<sensor_msgs::PointCloud2>("/keyPointDisplay", 100);

    std::thread threadSC(ScancontextProcess);
    std::thread threadKP(KeypointDetectionProcess);

 	ros::spin();

	return 0;
}
