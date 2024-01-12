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

float rops_RadiusSearch = 1.0;
int rops_NumberOfPartitionBins = 5;
int rops_NumberOfRotations = 3;
float rops_SupportRadius = 1.0;

float triangulation_SearchRadius = 0.0;
float triangulation_Mu = 0.0;
int triangulation_MaximumNearestNeighbors = 0;

ros::Publisher pubDescriptor, pubKeyregionDisplay, pubKeyregionDisplay3;

std::mutex mtxQue;

std::queue<pcl::PointCloud<PointType>::Ptr> SorroundQue;
std::queue<pcl::PointCloud<PointType>::Ptr> KPQue;
pcl::PointCloud<PointType>::Ptr globalregienforDisplay(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr globalregienforDisplayFail(new pcl::PointCloud<PointType>());

int pointSubCount = 0;
int pointDescriptorProcessedCount = 0;

void KPAndSurroundHandler( const aloam_velodyne::KPAndSurroundPC::ConstPtr &_KPAndSurround ) {
    pcl::PointCloud<PointType>::Ptr surroundCache(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr KPCache(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(_KPAndSurround->surround_point_cloud, *surroundCache);
    pcl::fromROSMsg(_KPAndSurround->keypoint_point_cloud, *KPCache);
    mtxQue.lock();
    SorroundQue.push(surroundCache);
    KPQue.push(KPCache);
    mtxQue.unlock();
    pointSubCount++;
}

void KeypointDescriptorDetectionProcess( void ) {
    ros::Rate rate(1.0);
    int processtimeN = 0;
    double processtimetotal = 0;
    while (ros::ok) {
        cout << "[SorroundQue.size] " << SorroundQue.size() << endl;
        while (SorroundQue.size() == 0) {
            auto start_time = std::chrono::high_resolution_clock::now();
            pcl::PointCloud<PointType>::Ptr SorroundPC(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr KPPC(new pcl::PointCloud<PointType>());

            // mtxQue.lock();
            // SorroundPC = SorroundQue.front();
            // KPPC = KPQue.front();
            // SorroundQue.pop();
            // KPQue.pop();
            // mtxQue.unlock();
            // *globalregienforDisplay += *SorroundPC;

            // if (SorroundPC->points.size() == 0) {
            //     globalregienforDisplayFail->points.push_back(KPPC->points[0]);
            //     sensor_msgs::PointCloud2 Smsg;
            //     pcl::toROSMsg(*globalregienforDisplayFail, Smsg);
            //     Smsg.header.frame_id = "/camera_init";
            //     pubKeyregionDisplay3.publish(Smsg);
            //     continue;
            // }

            //저장하는 코드하나
            //불러오는 코드하나
            //해서 저장만 한번 돌리고, 주석처리 후 불러오는거 반복해서 퀄리티 체크
            // pcl::io::savePCDFileASCII("/home/vision/catkin_ws/src/SC-A-LOAM/src/point_cloud_Sorround.pcd", *SorroundPC);
            // pcl::io::savePCDFileASCII("/home/vision/catkin_ws/src/SC-A-LOAM/src/point_cloud_KP.pcd", *KPPC);

            if (pcl::io::loadPCDFile<PointType>("/home/vision/catkin_ws/src/SC-A-LOAM/src/point_cloud_Sorround.pcd", *SorroundPC) == -1) {
                PCL_ERROR("Couldn't read file 'point_cloud_Sorround.pcd'\n");
            }
            if (pcl::io::loadPCDFile<PointType>("/home/vision/catkin_ws/src/SC-A-LOAM/src/point_cloud_KP.pcd", *KPPC) == -1) {
                PCL_ERROR("Couldn't read file 'point_cloud_KP.pcd'\n");
            }

            float triangulation_SearchRadiustest[] = { 1.0, 2.0, 3.0 };
            float triangulation_Mutest[] = { 5.0, 7.0, 10.0 };
            int triangulation_MaximumNearestNeighborstest[] = { 50, 70, 100 };

            for (const auto &triangulation_SearchRadius : triangulation_SearchRadiustest) {
                for (const auto &triangulation_Mu : triangulation_Mutest) {
                    for (const auto &triangulation_MaximumNearestNeighbors : triangulation_MaximumNearestNeighborstest) {
                        // cout << "포인트 개수: " << SorroundPC->points.size() << " || " << KPPC->points.size() << endl;
                        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
                        pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
                        pcl::search::KdTree<pcl::PointXYZI>::Ptr treeNe(new pcl::search::KdTree<pcl::PointXYZI> ());
                        ne.setSearchMethod(treeNe);
                        ne.setKSearch(0);
                        ne.setViewPoint(KPPC->points[1].x, KPPC->points[1].y, KPPC->points[1].z);// 뷰포인트가 이게 아니다. keypose를 가져와야된다.
                        ne.setRadiusSearch(1.0f); // 1
                        ne.setInputCloud(SorroundPC);
                        ne.compute(*normals);

                        // Perform triangulation.
                        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloudNormals(new pcl::PointCloud<pcl::PointXYZINormal>);
                        // pcl::concatenateFields(*SorroundPC, *normals, *cloudNormals);
                        // pcl::search::KdTree<pcl::PointXYZINormal>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointXYZINormal>);
                        // kdtree2->setInputCloud(cloudNormals);
                        // pcl::GreedyProjectionTriangulation<pcl::PointXYZINormal> triangulation;
                        // pcl::PolygonMesh triangles;
                        // triangulation.setSearchRadius(0.3f);
                        // triangulation.setMu(2.5);
                        // triangulation.setMaximumNearestNeighbors(15);
                        // triangulation.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees.
                        // triangulation.setNormalConsistency(false);
                        // triangulation.setMinimumAngle(M_PI / 18); // 10 degrees.
                        // triangulation.setMaximumAngle(2 * M_PI / 3); // 120 degrees.
                        // triangulation.setInputCloud(cloudNormals);
                        // triangulation.setSearchMethod(kdtree2);
                        // triangulation.reconstruct(triangles);
                        // pcl::io::savePolygonFileVTK("/home/vision/catkin_ws/dd.vtk", triangles);

                        pcl::concatenateFields(*SorroundPC, *normals, *cloudNormals);
                        pcl::search::KdTree<pcl::PointXYZINormal>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointXYZINormal>);
                        kdtree2->setInputCloud(cloudNormals);
                        pcl::GreedyProjectionTriangulation<pcl::PointXYZINormal> triangulation;
                        pcl::PolygonMesh triangles;
                        triangulation.setSearchRadius(triangulation_SearchRadius);
                        triangulation.setMu(triangulation_Mu);
                        triangulation.setMaximumNearestNeighbors(triangulation_MaximumNearestNeighbors);
                        triangulation.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees.
                        triangulation.setNormalConsistency(false);
                        triangulation.setMinimumAngle(M_PI / 18); // 10 degrees.
                        triangulation.setMaximumAngle(2 * M_PI / 3); // 120 degrees.
                        triangulation.setInputCloud(cloudNormals);
                        triangulation.setSearchMethod(kdtree2);
                        triangulation.reconstruct(triangles);

                        std::string vtxFileName = "/home/vision/catkin_ws/SearchRadius" + std::to_string(triangulation_SearchRadius) + "Mu" + std::to_string(triangulation_Mu) + "Ne" + std::to_string(triangulation_MaximumNearestNeighbors) + ".vtk";
                        pcl::io::savePolygonFileVTK(vtxFileName, triangles);
                    }
                }
            }
            break;
            
            // pcl::PointCloud<PointType>::Ptr inputcloudCache(new pcl::PointCloud<PointType>);
            // inputcloudCache->points.push_back(KPPC->points[0]);
            
            // pcl::ROPSEstimation<pcl::PointXYZI, pcl::Histogram<135>> rops;
            // rops.setInputCloud(inputcloudCache); 
            // rops.setSearchMethod(treeNe);
            // rops.setSearchSurface(SorroundPC);
            // rops.setRadiusSearch(rops_RadiusSearch);
            // rops.setTriangles(triangles.polygons);
            // rops.setNumberOfPartitionBins(rops_NumberOfPartitionBins);
            // rops.setNumberOfRotations(rops_NumberOfRotations);
            // rops.setSupportRadius(rops_SupportRadius); //이게 25mr(mesh resolution)이어야 한다. 즉, support_radius = 0.0285f;일 때 
            // //setRadiusSearch == setSupportRadius로 세팅되어 있다. 실험해볼 것
            // // 즉, 일단 대략적인 mesh resolution이 필요하다. 위의 save로 vik를 받아갔으니, 집에서 열어볼 것
            // pcl::PointCloud<pcl::Histogram<135>>::Ptr descriptors(new pcl::PointCloud<pcl::Histogram<135>>());
            // rops.compute(*descriptors);
            // pointDescriptorProcessedCount++;
            // // cout << "end of ROPSEstimation: " << descriptors->points[0].descriptorSize() << endl;

            // auto end_time = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            // processtimetotal += duration.count()/1000.0;
            // processtimeN++;

            // //pub하기
            // aloam_velodyne::KPAndDescriptor KPDmsg;
            
            // sensor_msgs::PointCloud2 KPmsg;
            // std_msgs::Float64MultiArray Dmsg;            
            // pcl::toROSMsg(*KPPC, KPmsg);

            // for (size_t i = 0; i < descriptors->points[0].descriptorSize(); i++) {
            //     Dmsg.data.push_back(descriptors->points[0].histogram[i]);
            // }

            // KPDmsg.keypoint_point_cloud = KPmsg;
            // KPDmsg.descriptor = Dmsg;
            // pubDescriptor.publish(KPDmsg);

            // sensor_msgs::PointCloud2 Smsg;
            // pcl::toROSMsg(*globalregienforDisplay, Smsg);
            // Smsg.header.frame_id = "/camera_init";
            // pubKeyregionDisplay.publish(Smsg);

            // cout << "[KeypointDescriptorDetectionProcess] 처리한 point 수: " << pointDescriptorProcessedCount << " || 남은 수: " << SorroundQue.size() << " || prcessing time: " << processtimetotal/processtimeN << "(ms)" << endl;
        }
        rate.sleep();
        break;
    }
    
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "alaserDescriptor");
	ros::NodeHandle nh;

    nh.param<float>("rops_RadiusSearch", rops_RadiusSearch, 1.0);
    nh.param<int>("rops_NumberOfPartitionBins", rops_NumberOfPartitionBins, 5);
    nh.param<int>("rops_NumberOfRotations", rops_NumberOfRotations, 3); 
    nh.param<float>("rops_SupportRadius", rops_SupportRadius, 1.0);

    nh.param<float>("triangulation_setSearchRadius", triangulation_SearchRadius, 0.03);
    nh.param<float>("triangulation_setMu", triangulation_Mu, 2.5); 
    nh.param<int>("triangulation_setMaximumNearestNeighbors", triangulation_MaximumNearestNeighbors, 50);

    // nh.param<float>("rops_RadiusSearch", rops_RadiusSearch, 1.0);
    // nh.param<int>("rops_NumberOfPartitionBins", rops_NumberOfPartitionBins, 5);
    // nh.param<int>("rops_NumberOfRotations", rops_NumberOfRotations, 3); 
    // nh.param<float>("rops_SupportRadius", rops_SupportRadius, 1.0);

    // ros::Subscriber subKeyLocalMap = nh.subscribe<aloam_velodyne::LocalMapAndPose>("/LGMLocalMap", 100, LocalMapHandler);
    // ros::Subscriber subAllPose = nh.subscribe<geometry_msgs::PoseArray>("/LGMAllPose", 100, PoseHandler);
    ros::Subscriber subKeyPointSurround = nh.subscribe<aloam_velodyne::KPAndSurroundPC>("/KPSurroundPC", 100, KPAndSurroundHandler);

	// pubLCdetectResult = nh.advertise<aloam_velodyne::LCPair>("/LCdetectResult", 100);
    // pubKeyPointResult = nh.advertise<aloam_velodyne::PointCloud2List>("/keyPointResult", 100);
    pubDescriptor = nh.advertise<aloam_velodyne::KPAndDescriptor>("/KPAndDescriptor", 100);
    pubKeyregionDisplay = nh.advertise<sensor_msgs::PointCloud2>("/keyregionDisplay2", 100);
    pubKeyregionDisplay3 = nh.advertise<sensor_msgs::PointCloud2>("/keyregionDisplay3", 100);

    std::thread threadKPDetect(KeypointDescriptorDetectionProcess);
    // std::thread threadKPMerge(KeypointMergingProcess);
    // std::thread threadKPDisplay(KeypointDisplayProcess);
    // std::thread threadKPDescriptor(KeypointDescriptorProcess);

 	ros::spin();

	return 0;
}
