#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/region_growing.h>

#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/features/fpfh.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Int64.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>

#include <aloam_velodyne/LCPair.h>
#include <aloam_velodyne/LocalMapAndPose.h>
#include <aloam_velodyne/PointCloud2List.h>
#include <aloam_velodyne/Float64MultiArrayArray.h>
#include <aloam_velodyne/MapAndDescriptors.h>


#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "scancontext/Scancontext.h"

using namespace gtsam;

using std::cout;
using std::endl;

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false; 

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 

std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;
std::queue<int> localMapIdxQueue;

std::mutex mBuf;
std::mutex mKF;
std::mutex mKeyPoint;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds;
std::vector<aloam_velodyne::Float64MultiArrayArray> descriptorsVector;

std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<Pose6D> keyframePosesForLocalMap;
std::vector<double> keyframeTimes;

std::vector<std::pair<int, int>> ICPPassedPair;
std::vector<std::pair<int, int>> HandlcPair;
// std::vector<int, int> ICPPassedPair;

std::vector<int> compareResultVec;

int recentIdxUpdated = 0;

gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
// SCManager scManager;
double scDistThres, scMaximumRadius;

pcl::VoxelGrid<PointType> downSizeFilterLocalMap;

pcl::VoxelGrid<PointType> downSizeFilterICP;
std::mutex mtxICP;
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudLocal(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr keypoints(new pcl::PointCloud<PointType>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypointsRGB(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<PointType>::Ptr localMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr globalkeypointvis(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr localkeypointvis(new pcl::PointCloud<PointType>);
std::vector<pcl::PointCloud<PointType>::Ptr> keypointsVector;
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
bool laserCloudMapPGORedraw = true;

bool useGPS = true;
// bool useGPS = false;
sensor_msgs::NavSatFix::ConstPtr currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

float LocalMapBoundary = 20.0; // pubLocalMap
int keypointsVectorSize = 0;

ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO, pubKeyPointglobal, pubKeyPointlocal, pubKeyPointglobalGraph, pubKeyPointlocalGraph;
ros::Publisher pubKeyPoseforLC, pubKeyLocalMapforLC, pubLoopScanLocal, pubLoopSubmapLocal, pubConstraintEdge, pubHandlc, pubLoopIcpResult, pubDisplayLocalMap;//, pubLoopIcpResult;
ros::Publisher pubOdomRepubVerifier;
ros::Publisher pubKeyFrameDS, pubDetectTrigger;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory, pgSCDsDirectory, pgLocalScansDirectory;
std::string odomKITTIformat;
std::fstream pgG2oSaveStream, pgTimeSaveStream;

std::vector<std::string> edges_str; // used in writeEdge

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p);
void pubLocalMap(void);
pcl::PointXYZ pointToPCL(const Pose6D& pose);
std::optional<gtsam::Pose3> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx );

pcl::PointCloud<PointType>::Ptr saveClosedPointCloud(const pcl::PointCloud<PointType> &cloud_in, Pose6D centerPoint, float thres)
{
    pcl::PointCloud<PointType>::Ptr cloud_out(new pcl::PointCloud<PointType>());
    cloud_out->header = cloud_in.header;
    cloud_out->points.resize(cloud_in.points.size());

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (std::pow(cloud_in.points[i].x - centerPoint.x, 2) + std::pow(cloud_in.points[i].y - centerPoint.y, 2) + std::pow(cloud_in.points[i].z - centerPoint.z, 2) < thres * thres)
            cloud_out->points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size()) { cloud_out->points.resize(j); }

    cloud_out->height = 1;
    cloud_out->width = static_cast<uint32_t>(j);
    cloud_out->is_dense = true;
    return cloud_out;
}


geometry_msgs::Pose gtsamPoseToROSPose(const Pose6D& gtsamPose) {
    geometry_msgs::Pose rosPose;

    gtsam::Pose3 pose = Pose6DtoGTSAMPose3(gtsamPose);

    // Translation
    rosPose.position.x = pose.x();
    rosPose.position.y = pose.y();
    rosPose.position.z = pose.z();

    // Quaternion rotation
    rosPose.orientation.x = pose.rotation().toQuaternion().x();
    rosPose.orientation.y = pose.rotation().toQuaternion().y();
    rosPose.orientation.z = pose.rotation().toQuaternion().z();
    rosPose.orientation.w = pose.rotation().toQuaternion().w();

    return rosPose;
}


std::string padZeros(int val, int num_digits = 6) 
{
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

std::string getVertexStr(const int _node_idx, const gtsam::Pose3& _Pose)
{
    gtsam::Point3 t = _Pose.translation();
    gtsam::Rot3 R = _Pose.rotation();

    std::string curVertexInfo {
        "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // pgVertexSaveStream << curVertexInfo << std::endl;
    // vertices_str.emplace_back(curVertexInfo);
    return curVertexInfo;
}

void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose, std::vector<std::string>& edges_str)
{
    gtsam::Point3 t = _relPose.translation();
    gtsam::Rot3 R = _relPose.rotation();

    std::string curEdgeInfo {
        "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // pgEdgeSaveStream << curEdgeInfo << std::endl;
    edges_str.emplace_back(curEdgeInfo);
}

void saveSCD(std::string fileName, Eigen::MatrixXd matrix, std::string delimiter = " ")
{
    // delimiter: ", " or " " etc.

    int precision = 3; // or Eigen::FullPrecision, but SCD does not require such accruate precisions so 3 is enough.
    const static Eigen::IOFormat the_format(precision, Eigen::DontAlignCols, delimiter, "\n");
 
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(the_format);
        file.close();
    }
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

void saveGTSAMgraphG2oFormat(const gtsam::Values& _estimates)
{
    // save pose graph (runs when programe is closing)
    // cout << "****************************************************" << endl; 
    cout << "Saving the posegraph ..." << endl; // giseop

    pgG2oSaveStream = std::fstream(save_directory + "singlesession_posegraph.g2o", std::fstream::out);

    int pose_idx = 0;
    for(const auto& _pose6d: keyframePoses) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);    
        pgG2oSaveStream << getVertexStr(pose_idx, pose) << endl;
        pose_idx++;
    }
    for(auto& _line: edges_str)
        pgG2oSaveStream << _line << std::endl;

    pgG2oSaveStream.close();
}

void saveOdometryVerticesKITTIformat(std::string _filename)
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);
    for(const auto& _pose6d: keyframePoses) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
{
    using namespace gtsam;

    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);

    for(const auto& key_value: _estimates) {
        auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
        if (!p) continue;

        const Pose3& pose = p->value();

        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(_laserOdometry);
	mBuf.unlock();
} // laserOdometryHandler

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
{
	mBuf.lock();
	fullResBuf.push(_laserCloudFullRes);
	mBuf.unlock();
} // laserCloudFullResHandler

void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &_gps)
{
    if(useGPS) {
        mBuf.lock();
        gpsBuf.push(_gps);
        mBuf.unlock();
    }
} // gpsHandler

// <aloam_velodyne/LCPair>
void LCHandler(const aloam_velodyne::LCPair::ConstPtr &_laserOdometry){
    mBuf.lock();
    scLoopICPBuf.push(std::pair<int, int>(_laserOdometry->a_int, _laserOdometry->b_int));
    // addding actual 6D constraints in the other thread, icp_calculation.
    mBuf.unlock();
}  // ScancontextHandler


void keyPointHandler(const aloam_velodyne::PointCloud2List::ConstPtr &_keyPoints){
    mKeyPoint.lock();
    keypointsVector.clear();
    descriptorsVector.clear();
    for (int i = 0; i < _keyPoints->size; i++) {
        pcl::PointCloud<PointType>::Ptr thisKeyPoints(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(_keyPoints->keypoints[i], *thisKeyPoints);
        keypointsVector.push_back(thisKeyPoints);
        aloam_velodyne::Float64MultiArrayArray descriptorsCache;
        for (int a = 0; a < thisKeyPoints->points.size(); a++){
            descriptorsCache.descriptor.push_back(_keyPoints->descriptors[i].descriptor[a]);
        }
        descriptorsVector.push_back(descriptorsCache);

        keypointsVectorSize = keypointsVector.size();
    }
    // _keyPoints 5째 노드의 키포인트의 2번째 포인트
    // cout <<"[[[[sub]"<< keypointsVector[5]->points[1] << "|||" << _keyPoints->descriptors[5].descriptor[1] <<"]]]]"<< endl;
    mKeyPoint.unlock();
} 

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    // odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );

    double bigNoiseTolerentToXY = 1000000000.0; // 1e9
    double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
    gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
    robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
    robustGPSNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises

Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw}; 
} // getOdom

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // SE3Diff

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);

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

pcl::PointCloud<PointType>::Ptr global2local(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
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


void pubPath( void )
{
    // pub odom and path 
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "/camera_init";
    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        const Pose6D& pose_est = keyframePosesUpdated.at(node_idx); // upodated poses
        // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);

        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "/camera_init";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x = pose_est.x;
        odomAftPGOthis.pose.pose.position.y = pose_est.y;
        odomAftPGOthis.pose.pose.position.z = pose_est.z;
        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "/camera_init";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    mKF.unlock(); 
    pubOdomAftPGO.publish(odomAftPGO); // last pose 
    pubPathAftPGO.publish(pathAftPGO); // poses 

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
    q.setW(odomAftPGO.pose.pose.orientation.w);
    q.setX(odomAftPGO.pose.pose.orientation.x);
    q.setY(odomAftPGO.pose.pose.orientation.y);
    q.setZ(odomAftPGO.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "/camera_init", "/aft_pgo"));
} // pubPath

void updatePoses(void)
{
    mKF.lock(); 
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        Pose6D& p =keyframePosesUpdated[node_idx];
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
    }
    mKF.unlock();

    mtxRecentPose.lock();
    const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    recentOptimizedX = lastOptimizedPose.translation().x();
    recentOptimizedY = lastOptimizedPose.translation().y();

    // recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

    mtxRecentPose.unlock();
} // updatePoses

void runISAM2opt(void)
{
    // called when a variable added 
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    updatePoses();
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
                                    transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(), 
                                    transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw() );
    
    int numberOfCores = 8; // TODO move to yaml 
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        pointFrom = &cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
} // transformPointCloud

void loopFindNearKeyframesCloud( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& submap_size, const int& root_idx)
{
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();
    for (int i = -submap_size; i <= submap_size; ++i) {
        int keyNear = key + i; // see https://github.com/gisbi-kim/SC-A-LOAM/issues/7 ack. @QiMingZhenFan found the error and modified as below. 
        if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()) )
            continue;

        mKF.lock(); 
        *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[root_idx]);
        mKF.unlock(); 
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
} // loopFindNearKeyframesCloud


std::optional<gtsam::Pose3> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx )
{   
    
    // parse pointclouds
    int historyKeyframeSearchNum = 25; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
    loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0, _loop_kf_idx); // use same root of loop kf idx 
    loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx); 

    // loop verification 
    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "/camera_init";
    pubLoopScanLocal.publish(cureKeyframeCloudMsg);

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "/camera_init";
    pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);


    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);
    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);    
 
    float loopFitnessScoreThreshold = 0.3; // user parameter but fixed low value is safe. 
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
        std::cout << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << _loop_kf_idx << ", " << _curr_kf_idx << std::endl;
    
        return std::nullopt;
    } else {
        std::cout << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << _loop_kf_idx << ", " << _curr_kf_idx << std::endl;
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFromR = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseToR = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
    gtsam::Pose3 relPoseR = poseFromR.between(poseToR);

    ICPPassedPair.push_back(std::make_pair(_loop_kf_idx, _curr_kf_idx));

    // // 절대 좌표 더해주기_loop_kf_idx, int _curr_kf_idx 
    // gtsam::Pose3 poseFromG = Pose6DtoGTSAMPose3(keyframePoses.at(_loop_kf_idx));
    // gtsam::Pose3 poseToG = Pose6DtoGTSAMPose3(keyframePoses.at(_curr_kf_idx));
    // gtsam::Pose3 relPoseG = poseFromG.between(poseToG);
    // // //
    // gtsam::Pose3 relPoseF = relPoseR * relPoseG;
    // //cout << relPoseR << relPoseG << relPoseF << endl; 
    // original -> return poseToR;
    return relPoseR;
} // doICPVirtualRelative

void process_pg()
{
    while(1)
    {   
        while ( !odometryBuf.empty() && !fullResBuf.empty() )
        {
            //
            // pop and check keyframe is or not  
            // 
			mBuf.lock();       
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
                odometryBuf.pop();
            if (odometryBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            // Time equal check
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timeLaser = fullResBuf.front()->header.stamp.toSec();
            // TODO
            laserCloudFullRes->clear();
            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
            fullResBuf.pop();

            Pose6D pose_curr = getOdom(odometryBuf.front());
            odometryBuf.pop();

            mBuf.unlock(); 

            //
            // Early reject by counting local delta movement (for equi-spereated kf drop)
            // 
            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value. 
            translationAccumulated += delta_translation;
            rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.  

            if( translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap ) {
                isNowKeyFrame = true;
                translationAccumulated = 0.0; // reset 
                rotaionAccumulated = 0.0; // reset 
            } else {
                isNowKeyFrame = false;
            }

            if( ! isNowKeyFrame ) 
                continue; 

            //
            // Save data and Add consecutive node 
            //

            pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
            downSizeFilterScancontext.setInputCloud(thisKeyFrame);
            downSizeFilterScancontext.filter(*thisKeyFrameDS);

            mKF.lock(); 
            keyframeLaserClouds.push_back(thisKeyFrameDS);
            keyframePoses.push_back(pose_curr);
            keyframePosesUpdated.push_back(pose_curr); // init
            keyframePosesForLocalMap.push_back(pose_curr);
            keyframeTimes.push_back(timeLaserOdometry);


            recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

            laserCloudMapPGORedraw = true;

            // sensor_msgs::PointCloud2 thisKeyFrameDSMsg;
            // pcl::toROSMsg(*thisKeyFrameDS, thisKeyFrameDSMsg);
            // thisKeyFrameDSMsg.header.frame_id = "/camera_init";
            // pubKeyFrameDS.publish(thisKeyFrameDSMsg);

            mKF.unlock();

            const int prev_node_idx = keyframePoses.size() - 2; 
            const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
            
            if( ! gtSAMgraphMade /* prior node */) {
                const int init_node_idx = 0; 
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));
                // auto poseOrigin = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

                mtxPosegraph.lock();
                {
                    // prior factor 
                    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
                    initialEstimate.insert(init_node_idx, poseOrigin);
                    // runISAM2opt();          
                }   
                mtxPosegraph.unlock();

                gtSAMgraphMade = true; 

                cout << "posegraph prior node " << init_node_idx << " added" << endl;
            } else /* consecutive node (and odom factor) after the prior added */ { // == keyframePoses.size() > 1 
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

                mtxPosegraph.lock();
                {
                    // odom factor
                    gtsam::Pose3 relPose = poseFrom.between(poseTo);
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relPose, odomNoise));

                    initialEstimate.insert(curr_node_idx, poseTo);                
                    writeEdge({prev_node_idx, curr_node_idx}, relPose, edges_str); // giseop
                    // runISAM2opt();
                }
                mtxPosegraph.unlock();

                if(curr_node_idx % 10 == 0)
                    cout << "posegraph odom node " << curr_node_idx << " added." << endl;
            }
            // if want to print the current graph, use gtSAMgraph.print("\nFactor Graph:\n");

            // save utility 
            // std::string curr_node_idx_str = padZeros(curr_node_idx);
            // pcl::io::savePCDFileBinary(pgScansDirectory + curr_node_idx_str + ".pcd", *thisKeyFrame); // scan 
            // // mKF.lock();
            // const auto& curr_scd = scManager.getConstRefRecentSCD();
            // // mKF.unlock();
            // saveSCD(pgSCDsDirectory + curr_node_idx_str + ".scd", curr_scd);

            // pgTimeSaveStream << timeLaser << std::endl; // path 

        }

        // ps. 
        // scan context detector is running in another thread (in constant Hz, e.g., 1 Hz)
        // pub path and point cloud in another thread

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg


pcl::PointXYZ pointToPCL(const Pose6D& pose) {
    pcl::PointXYZ pcl_point;
    pcl_point.x = pose.x;
    pcl_point.y = pose.y;
    pcl_point.z = pose.z;
    return pcl_point;
}

int currentKeyPoseIdxForLocalMap = 0;
int rangeOfKeyPoseIdxForLocalMap[2] = {0, 0};
int localMapPointsNum[2] = {1, 0};
std::vector<std::pair<int, pcl::PointCloud<PointType>::Ptr>> localMapCashe;

pcl::PointCloud<PointType>::Ptr CropPointcoud(pcl::PointCloud<PointType>::Ptr cloudIn, Pose6D localMapCenterPoint, float boundaryRange)
{

    pcl::PassThrough<PointType> pass;
    pass.setInputCloud (cloudIn);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (localMapCenterPoint.x - boundaryRange, localMapCenterPoint.x + boundaryRange);
    pass.filter (*cloudIn);

    pass.setFilterFieldName ("y");
    pass.setFilterLimits (localMapCenterPoint.y - boundaryRange, localMapCenterPoint.y + boundaryRange);
    pass.filter (*cloudIn);

    pass.setFilterFieldName ("z");
    pass.setFilterLimits (localMapCenterPoint.z - 10.0, localMapCenterPoint.z + 10.0);
    pass.filter (*cloudIn);

    return cloudIn;
}

void setNewLocalMap(Pose6D currentPose) {
    localMap->clear();
    
    mKF.lock();
    *localMap += *local2global(keyframeLaserClouds[currentKeyPoseIdxForLocalMap], currentPose);
    mKF.unlock();
    localMap = CropPointcoud(localMap, currentPose, LocalMapBoundary);
    // localMap = saveClosedPointCloud(*localMap, currentPose, LocalMapBoundary);

    localMapPointsNum[0] = 1;
    localMapPointsNum[1] = localMap->size();

    while(rangeOfKeyPoseIdxForLocalMap[0] > 0) {
        rangeOfKeyPoseIdxForLocalMap[0] -= 1;
        mKF.lock();
        *localMap += *local2global(keyframeLaserClouds[rangeOfKeyPoseIdxForLocalMap[0]], keyframePosesForLocalMap[rangeOfKeyPoseIdxForLocalMap[0]]);
        // *localMap += *saveClosedPointCloud(*local2global(keyframeLaserClouds[rangeOfKeyPoseIdxForLocalMap[0]], keyframePosesForLocalMap[rangeOfKeyPoseIdxForLocalMap[0]]), currentPose, LocalMapBoundary);
        mKF.unlock();
        // localMap = saveClosedPointCloud(*localMap, currentPose, LocalMapBoundary);
        localMap = CropPointcoud(localMap, currentPose, LocalMapBoundary);
        localMapPointsNum[0] = localMapPointsNum[1];
        localMapPointsNum[1] = localMap->size();
        if((float(localMapPointsNum[1])/float(localMapPointsNum[0]) < 1.1)) {
            break;
        }
    }
}

void pubLocalMap(void) {
    Pose6D currentPose = keyframePosesForLocalMap[currentKeyPoseIdxForLocalMap];
    if (rangeOfKeyPoseIdxForLocalMap[0] == rangeOfKeyPoseIdxForLocalMap[1]){
        setNewLocalMap(currentPose);
    }
    
    // 이거 하는 동안은 isam이 돌면 안됨.
    while(rangeOfKeyPoseIdxForLocalMap[1] < recentIdxUpdated) {
        rangeOfKeyPoseIdxForLocalMap[1] += 1;
        mKF.lock();
        *localMap += *local2global(keyframeLaserClouds[rangeOfKeyPoseIdxForLocalMap[1]], keyframePosesForLocalMap[rangeOfKeyPoseIdxForLocalMap[1]]);
        mKF.unlock();
        // localMap = saveClosedPointCloud(*localMap, currentPose, LocalMapBoundary);
        localMap = CropPointcoud(localMap, currentPose, LocalMapBoundary);
        localMapPointsNum[0] = localMapPointsNum[1];
        localMapPointsNum[1] = localMap->size();
        
        if((float(localMapPointsNum[1])/float(localMapPointsNum[0]) < 1.1)) {
            cout << "[LocalMap] Pub new localMap" << currentKeyPoseIdxForLocalMap <<": " << rangeOfKeyPoseIdxForLocalMap[0] << "||" << rangeOfKeyPoseIdxForLocalMap[1] << "|||" << float(localMapPointsNum[1])/float(localMapPointsNum[0]) << endl;

            localMap = saveClosedPointCloud(*localMap, currentPose, LocalMapBoundary);

            downSizeFilterLocalMap.setInputCloud(localMap);
            downSizeFilterLocalMap.filter(*localMap);

            // 이상치 제거 (Statistical Outlier Removal)
            pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
            sor.setInputCloud(localMap);
            sor.setMeanK(50); // 주변 이웃의 수
            sor.setStddevMulThresh(1.0); // 표준 편차의 배수
            sor.filter(*localMap);

            aloam_velodyne::LocalMapAndPose LocalMapAndPoseMsg;
            geometry_msgs::Pose rosPose = gtsamPoseToROSPose(currentPose);
            
            sensor_msgs::PointCloud2 localMapMsg;
            pcl::toROSMsg(*localMap, localMapMsg);
            localMapMsg.header.frame_id = "/camera_init";

            // 즉, 일정 범위의 맵을 포즈 값과 함께 전송한다.
            LocalMapAndPoseMsg.pose = rosPose;
            LocalMapAndPoseMsg.point_cloud = localMapMsg;
            LocalMapAndPoseMsg.idx = currentKeyPoseIdxForLocalMap;
            
            pubKeyLocalMapforLC.publish(LocalMapAndPoseMsg);
            pubDisplayLocalMap.publish(localMapMsg);

            currentKeyPoseIdxForLocalMap += 1;
            rangeOfKeyPoseIdxForLocalMap[0] = currentKeyPoseIdxForLocalMap;
            rangeOfKeyPoseIdxForLocalMap[1] = currentKeyPoseIdxForLocalMap;

            break;
        }
    }
}

void process_lcd(void)
{
    float loopClosureFrequency = 1.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        std_msgs::Int64 keyframePosesSize;
        keyframePosesSize.data = int(keyframePoses.size());

        // pubDetectTrigger.publish(keyframePosesSize);

    }
} // process_lcd

void process_icp(void)
{
    while(1)
    {
		while ( !scLoopICPBuf.empty() )
        {
            if( scLoopICPBuf.size() > 30 ) {
                ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
            }

            mBuf.lock(); 
            std::pair<int, int> loop_idx_pair = scLoopICPBuf.front();
            scLoopICPBuf.pop();
            mBuf.unlock(); 

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;
            auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
            if(relative_pose_optional) {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                mtxPosegraph.lock();
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
                writeEdge({prev_node_idx, curr_node_idx}, relative_pose, edges_str); // giseop
                // runISAM2opt();
                mtxPosegraph.unlock();
            } 
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
        // visualizeConstraint();
    }
} // process_icp

void process_viz_path(void)
{
    float hz = 10.0; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        // cout << "path publist process..." << endl;
        if(recentIdxUpdated > 1) {
            pubPath();
        }
    }   
}

void process_isam(void)
{
    float hz = 1; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if( gtSAMgraphMade ) {
            mtxPosegraph.lock();
            runISAM2opt();
            cout << "running isam2 optimization ..." << endl;
            mtxPosegraph.unlock();

            saveOptimizedVerticesKITTIformat(isamCurrentEstimate, pgKITTIformat); // pose
            saveOdometryVerticesKITTIformat(odomKITTIformat); // pose
            saveGTSAMgraphG2oFormat(isamCurrentEstimate);
        }
    }
}

int clusterCountVector[20];

void pubMap(void)
{
    int SKIP_FRAMES = 1; // sparse map visulalization to save computations 
    int counter = 0;

    laserCloudMapPGO->clear();

    mKF.lock(); 
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) {
        if(counter % SKIP_FRAMES == 0) {
            *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
        }
        counter++;
    }
    mKF.unlock(); 

    downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
    downSizeFilterMapPGO.filter(*laserCloudMapPGO);

    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "/camera_init";
    pubMapAftPGO.publish(laserCloudMapPGOMsg);
}

void pubKeyPoint(void) {
    globalkeypointvis->clear();
    localkeypointvis->clear();
    aloam_velodyne::MapAndDescriptors gloabalMapAndDescriptorsMsg; //여기에 같은 순서로 디스크립터도 채워야 함. 그전에 콜백에서 디스크립터 담기(그냥 메시지 벡터)
    aloam_velodyne::MapAndDescriptors localMapAndDescriptorsMsg;
    int gloabalPointSize = 0;
    int localPointSize = 0;
    mKeyPoint.lock();
    if ( keypointsVectorSize > 35 ){
        // gloabalMapAndDescriptorsMsg.size = keypointsVectorSize-50;
        for (int node_idx=0; node_idx < keypointsVectorSize-35; node_idx++) {
            *globalkeypointvis += *local2global(keypointsVector[node_idx], keyframePosesUpdated[node_idx]);//*keypointsVector[node_idx];
            std_msgs::Float64MultiArray descriptorsCashe;
            for (int a = 0; a < keypointsVector[node_idx]->size(); a++){
                gloabalMapAndDescriptorsMsg.descriptors.descriptor.push_back(descriptorsVector[node_idx].descriptor[a]);
                gloabalPointSize++;
            }
        }
        gloabalMapAndDescriptorsMsg.size = gloabalPointSize;
        // localMapAndDescriptorsMsg.size = 50;
        for (int node_idx=keypointsVectorSize-35; node_idx < keypointsVectorSize; node_idx++) {
            *localkeypointvis += *local2global(keypointsVector[node_idx], keyframePosesUpdated[node_idx]);//*keypointsVector[node_idx];
            for (int a = 0; a < keypointsVector[node_idx]->size(); a++){
                localMapAndDescriptorsMsg.descriptors.descriptor.push_back(descriptorsVector[node_idx].descriptor[a]);
                localPointSize++;
            }
        }
        localMapAndDescriptorsMsg.size = localPointSize;

        mKeyPoint.unlock();
        sensor_msgs::PointCloud2 KeypointMsg;
        pcl::toROSMsg(*globalkeypointvis, KeypointMsg);
        KeypointMsg.header.frame_id = "/camera_init";
        pubKeyPointglobal.publish(KeypointMsg);

        sensor_msgs::PointCloud2 KeypointLocalMsg;
        pcl::toROSMsg(*localkeypointvis, KeypointLocalMsg);
        KeypointLocalMsg.header.frame_id = "/camera_init";
        pubKeyPointlocal.publish(KeypointLocalMsg);

        gloabalMapAndDescriptorsMsg.keypoints = KeypointMsg;
        pubKeyPointglobalGraph.publish(gloabalMapAndDescriptorsMsg);

        localMapAndDescriptorsMsg.keypoints = KeypointLocalMsg;
        pubKeyPointlocalGraph.publish(localMapAndDescriptorsMsg);
    }
    else {
        aloam_velodyne::Float64MultiArrayArray::Ptr globalDescriptors(new aloam_velodyne::Float64MultiArrayArray());
        for (int node_idx=0; node_idx < keypointsVectorSize; node_idx++) {
            *globalkeypointvis += *local2global(keypointsVector[node_idx], keyframePosesUpdated[node_idx]);//*keypointsVector[node_idx];
            for (int a = 0; a < keypointsVector[node_idx]->size(); a++){
                gloabalMapAndDescriptorsMsg.descriptors.descriptor.push_back(descriptorsVector[node_idx].descriptor[a]);
                gloabalPointSize++;
            }
        }
        gloabalMapAndDescriptorsMsg.size = gloabalPointSize;
        mKeyPoint.unlock();

        sensor_msgs::PointCloud2 KeypointMsg;
        pcl::toROSMsg(*globalkeypointvis, KeypointMsg);
        KeypointMsg.header.frame_id = "/camera_init";
        pubKeyPointglobal.publish(KeypointMsg);

        gloabalMapAndDescriptorsMsg.keypoints = KeypointMsg;
        pubKeyPointglobalGraph.publish(gloabalMapAndDescriptorsMsg);
    }
}

void process_viz_map(void)
{
    int processCount = 0;
    float vizmapFrequency = 1.0; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {
            if(processCount > 10){
                pubMap();
                processCount = 0;
            }
            processCount++;
        }
        if(keypointsVectorSize > 1) {
            pubKeyPoint();
        }
    }
} // pointcloud_viz

void process_local_map(void){
    float localMapFrequency = 10.0; // 0.1 means run onces every 10s
    ros::Rate rate(localMapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 0){
            pubLocalMap();
        }
    }
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;
    // save directories 
	// nh.param<std::string>("save_directory", save_directory, "/"); // pose assignment every k m move 
    nh.param<std::string>("save_directory", save_directory, "/home/user/Documents/scaloam_scd_saver/data/");
    // save_directory = "/home/user/Documents/scaloam_scd_saver/data/";
    pgKITTIformat = save_directory + "optimized_poses.txt";
    odomKITTIformat = save_directory + "odom_poses.txt";

    // pgG2oSaveStream = std::fstream(save_directory + "singlesession_posegraph.g2o", std::fstream::out);

    pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out); 
    pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);

    pgScansDirectory = save_directory + "Scans/";
    auto unused = system((std::string("exec rm -r ") + pgScansDirectory).c_str());
    unused = system((std::string("mkdir -p ") + pgScansDirectory).c_str());

    pgSCDsDirectory = save_directory + "SCDs/"; // SCD: scan context descriptor 
    unused = system((std::string("exec rm -r ") + pgSCDsDirectory).c_str());
    unused = system((std::string("mkdir -p ") + pgSCDsDirectory).c_str());

    pgLocalScansDirectory = save_directory + "LocalSCDs/"; // SCD: scan context descriptor 
    unused = system((std::string("exec rm -r ") + pgLocalScansDirectory).c_str());
    unused = system((std::string("mkdir -p ") + pgLocalScansDirectory).c_str());

    // system params 
	nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0); // pose assignment every k m move 
	nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot 
    keyframeRadGap = deg2rad(keyframeDegGap);

    nh.param<float>("LocalMapBoundary", LocalMapBoundary, 30.0);

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    float filter_size = 0.3; 
    float localMapFilterSize = 0.2; 
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterLocalMap.setLeafSize(localMapFilterSize, localMapFilterSize, localMapFilterSize);

    double mapVizFilterSize;
	nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.2); // pose assignment every k frames 
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);
    

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered_local", 100, laserCloudFullResHandler);
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, laserOdometryHandler);
	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);

    ros::Subscriber subLCdetectResult = nh.subscribe<aloam_velodyne::LCPair>("/LCdetectResult", 100, LCHandler); // std::pair<int, float> 가 리턴
    ros::Subscriber subKeyPoint = nh.subscribe<aloam_velodyne::PointCloud2List>("/keyPointResult", 100, keyPointHandler);

	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);
	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
	pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);

    pubKeyPointglobal = nh.advertise<sensor_msgs::PointCloud2>("/LGM_keypointGlobal", 100);
    pubKeyPointlocal = nh.advertise<sensor_msgs::PointCloud2>("/LGM_keypointLocal", 100);
    pubKeyLocalMapforLC = nh.advertise<aloam_velodyne::LocalMapAndPose>("/LGMLocalMap", 100);
    pubKeyPoseforLC = nh.advertise<geometry_msgs::PoseArray>("/LGMAllPose", 100);
    pubDisplayLocalMap = nh.advertise<sensor_msgs::PointCloud2>("/DisplayLGMLocalMap", 100);

    pubKeyPointglobalGraph = nh.advertise<aloam_velodyne::MapAndDescriptors>("/keypointGlobalGraph", 100);
    pubKeyPointlocalGraph = nh.advertise<aloam_velodyne::MapAndDescriptors>("/keypointLocalGraph", 100);

    pubKeyFrameDS = nh.advertise<sensor_msgs::PointCloud2>("/KeyFrameDSforLC", 100);

	pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);
    // pubLoopIcpResult = nh.advertise<sensor_msgs::PointCloud2>("/loop_icpResult", 100);

    // pubConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/sc_match_constraints", 10);
    // pubHandlc = nh.advertise<visualization_msgs::MarkerArray>("/hand_lc_point", 10);

	std::thread posegraph_slam {process_pg}; // pose graph construction
    // std::thread key_point {process_keypoints};
	std::thread lc_detection {process_lcd}; // loop closure detection 
	std::thread icp_calculation {process_icp}; // loop constraint calculation via icp 
	std::thread isam_update {process_isam}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added. 
    std::thread local_map {process_local_map}; // visualization - path (high frequency)
	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

 	ros::spin();

	return 0;
}
