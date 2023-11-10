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
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

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

std::mutex mBuf;
std::mutex mKF;
std::mutex mSur;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds;

std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
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
SCManager scManager;
double scDistThres, scMaximumRadius;

pcl::VoxelGrid<PointType> downSizeFilterICP;
std::mutex mtxICP;
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudLocal(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr keypoints(new pcl::PointCloud<PointType>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypointsRGB(new pcl::PointCloud<pcl::PointXYZRGB>);
// pcl::PointCloud<PointType>::Ptr keypointsMap(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr globalkeypointvis(new pcl::PointCloud<PointType>);
std::vector<pcl::PointCloud<PointType>::Ptr> keypointsVector;
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
pcl::VoxelGrid<PointType> downSizeFilterLocal;
bool laserCloudMapPGORedraw = true;

bool useGPS = true;
// bool useGPS = false;
sensor_msgs::NavSatFix::ConstPtr currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO, pubKeyPoint, pubKeyPointMap;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal, pubConstraintEdge, pubHandlc, pubLoopIcpResult;//, pubLoopIcpResult;
ros::Publisher pubOdomRepubVerifier;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory, pgSCDsDirectory, pgLocalScansDirectory;
std::string odomKITTIformat;
std::fstream pgG2oSaveStream, pgTimeSaveStream;

std::vector<std::string> edges_str; // used in writeEdge

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
    // cout << "[process_icp]                                  icp process test start" << endl;
    // sensor_msgs::PointCloud2 IcpResultCloudMsg;
    // pcl::toROSMsg(*unused_result, IcpResultCloudMsg);
    // IcpResultCloudMsg.header.frame_id = "/camera_init";
    // pubLoopIcpResult.publish(IcpResultCloudMsg);
    // cout << "[process_icp]                                  icp process test end" << endl;
    
 
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
            keyframeTimes.push_back(timeLaserOdometry);

            scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);

            recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

            laserCloudMapPGORedraw = true;
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

void performSCLoopClosure(void)
{
    mKF.lock();
    if( int(keyframePoses.size()) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        {   mKF.unlock();
        return;}

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
    mKF.unlock();
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 ) { 
        const int prev_node_idx = SCclosestHistoryFrameID;
        const int curr_node_idx = keyframePoses.size() - 1; // because cpp starts 0 and ends n-1
        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

        mBuf.lock();
        scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
        // addding actual 6D constraints in the other thread, icp_calculation.
        mBuf.unlock();
    }
} // performSCLoopClosure

float ISS_SalientRadius = 10;
float ISS_NonMaxRadius = 6;
float ISS_Gamma21 = 0.9;
float ISS_Gamma23 = 0.9;
int ISS_MinNeighbors = 10;
int Local_map_idx = 6;
float Local_map_boundary = 25.0;
pcl::PointXYZ lastCenterPoint = pcl::PointXYZ(0, 0, 0);


pcl::PointXYZ pointToPCL(const Pose6D& pose)
{
    pcl::PointXYZ pcl_point;
    pcl_point.x = pose.x;
    pcl_point.y = pose.y;
    pcl_point.z = pose.z;
    return pcl_point;
}

double calculateDistance(const pcl::PointXYZ& point1, const pcl::PointXYZ& point2)
{
    double dx = point1.x - point2.x;
    double dy = point1.y - point2.y;
    double dz = point1.z - point2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void surround_keypoint_detection( int Idx ){
    // cout << "[surround_keypoint_detection] process..." << endl;
    // LocalCloud 초기화
    laserCloudLocal->clear();

    // Local_map_idx(키프레임 범위) 범위 중 중간 keyframe 인덱스 기준으로 Local_map_boundary만큼 크롭 -> LocalCloud에 저장
    mKF.lock();
    pcl::PointXYZ pclCenterPoint = pointToPCL(keyframePosesUpdated[Idx - (Local_map_idx/2)]);
    for (int node_idx = int(Idx - Local_map_idx); node_idx <= Idx; node_idx++) {
        *laserCloudLocal += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
    }
    mKF.unlock();

    // 오브젝트 생성 
    pcl::PassThrough<PointType> pass;
    pass.setInputCloud (laserCloudLocal);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (pclCenterPoint.x - Local_map_boundary, pclCenterPoint.x + Local_map_boundary);
    pass.filter (*laserCloudLocal);

    pass.setFilterFieldName ("y");
    pass.setFilterLimits (pclCenterPoint.y - Local_map_boundary, pclCenterPoint.y + Local_map_boundary);
    pass.filter (*laserCloudLocal);

    pass.setFilterFieldName ("z");
    pass.setFilterLimits (pclCenterPoint.z - 10.0, pclCenterPoint.z + 10.0);
    pass.filter (*laserCloudLocal);

    // downSizeFilterLocal.setLeafSize(0.05, 0.05, 0.05);
    // downSizeFilterLocal.setInputCloud(laserCloudLocal);
    // downSizeFilterLocal.filter(*laserCloudLocal);

    // 키포인트 추출
    pcl::PointCloud<PointType>::Ptr currkeypoints(new pcl::PointCloud<PointType>);

    // [] ISS keypoint 추출
	pcl::ISSKeypoint3D<PointType, PointType> detector;
	detector.setInputCloud(laserCloudLocal);
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

    // 바운더리 근처의 키포인트는 제거
    pass.setInputCloud (currkeypoints);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (pclCenterPoint.x - (2*Local_map_boundary/3), pclCenterPoint.x + (2*Local_map_boundary/3));
    pass.filter (*currkeypoints);

    pass.setFilterFieldName ("y");
    pass.setFilterLimits (pclCenterPoint.y - (2*Local_map_boundary/3), pclCenterPoint.y + (2*Local_map_boundary/3));
    pass.filter (*currkeypoints);

    pass.setFilterFieldName ("z");
    pass.setFilterLimits (pclCenterPoint.z - 10.0, pclCenterPoint.z + 10.0);
    pass.filter (*currkeypoints);

    cout << "Keypoint's num : " << currkeypoints->points.size() << endl;

    *currkeypoints = *global2local(currkeypoints, keyframePosesUpdated[Idx]);
    keypointsVector.push_back(currkeypoints);




    // // 클러스터 추출 파라미터 설정
    // pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    // tree->setInputCloud(keypoints);

    // std::vector<pcl::PointIndices> cluster_indices;
    // pcl::EuclideanClusterExtraction<PointType> ec;
    // ec.setClusterTolerance(1.0); // 클러스터의 거리 허용 오차 (조절 가능)
    // ec.setMinClusterSize(1);    // 클러스터의 최소 크기 (조절 가능)
    // ec.setMaxClusterSize(100);  // 클러스터의 최대 크기 (조절 가능)
    // ec.setSearchMethod(tree);
    // ec.setInputCloud(keypoints);
    // ec.extract(cluster_indices);

    // currkeypoints->clear();

    // // 각 클러스터의 중심점을 찾고 추출
    // for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
    //     pcl::PointCloud<PointType>::Ptr cluster(new pcl::PointCloud<PointType>);

    //     for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
    //         cluster->points.push_back(keypoints->points[*pit]);
    //     }

    //     // 중심점 계산
    //     Eigen::Vector4f centroid;
    //     pcl::compute3DCentroid(*cluster, centroid);
        
    //     PointType cluster_center;
    //     cluster_center.x = centroid[0];
    //     cluster_center.y = centroid[1];
    //     cluster_center.z = centroid[2];
        
    //     currkeypoints->push_back(cluster_center);
    // }
    // keypoints = currkeypoints;


    // // [] Normal 생성
    // pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>()); // 352 size vector descriptor
    // // pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());
	
	// // Estimate the normals.
	// pcl::NormalEstimation<PointType, pcl::Normal> normalEstimation;
    // normalEstimation.setInputCloud(laserCloudLocal);
    // normalEstimation.setKSearch(50);
    // // normalEstimation.setRadiusSearch(5.0);
	// kdtree.reset(new pcl::search::KdTree<PointType>);
	// normalEstimation.setSearchMethod(kdtree);
	// normalEstimation.compute(*normals);

    // // [] SHOT estimation object.
	// pcl::SHOTEstimation<PointType, pcl::Normal, pcl::SHOT352> shot;
    // shot.setInputCloud(keypoints);
	// shot.setInputNormals(normals);
	// // The radius that defines which of the keypoint's neighbors are described.
	// // If too large, there may be clutter, and if too small, not enough points may be found.
    // shot.setSearchMethod (kdtree);
    // shot.setRadiusSearch(3.0);
    // shot.compute(*descriptors);

    // // [] SHOT 디스크립터 출력
    // for (size_t i = 0; i < descriptors->points.size(); ++i) {
    //     pcl::SHOT352 descriptor = descriptors->points[i];
    //     cout << "SHOT Descriptor " << i << ": ";
    //     for (size_t j = 0; j < 352; ++j) {
    //         cout << descriptor.descriptor[j] << " ";
    //     }
    //     cout << std::endl;
    // }

    sensor_msgs::PointCloud2 KeypointMapMsg;
    pcl::toROSMsg(*laserCloudLocal, KeypointMapMsg);
    KeypointMapMsg.header.frame_id = "/camera_init";
    pubKeyPointMap.publish(KeypointMapMsg);

} 

int recentIdxprocessed = Local_map_idx;
void process_keypoints(void){
    float loopClosureFrequency = 10.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    for (int i = 0; i < Local_map_idx; i++){
        pcl::PointCloud<PointType>::Ptr nonkeypoints(new pcl::PointCloud<PointType>);
        keypointsVector.push_back(nonkeypoints);
    }
    while (ros::ok()){
        rate.sleep();
        if (recentIdxprocessed < recentIdxUpdated) {
            // cout << recentIdxprocessed << endl;
            surround_keypoint_detection(recentIdxprocessed);
            recentIdxprocessed++;
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
        performSCLoopClosure();
        // handmadeLoopClosure();

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



void pubMap(void)
{
    int SKIP_FRAMES = 1; // sparse map visulalization to save computations 
    int counter = 0;

    laserCloudMapPGO->clear();
    keypointsRGB->clear();
    globalkeypointvis->clear();

    mKF.lock(); 
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) {
        if(counter % SKIP_FRAMES == 0) {
            *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
        }
        counter++;
    }
    for (int node_idx=Local_map_idx; node_idx < keypointsVector.size(); node_idx++) {
        *globalkeypointvis += *local2global(keypointsVector[node_idx], keyframePosesUpdated[node_idx]);
    }
    mKF.unlock(); 

    downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
    downSizeFilterMapPGO.filter(*laserCloudMapPGO);

    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
    tree->setInputCloud (globalkeypointvis);

    std::vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<PointType> ec;
    ec.setClusterTolerance (0.5);
    ec.setMinClusterSize (12);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (globalkeypointvis);
    ec.extract (cluster_indices);

    keypointsRGB->width = globalkeypointvis->width;
    keypointsRGB->height = globalkeypointvis->height;
    keypointsRGB->is_dense = false;
    keypointsRGB->points.resize(globalkeypointvis->points.size());

    // 각 클러스터에 랜덤 색상 부여
    for (const auto& cluster : cluster_indices) {
        // 클러스터 내의 각 인덱스에 대해 작업
        int r = rand() % 256;
        int g = rand() % 256;
        int b = rand() % 256;
        for (const auto& idx : cluster.indices) {
            pcl::PointXYZI pointI = globalkeypointvis->points[idx];
            pcl::PointXYZRGB pointRGB;

            // 인텐시티 값을 0-255 범위로 매핑
            int intensity = static_cast<int>(pointI.intensity * 255);

            // R, G, B 색상을 설정
            pointRGB.r = r;
            pointRGB.g = g;
            pointRGB.b = b;

            pointRGB.x = pointI.x;
            pointRGB.y = pointI.y;
            pointRGB.z = pointI.z;

            keypointsRGB->points[idx] = pointRGB;
        }
    }
    sensor_msgs::PointCloud2 KeypointMsg;
    pcl::toROSMsg(*keypointsRGB, KeypointMsg);
    KeypointMsg.header.frame_id = "/camera_init";
    pubKeyPoint.publish(KeypointMsg);

    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "/camera_init";
    pubMapAftPGO.publish(laserCloudMapPGOMsg);
}


void process_viz_map(void)
{
    float vizmapFrequency = 0.1; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {
            pubMap();
        }
    }
} // pointcloud_viz

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

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 

    nh.param<float>("ISS_SalientRadius", ISS_SalientRadius, 10.0);
	nh.param<float>("ISS_NonMaxRadius", ISS_NonMaxRadius, 6.0); 
    nh.param<float>("ISS_Gamma21", ISS_Gamma21, 0.9);
	nh.param<float>("ISS_Gamma23", ISS_Gamma23, 0.9);
    nh.param<int>("ISS_MinNeighbors", ISS_MinNeighbors, 50);

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.4; 
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);
    

    double mapVizFilterSize;
	nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4); // pose assignment every k frames 
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered_local", 100, laserCloudFullResHandler);
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, laserOdometryHandler);
	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);
    // ros::Subscriber subLaserCloudSurround = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surround", 100, cloud_surround_Handler);

	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);
	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
	pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);
    pubKeyPoint = nh.advertise<sensor_msgs::PointCloud2>("/LGM_keypoint", 100);
    pubKeyPointMap = nh.advertise<sensor_msgs::PointCloud2>("/LGM_keypoint_Map", 100);

	pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);
    // pubLoopIcpResult = nh.advertise<sensor_msgs::PointCloud2>("/loop_icpResult", 100);

    pubConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/sc_match_constraints", 10);
    pubHandlc = nh.advertise<visualization_msgs::MarkerArray>("/hand_lc_point", 10);

	std::thread posegraph_slam {process_pg}; // pose graph construction
    std::thread key_point {process_keypoints};
	std::thread lc_detection {process_lcd}; // loop closure detection 
	std::thread icp_calculation {process_icp}; // loop constraint calculation via icp 
	std::thread isam_update {process_isam}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added. 

	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

 	ros::spin();

	return 0;
}
