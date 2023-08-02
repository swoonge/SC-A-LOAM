#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <ceres/ceres.h>
#include "glog/logging.h"

#include <iostream>

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

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct CostFunctor {
  template <typename T> bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lib_test");
	ros::NodeHandle nh;

    ////////////////////// ceres test //////////////////////
    google::InitGoogleLogging(argv[0]);

    // The variable to solve for with its initial value. It will be
    // mutated in place by the solver.
    double x = 0.5;
    const double initial_x = x;
    // Build the problem.
    Problem problem;
    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).
    CostFunction* cost_function =
        new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(cost_function, NULL, &x);
    // Run the solver!
    Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "x : " << initial_x
            << " -> " << x << "\n";
    ////////////////////// ceres test //////////////////////

    ////////////////////// pcl test //////////////////////
    // pcl::PointCloud<pcl::PointXYZ> cloud;

    // cloud.width = 10000;

    // cloud.height = 1;

    // cloud.is_dense = false;

    // cloud.points.resize(cloud.width * cloud.height);

    // for (size_t i = 0; i < cloud.points.size(); ++i) {

    //     cloud.points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);

    //     cloud.points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);

    //     cloud.points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);

    // }

 
    // for (size_t i = 0; i < cloud.points.size(); ++i)

    //     std::cerr << " " << cloud.points[i].x << " " << cloud.points[i].y << " " << cloud.points[i].z << std::endl;

 

    // pcl::visualization::CloudViewer viewer("PCL Viewer");

    // viewer.showCloud(cloud.makeShared());

    // while (!viewer.wasStopped());

    ////////////////////// pcl test //////////////////////

     // 두 개의 3x3 행렬 생성
    gtsam::Matrix A = gtsam::Matrix::Identity(3, 3);
    gtsam::Matrix B = gtsam::Matrix::Zero(3, 3);

    // 두 행렬을 더해서 결과 행렬 C 생성
    gtsam::Matrix C = A + B;

    // 결과 행렬 C 출력
    std::cout << "Resultant Matrix C:\n" << C << std::endl;

    ///////////

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->width = 5;
    cloud->height = 1;
    cloud->points.resize(cloud->width * cloud->height);
    
    // PointCloud 데이터 채우기
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        cloud->points[i].x = static_cast<float>(i);
        cloud->points[i].y = static_cast<float>(i * 2);
        cloud->points[i].z = static_cast<float>(i * 0.5);
    }

    // PCD 파일로 저장
    pcl::io::savePCDFileASCII("test_cloud.pcd", *cloud);
    std::cout << "PointCloud 데이터가 저장되었습니다." << std::endl;

    // PointCloud 데이터 출력
    std::cout << "PointCloud 데이터 출력:" << std::endl;
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        std::cout << "  " << cloud->points[i].x
                  << " " << cloud->points[i].y
                  << " " << cloud->points[i].z << std::endl;
    }

    return 0;
}