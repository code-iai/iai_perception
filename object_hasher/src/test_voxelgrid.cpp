#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

int
main (int argc, char** argv)
{
  sensor_msgs::PointCloud2::Ptr cloud (new sensor_msgs::PointCloud2 ());
  sensor_msgs::PointCloud2::Ptr cloud_filtered (new sensor_msgs::PointCloud2 ());

  // Fill in the cloud data
  // Replace the path below with the path where you saved your file
  pcl::io::loadPCDFile("data/0.pcd", *cloud); // Remember to download the file first!

  std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height 
       << " data points (" << pcl::getFieldsList (*cloud) << ").";

  // Create the filtering object
  pcl::PCLPointCloud2Ptr
    pcl_cloud(new pcl::PCLPointCloud2()),
    pcl_cloud_filtered(new pcl::PCLPointCloud2());
  pcl_conversions::toPCL(*cloud, *pcl_cloud);

  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (pcl_cloud);
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*pcl_cloud_filtered);

  pcl_conversions::fromPCL(*pcl_cloud_filtered, *cloud_filtered);

  std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height 
       << " data points (" << pcl::getFieldsList (*cloud_filtered) << ").";

  pcl::io::savePCDFile("data/cloud_downsampled.pcd", *cloud_filtered,
         Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), false);

  return (0);
}
