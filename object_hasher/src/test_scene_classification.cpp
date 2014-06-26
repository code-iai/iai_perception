/*
 * test_scene_classification.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: bbferka
 */

#include <pcl/io/pcd_io.h>
#include <iostream>
#include <object_hasher/ClassifyScene.h>
#include <pcl/point_types.h>
#include <object_hasher/point_type.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>


int main(int argc, char** argv)
{

  if(argc<2)
  {
    std::cerr<<"please specify file needed."<<std::endl;
    exit(0);
  }
  std::string filename (argv[1]);
  ros::init(argc, argv, "classify_scene_client");

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZLRegion>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZLRegion>());

  if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(filename, *cloud_in) == -1) //* load the file
  {
    PCL_ERROR("Couldn't read file test_pcd.pcd \n");
    exit(0);
  }
  sensor_msgs::PointCloud2 in_cloud_blob, out_cloud_blob;
  pcl::toROSMsg(*cloud_in, in_cloud_blob);


  ros::NodeHandle n("~");
  ros::ServiceClient client = n.serviceClient<object_hasher::ClassifyScene>("/classifier/classify_scene");
  object_hasher::ClassifyScene srv;
  srv.request.in_cloud = in_cloud_blob;
  //srv.request.ID=0;
  srv.request.params="-n 5 -s 0 -r 0.01";
  if (client.call(srv))
  {
    ROS_ERROR("Calling classify scene service");
    out_cloud_blob = srv.response.out_cloud;
    pcl::fromROSMsg(out_cloud_blob,*cloud_out);
    ROS_INFO("Response Clouds Size: %ld", cloud_out->points.size());
    pcl::io::savePCDFile("/tmp/result.pcd",*cloud_out);
  }
  else
  {
    ROS_ERROR("Failed to call service classify_scene");
    return 1;
  }
  return 0;
}


