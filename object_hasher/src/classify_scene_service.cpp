/*
 * classify_scene_service.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: bbferka
 */

#include <iostream>
#include <ros/ros.h>
#include <icf_core/client/Client.h>
#include <icf_core/service/ClassifierManager.h>

#include <object_hasher/ObjectPartDecomposition.hpp>

//ros service
#include <object_hasher/ClassifyScene.h>

//#include <ias_classifier_client/OcClient.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>

#include <pcl_conversions/pcl_conversions.h>

#define BOOST_FILESYSTEM_VERSION 2
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <ros/package.h>

using namespace pcl;
using namespace pcl::io;
using namespace std;

typedef pcl::PointXYZRGB PointT;

double tolerance = 0.020;
double max_curvature = 0.01;
int use_srand = 0;
bool debug_mode = false;
int min_points_in_part = 100;
int feature_type = 0;
bool calc_normals = true;
int other = 0;
int max_nr_of_grouped_segments = 8;
int save_segmented_cloud = 0;
double vicinity_threshold = 0.015;

std::string c_type, path, base;

bool g_stop = false;
bool g_hasStopped;
bool first = true;

icf::ClassifierClient *client;

void managerThread()
{
  ros::Rate r(10);
  while(!g_stop)
  {
    ros::spinOnce();
    r.sleep();
  }
  g_hasStopped = true;
}

void startManagerThread()
{
  boost::thread thread(&managerThread);
}

void process(pcl::PointCloud<PointT>::Ptr cloud,
             pcl::PointCloud<pcl::PointXYZLRegion>::Ptr result_cloud)
{
  //needed for the client
  std::stringstream modelPath;
  std::string packagePath ;//= "object_hasher";
  packagePath = ros::package::getPath("object_hasher");

  modelPath << packagePath << "/data/rgbd_corrected_grsd.hdf5";
  std::cerr<<"Model Path: "<<modelPath.str();
  ros::NodeHandle nh_client;
  nh_client.param<std::string>("classification_type", c_type, "oph");
  nh_client.param<std::string>("base", base, "/ias_classifier_manager");
  nh_client.getParam("path", path);
  //OcClient *oc = new OcClient(nh_client);

  std::string manager_name("ias_classifier_manager");


  std::cerr << "New Client created" << std::endl;
  icf::ServerSideRepo data_store(nh_client, manager_name);
  if(first)
  {
    client = new icf::ClassifierClient(nh_client, manager_name, "oph", "0");
    icf::DS ds_train(modelPath.str(), false, true);
    data_store.uploadData(ds_train, "train");
    std::cerr << "Uploading to server" << std::endl;
    client->assignData("train", icf::Train);
    client->train("");
    std::cerr << "Client Trained!" << std::endl;
    first = false;
  }

  pcl::PointCloud<PointT>::Ptr cloud_for_opd(new pcl::PointCloud<PointT>());
  pcl::copyPointCloud(*cloud, *cloud_for_opd);

  std::cerr << "Size of cloud to be processed: " << cloud->points.size() << std::endl;
  //TODO get rid of this somehow

  pcl::PointCloud<PointT>::Ptr cloud_sor_out(new pcl::PointCloud<PointT>());
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(cloud);
  sor.setMeanK(10);
  sor.setStddevMulThresh(2.0);
  std::cerr << "before filtering" << std::endl;
  sor.filter(*cloud);
  //decomposing

  std::cerr << "Decomposition in progress!" << std::endl;
  ObjectPartDecomposition<PointT> *opd = new ObjectPartDecomposition<PointT>(path, use_srand);
  opd->setInputCloud(cloud_for_opd);
  opd->setfilename("/tmp/cloud.pcd");
  opd->setNrOfGroupedParts(max_nr_of_grouped_segments);
  if(save_segmented_cloud)
  {
    opd->saveSegmentedCloud(true);
  }
  opd->setVicinityThreshold(vicinity_threshold);
  std::cerr << "Set all parameters!" << std::endl;

  std::vector<ObjectGroup> possible_groups = opd->getFeatures(-1, tolerance, max_curvature, min_points_in_part,
      feature_type, calc_normals); //0 object ID, because it is a scene we don't consider it

  std::cerr << "Found all possible groupings!" << std::endl;
  if(possible_groups.size() < 1)
  {
    ROS_ERROR("NO GROUPS/SEGMENTS FOUND FOR THIS OBJECT! EXITING!");
    exit(0);
  }
  opd->computeArrangementKey(possible_groups);
  std::cerr << "Number of possible groupings in scene: " << possible_groups.size() << std::endl;
  for(unsigned int j = 0; j < possible_groups.size(); ++j)
  {
    possible_groups[j].determinePartIdsList();
  }

  //create DS for testing data!
  icf::DS ds_test;

  if(possible_groups.size() > 0)
  {
    stringstream out;
    icf::DS::Matrix size_of_parts = icf::DS::Matrix(opd->clusters.size(), 1); //nr of point in parts
    icf::DS::Matrix lMatrix = icf::DS::Matrix(possible_groups.size(), 1); //label of the grouping TODO this can have the size 1.
    icf::DS::Matrix aMatrix = icf::DS::Matrix(possible_groups.size(), 1); //arrangement of the grouping
    icf::DS::Matrix part_NrMatrix = icf::DS::Matrix(possible_groups.size(), 1); //nr of parts in the grouping
    icf::DS::Matrix partIdsMatrix = icf::DS::Matrix(possible_groups.size(), opd->clusters.size() + 1); //ids of parts in the grouping
    icf::DS::Matrix group_nr_of_points = icf::DS::Matrix(possible_groups.size(), 1); //nr of points in the grouping
    icf::DS::Matrix grown_from = icf::DS::Matrix(possible_groups.size(), 1); //parent node of this grouping
    icf::DS::Matrix extractedFeatures = icf::DS::Matrix(possible_groups.size(), possible_groups[0].descriptor_.rows()); //feature descriptor of the grouping
    std::cerr << "SET THE SIZE OF THE FEATURE MATRIX: " << possible_groups.size() << " x " << possible_groups[0].descriptor_.rows() << std::endl;
    //    out << opd->clusters.size() << std::endl;
    std::cerr << "Size of the part IDs matrix: " << partIdsMatrix.rows() << " x " << partIdsMatrix.cols() << std::endl;

    for(unsigned int j = 0; j < opd->clusters.size(); ++j)
      //      out << opd->clusters[j]->indices.size() << ",";
    {
      size_of_parts(j, 0) = opd->clusters[j]->indices.size();
    }
    //        out << std::endl;

    for(unsigned int j = 0; j < possible_groups.size(); ++j)
    {

      lMatrix(j, 0) =  possible_groups[j].ID_ ;
      part_NrMatrix(j, 0) = possible_groups[j].part_nr_;
      aMatrix(j, 0) =  possible_groups[j].arrangement_key_;
      //for (int k=0;k< possible_groups[j].partIDs.rows();++k)
      partIdsMatrix(j, 0) = possible_groups[j].partIDs.rows();
      for(int k = 1; k <= possible_groups[j].partIDs.rows(); ++k)
      {
        partIdsMatrix(j, k) = possible_groups[j].partIDs(k - 1);
      }

      //      out << "(" << possible_groups[j].ID_ << "," << possible_groups[j].part_nr_ << ","
      //          << possible_groups[j].arrangement_key_ << ",";
      //      out << possible_groups[j].partIDs.transpose() << "," << possible_groups[j].size << ",";
      group_nr_of_points(j, 0) =  possible_groups[j].size;
      //      out << possible_groups[j].grown_form << ")(" << possible_groups[j].descriptor_.rows() <<" "<<possible_groups[j].descriptor_.transpose() << ")";
      grown_from(j, 0) =  possible_groups[j].grown_form;
      for(int k = 0; k < possible_groups[j].descriptor_.rows(); ++k)
      {
        extractedFeatures(j, k) = possible_groups[j].descriptor_(k);
      }
      //      out << std::endl;
    }

    ds_test.setFeatureMatrix(size_of_parts, "size_of_parts");
    ds_test.setFeatureMatrix(lMatrix, "y");
    ds_test.setFeatureMatrix(aMatrix, "arr");
    ds_test.setFeatureMatrix(part_NrMatrix, "part_nr");
    ds_test.setFeatureMatrix(partIdsMatrix, "partIds");
    ds_test.setFeatureMatrix(group_nr_of_points, "group_size");
    ds_test.setFeatureMatrix(grown_from, "seed");
    ds_test.setFeatureMatrix(extractedFeatures, "x");
  }
  data_store.uploadData(ds_test, "test");
  std::cerr << "Uploading test data to server" << std::endl;
  client->assignData("test", icf::Classify);
  icf::ClassificationResult result = client->classify();
  icf::DataSet<double>::MatrixPtr confsPtr = result.confidences;
  std::stringstream res  ;
  res << *result.confidences;
  std::string res_str = res.str();
  std::cerr << "RETURNED BY CLASSIFIER:" << std::endl << res_str << std::endl;
  std::vector<std::string> lines;
  boost::split(lines, res_str, boost::is_any_of("\n"), boost::token_compress_on);
  std::cerr << "Number of returned part results: " << lines.size() << std::endl;
  int nrOfClasses = (*confsPtr).cols();// atoi(lines[0].c_str()) + 1;
  std::cerr << "Number of Object classes: " << nrOfClasses << std::endl;
  std::map<int, std::vector<float> > result_per_part_ID;
  for(unsigned int i = 0; i < lines.size(); ++i)
  {
    std::vector<std::string> values;
    if(lines[i] != "")
    {
      boost::split(values, lines[i], boost::is_any_of(" "), boost::token_compress_on);
    }
    else
    {
      continue;
    }
    for(unsigned int j = 0; j < values.size(); ++j)
    {
      if(values[j] != "")
      {
        float f = atof(values[j].c_str());
        result_per_part_ID[i].push_back(f);
      }
      else
      {
        continue;
      }
    }
  }
  std::cerr << "Original Cloud size: " << cloud->size() << std::endl;
  vector<int> index2cluster(cloud->size(), -1);
  for(unsigned int i = 0; i < opd->clusters.size(); ++i)
  {
    //std::cerr<<"Segment "<<i<<" : "<<clusters[i].indices.size()<<" points;"<<std::endl;
    for(unsigned int j = 0; j < opd->clusters[i]->indices.size(); ++j)
    {
      index2cluster.at(opd->clusters[i]->indices[j]) = i;
    }
  }
  std::cerr << "Creating result cloud:" << std::endl;
  result_cloud->points.reserve(cloud->points.size());

  for(unsigned int i = 0; i < cloud->points.size(); ++i)
  {
    if((index2cluster[i] != -1))
    {
      pcl::PointXYZLRegion p;
      p.x = cloud->points[i].x;
      p.y = cloud->points[i].y;
      p.z = cloud->points[i].z;
      std::vector<float>::iterator max = std::max_element(result_per_part_ID[index2cluster[i]].begin(), result_per_part_ID[index2cluster[i]].end());
      int maxi = max - result_per_part_ID[index2cluster[i]].begin();

      //disregarding other class
      if(maxi == 5 && other == 0)
      {
        *max = 0;
        max = std::max_element(result_per_part_ID[index2cluster[i]].begin(),
                               result_per_part_ID[index2cluster[i]].end());
        maxi = max - result_per_part_ID[index2cluster[i]].begin();
      }
      p.label = maxi + 1;

      p.reg = index2cluster[i];
      result_cloud->points.push_back(p);
    }
  }
  result_cloud->height = 1;
  result_cloud->width = result_cloud->points.size();
  std::cerr << "RESULT CLOUD NR OF POINT:" << result_cloud->points.size() << std::endl;
  first = false;
  delete opd;
}

void parse_args(std::string param_string)
{
  /* *
   *    -m specify minimum number of points in a part| default value: 100
        -r specify radius of radius_search in region growing | default value: 0.02
        -c specify max cuvature of a point from where region growing can start | default value: 0.01
        -s specify usage of srand 0 use srand 1 don't
        -f feature type, specify feature(0-GRSD(default) 1-C3_HLAC 2-VOSCH )
        -o toggle existance of other class (default 1-exist 0-disregard)
        -n define max number of segments that can get grouped together (for scenes this should be somewhere between 3-5)
        -v set vicinity threshold for finding groupings (in meters) default value is 0.015
   */
  std::vector<std::string> values;
  boost::split(values, param_string, boost::is_any_of(" "), boost::token_compress_on);
  std::cerr << "Splitting " << param_string << std::endl;

  for(std::vector<std::string>::const_iterator token = values.begin(); token < values.end(); token++)
  {

    const std::string paramName = *token;
    if(token++ == values.end() || !boost::starts_with(paramName, "-"))
    {
      throw std::string("Invalid param format. Please see ... for valid parameters.");
    }

    const std::string &paramValue = *token;

    switch(paramName[1])
    {
    case 'r':
      tolerance = atof(paramValue.c_str());
      break;
    case 'm':
      min_points_in_part = atoi(paramValue.c_str());
      break;
    case 'c':
      max_curvature = atof(paramValue.c_str());
      break;
    case 's':
      use_srand = atoi(paramValue.c_str());
      break;
    case 'f':
      feature_type = atoi(paramValue.c_str());
      break;
    case 'o':
      other = atoi(paramValue.c_str());
      break;
    case 'n':
      max_nr_of_grouped_segments = atoi(paramValue.c_str());
      break;
    case 'v':
      vicinity_threshold = atof(paramValue.c_str());
      break;
    default:
      break;
    }
  }

}

bool call_back(object_hasher::ClassifyScene::Request &req,
               object_hasher::ClassifyScene::Response &res)
{
  ROS_INFO("RECEIVING POINT CLOUD");
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
  pcl::PointCloud<pcl::PointXYZLRegion>::Ptr res_cloud(new pcl::PointCloud<pcl::PointXYZLRegion>());

  std::cerr << req.params << std::endl;
  pcl::fromROSMsg(req.in_cloud, *cloud);
  parse_args(req.params);

  std::cerr << "Processing..." << std::endl;
  process(cloud, res_cloud);

  pcl::io::savePCDFile("/tmp/cloud_sent.pcd", *res_cloud);
  sensor_msgs::PointCloud2 result_cloud_blob;
  pcl::toROSMsg(*res_cloud, result_cloud_blob);
  res.out_cloud = result_cloud_blob;
  //g_stop=true;
  return true;
}

int main(int argc, char **argv)
{
  std::string node_name = "ias_classifier_manager";
  //  std::string node_name = "scene_classifier";
  ros::init(argc, argv, node_name);

  ros::NodeHandle n_manager("~");
  std::cerr << "Node started:" << node_name << std::endl;
  icf::ClassifierManager g_manager(n_manager);
  startManagerThread();
  sleep(1);

  ros::NodeHandle n("/classifier");
  ros::ServiceServer service = n.advertiseService("classify_scene", call_back);
  ROS_INFO("Ready To Classify Scenes");
  ros::spin();
  return (0);
}

