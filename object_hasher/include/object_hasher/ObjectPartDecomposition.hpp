/*
 * ObjectPartDecomposition.h
 *
 *  Created on: Mar 16, 2011
 *      Author: ferka
 */

#include <pcl/point_types.h>
#include "pcl/io/pcd_io.h"
#include "pcl/kdtree/kdtree.h"
#include "pcl/kdtree/kdtree_flann.h"
#include "pcl/features/normal_3d.h"

#include "pcl/features/rsd.h"
#include "pcl/features/feature.h"

#include <pcl/surface/mls.h>

#include "pcl/filters/statistical_outlier_removal.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/filters/extract_indices.h"


#include <pcl/segmentation/extract_clusters.h>
#include "pcl/segmentation/sac_segmentation.h"

#include "pcl/ModelCoefficients.h"
#include "pcl/sample_consensus/method_types.h"
#include "pcl/sample_consensus/model_types.h"
#include "pcl/features/feature.h"
#include <pcl/surface/mls.h>

#include <boost/algorithm/string.hpp>

#include <Eigen/Core>
#include <fstream>

#include <algorithm>
#include <stdlib.h>

#include "object_hasher/ObjectGroup.h"
#include "object_hasher/point_type.h"

//#include "c3_hlac/c3_hlac_tools.h"


#ifndef OBJECTPARTDECOMPOSITION_H_
#define OBJECTPARTDECOMPOSITION_H_

#define NR_CLASS 5
#define NOISE 0
#define PLANE 1
#define CYLINDER 2
#define SPHERE 3
#define EDGE 4
#define EMPTY 5

#define GRSDm 0
#define C3_HLAC 1
#define VOSCH 2

ptrdiff_t myrandom(ptrdiff_t i)
{
  return rand() % i;
}

template<typename PointT>
class ObjectPartDecomposition
{
public:
  typedef typename  pcl::PointCloud<PointT>::Ptr PointTPtr;
  typedef typename  pcl::search::KdTree<PointT>::Ptr KdTreePtr;

  //contructor for loading from a file
  ObjectPartDecomposition(std::string p, int type, std::string filen):
    cloud(new pcl::PointCloud<PointT>()),
    max_nr_of_parts(8), save_segmented_pcd(false), vicinity_threshold(0.015)
  {
    path = p;
    use_srand = type;
    filename = filen;
    std::string complete_path = path + "data/times.data";
    if(pcl::io::loadPCDFile<PointT> (filename, *cloud) == -1)  //* load the file
    {
      PCL_ERROR("Couldn't read file test_pcd.pcd \n");
    }
  }
  //
  ObjectPartDecomposition(std::string p, int type):
    cloud(new pcl::PointCloud<PointT>()),
    max_nr_of_parts(8), save_segmented_pcd(false), vicinity_threshold(0.015)
  {
    path = p;
    use_srand = type;
    std::string complete_path = path + "data/times.data";
  }

  PointTPtr cloud;
  Eigen::MatrixXi connected;
  std::vector<pcl::PointIndices::Ptr> clusters;
  std::string path, filename;
  int use_srand;//0 train 1 test
  int max_nr_of_parts;
  bool save_segmented_pcd;
  double vicinity_threshold;

  void setInputCloud(PointTPtr incloud)
  {
    pcl::copyPointCloud(*incloud, *cloud);
  }

  void setfilename(std::string name)
  {
    filename = name;
  }

  void saveSegmentedCloud(bool b)
  {
    save_segmented_pcd = b;
  }

  void setNrOfGroupedParts(int nr)
  {
    max_nr_of_parts = nr;
  }

  void setVicinityThreshold(double vt)
  {
    vicinity_threshold = vt;
  }

  void groupComponents(int level, int nr, Eigen::MatrixXi connected, int min_nr, int max_nr,
                       ObjectGroup added, Eigen::VectorXi next, Eigen::VectorXi considered,
                       std::vector<ObjectGroup> &groups, int max_start,
                       const std::vector<Eigen::VectorXf> &descriptor_list, int source)
  {
    // If we reached minimum number of components, add it as a result
    if(level >= min_nr)
    {
      added.grown_form = source;
      source = groups.size();
      groups.push_back(added);
    }
    // If we did not reach the maximum number of components
    if(level < max_nr)
    {
      /// @TODO: think about optimization.. should possible be a set or a lut? size as a parameter? if level=max_nr-1 add result directly?
      for(int i = max_start + 1; i < nr; i++)
      {
        // Loop through possible neighbors
        if((next[i]) && (!considered[i]))
        {
          // avoid loopbacks
          considered[i] = 1;
          // add current neighbor
          ObjectGroup *new_added = new ObjectGroup();
          new_added->group_ = added.group_;
          new_added->group_[i] = 1;
          new_added->descriptor_  = descriptor_list[i] + added.descriptor_;
          new_added->ID_ = added.ID_;
          new_added->size = added.size + clusters[i]->indices.size();
          // update status of possible neighors
          Eigen::VectorXi new_next = next;
          new_next[i] = 0;
          if(level == 0)
          {
            new_next = connected.row(i);
            max_start = i;
          }
          else
          {
            // check for already added things is implicitly handled by considered
            for(int j = 0; j < nr; j++)
            {
              new_next[j] = new_next[j] || connected(i, j);
            }
            //new_next[j] = new_next[j] || ((!added[j]) && connected[i][j]);
          }
          // recursive call to add more if needed/possible
          groupComponents(level + 1, nr, connected, min_nr, max_nr, *new_added, new_next, considered, groups, max_start, descriptor_list, source);
          delete new_added;
        }
      }
    }
  }

  std::vector<ObjectGroup> getFeatures(int ID, float tolerance, float max_curvature, int min_points_in_part, int ft, bool calc_normals)
  {
    // Load PCD
    vector<ObjectGroup> possible_groups;
    std::string complete_path = path + "data/times.data";
    std::ofstream  outfile(complete_path.c_str(), ios_base::app);

    //fileter the cloud
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(10);
    sor.setStddevMulThresh(2.0);
    sor.filter(*cloud);

    //construct Kd tree ---needed for normal estimation
    KdTreePtr tree(new pcl::search::KdTree<PointT> (cloud));
    tree->setInputCloud(cloud);

    const double normal_radius_search = 0.02;
    // Normal estimation
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal> ());
    if(calc_normals)
    {
      pcl::NormalEstimation<PointT, pcl::Normal> ne;
      ne.setInputCloud(cloud);
      ne.setSearchMethod(tree);
      ne.setRadiusSearch(normal_radius_search);
      ne.compute(*normals);
    }

    else
    {
      pcl::copyPointCloud(*cloud, *normals);
    }


    //double t2 = my_clock();
    //double duration = t2-t1;
    //outfile<<"Preprocessing: "<<duration<<std::endl;
    //region growing based on normals
    // t1 = my_clock();
    extractEuclideanClustersCurvature(*cloud, *normals,  tree, tolerance, clusters, 45 * M_PI / 180, max_curvature , min_points_in_part);
    std::cerr << "Number of segments: " << clusters.size() << std::endl;
    //initializing connection matrix and index to cluster mapping
    if(clusters.size() == 0)
    {
      return possible_groups;
    }
    vector<int> index2cluster(cloud->size(), -1);
    for(unsigned int i = 0; i < clusters.size(); ++i)
    {
      std::cerr << "Segment " << i << " : " << clusters[i]->indices.size() << " points;" << std::endl;
      for(unsigned int j = 0; j < clusters[i]->indices.size(); ++j)
      {
        index2cluster.at(clusters[i]->indices[j]) = i;
      }
    }
    //creating list of unclustered points
    std::vector<int> unclustered_indices;
    for(unsigned int i = 0; i != cloud->points.size(); ++i)
      if(index2cluster[i] == -1)
      {
        unclustered_indices.push_back(i);
      }

    //clustering remaining points as well, only based on distance
    /*  int old_cluster_size = clusters.size();
    KdTree<PointXYZ>::Ptr tree2 = boost::make_shared<KdTreeFLANN<PointXYZ> > (false);;
    tree2-> setInputCloud (cloud.makeShared (), boost::make_shared <vector<int> > (unclustered_indices));
    extractEuclideanClusters(cloud, unclustered_indices,tree2, 0.020, clusters, 10);

    //initializing connection matrix and index to cluster mapping
    for (unsigned int i = old_cluster_size;i<clusters.size();++i)
    {
      for(unsigned int j = 0; j <clusters[i]->indices.size();++j)
        index2cluster.at(clusters[i]->indices[j]) = i;
    }*/

    //TODO make this optional
    if(save_segmented_pcd)
    {

      pcl::PointCloud<pcl::PointXYZRGBNormalRegion> segmented_cloud_with_normals;
      segmented_cloud_with_normals.points.resize(cloud->points.size());
      float color_array[clusters.size() + 1];

      for(unsigned int i = 0; i < clusters.size(); ++i)
      {
        double  drgb[3];
        randRGB(drgb);
        float rgb = (int(255 * drgb[0]) << 16) | (int(255 * drgb[1]) << 8) | int(255 * drgb[2]);
        color_array[i] = rgb;
        //delete drgb;
      }

      float unclustered_color = 0x000000;//127<<16;//BLACK
      float grown_from_color = 0x00FF00;

      segmented_cloud_with_normals.width = cloud->points.size();
      segmented_cloud_with_normals.height = 1;

      for(unsigned int i = 0; i < cloud->points.size(); ++i)
      {
        segmented_cloud_with_normals.points[i].x = cloud->points[i].x;
        segmented_cloud_with_normals.points[i].y = cloud->points[i].y;
        segmented_cloud_with_normals.points[i].z = cloud->points[i].z;
        segmented_cloud_with_normals.points[i].normal_x = normals->points[i].normal_x;
        segmented_cloud_with_normals.points[i].normal_y = normals->points[i].normal_y;
        segmented_cloud_with_normals.points[i].normal_z = normals->points[i].normal_z;
        segmented_cloud_with_normals.points[i].reg = index2cluster[i];
        if(index2cluster[i] >= 0)
        {
          segmented_cloud_with_normals.points[i].rgb = *reinterpret_cast<float *>(&color_array[index2cluster[i]]);
        }
        else
        {
          segmented_cloud_with_normals.points[i].rgb = unclustered_color;
        }
        segmented_cloud_with_normals.points[i].curvature = normals->points[i].curvature;//index2cluster[i];
      }

      for(unsigned int i = 0; i < clusters.size(); ++i)
      {
        segmented_cloud_with_normals.points[clusters[i]->indices[0]].rgb = (grown_from_color);
      }

      std::vector<std::string> values;
      boost::split(values, filename, boost::is_any_of("."), boost::token_compress_on);
      std::stringstream s;

      //change this according to your file paths names etc.
      std::string st = filename.substr(0, filename.size() - 4);
      s << st << "_segmented.pcd";
      pcl::io::savePCDFile(s.str(), segmented_cloud_with_normals);
    }

    //calculate descriptor for each cluster
    std::vector<Eigen::VectorXf> descriptor_list;

    std::cerr << "GETING DESCRIPTOR FOR EACH CLUSTER!" << std::endl;
    for(unsigned int i = 0; i < clusters.size(); ++i)
    {
      Eigen::VectorXf descriptor;
      std::cerr << "[note] COMPUTING GRSD- FOR SEGMENT: " << i << std::endl;
      //      if(ft == GRSDm)
      computeGRSD2(cloud, normals , clusters[i], descriptor);
      /*      else if (ft == C3_HLAC)
        computeC3HLAC(cloud,clusters[i],descriptor);
      else if (ft == VOSCH)
      {
        Eigen::VectorXf descriptor_grsd, descriptor_c3hlac;
        computeGRSD2(cloud, normals ,clusters[i], descriptor_grsd);
        computeC3HLAC(cloud,clusters[i],descriptor_c3hlac);
        descriptor.resize(descriptor_grsd.rows()+descriptor_c3hlac.rows());
        for(unsigned int i = 0;i<descriptor_grsd.rows();++i)
          descriptor[i] = descriptor_grsd[i];
        for(unsigned int j=21; j<descriptor_c3hlac.rows()+21;++j)
          descriptor[j] = descriptor_c3hlac[j-21];
      }*/
      std::cerr << "RESULTING GRSD: " << descriptor.transpose() << std::endl;
      descriptor_list.push_back(descriptor);
    }


    //building up connection matrix between clusters

    std::cerr << "BUILDING CONNECTION MATRIX" << std::endl;
    connected = Eigen::MatrixXi::Zero(clusters.size(), clusters.size());
    for(size_t i = 0; i < cloud->points.size(); ++i)
    {
      int source = index2cluster[i];
      if(source == -1)
      {
        continue;
      }
      std::vector<int> nnIndices;
      std::vector<float> nnSqrDistances;
      tree->radiusSearch(i, 0.015, nnIndices, nnSqrDistances);
      std::vector<int>::const_iterator it = nnIndices.begin();
      for(; it != nnIndices.end(); ++it)
      {
        int target = index2cluster[*it];
        if(target == -1)
        {
          continue;
        }
        connected(source, target) = true;
        //TODO this soudln't be neccesary:P
        connected(target, source) = true;
      }
    }
    //uncomment to visualize connection matrix
    //std::cerr<<"Connection Matrix:"<<std::endl;
    //std::cerr<<connected<<endl;

    // Groups - starting from one element at a time
    //std::cerr<<"Laplacian Matrix: "<<std::endl;
    //std::cerr<<laplMatrix<<std::endl;

    int min_nr = 1;
    //int max_nr=8;

    for(unsigned i = 0; i < clusters.size(); i++)
    {
      //Eigen::VectorXi added = Eigen::VectorXi::Zero(clusters.size ());
      ObjectGroup *added = new ObjectGroup();
      added->group_ = Eigen::VectorXi::Zero(clusters.size());
      added->group_(i) = 1;
      added->descriptor_ = descriptor_list[i];
      added->ID_ = ID;
      added->size = clusters[i]->indices.size();
      Eigen::VectorXi next = connected.row(i);
      Eigen::VectorXi considered  = added->group_;
      groupComponents(1, clusters.size(), connected, min_nr, max_nr_of_parts, *added, next, considered, possible_groups, i, descriptor_list, -1);
      delete added;
    }
    //t2 = my_clock();
    // duration = t2-t1;
    //outfile<<"All other: "<<duration<<std::endl;
    return possible_groups;
  }

  void computeArrangementKey(std::vector<ObjectGroup> &possible_groups)
  {
    //int tmp_size1 = possible_groups.size ();
    connected = connected - Eigen::MatrixXi::Identity(clusters.size(), clusters.size());
    for(unsigned g = 0; g < possible_groups.size(); g++)
    {
      // create connectivity matrix of the sub-component
      int nr_component = possible_groups[g].group_.sum();
      Eigen::MatrixXi sub_connected_row(nr_component, clusters.size());
      int row = 0;
      for(int j = 0; j < possible_groups[g].group_.rows(); j++)
        if(possible_groups[g].group_[j])
        {
          sub_connected_row.row(row++) = connected.row(j);
        }
      Eigen::MatrixXi sub_connected(nr_component, nr_component);
      int col = 0;
      for(int j = 0; j < possible_groups[g].group_.rows(); j++)
        if(possible_groups[g].group_[j])
        {
          sub_connected.col(col++) = sub_connected_row.col(j);
        }

      std::ofstream eigens_file("data/eigenvs.data", ios_base::app);

      //std::cerr<<"The eigen values of the laplacian matrix are : "<<std::endl<<es.eigenvalues()<<std::endl;
      Eigen::MatrixXi laplMatrix = calcLaplacianMatrix(sub_connected);
      Eigen::EigenSolver<Eigen::MatrixXd> es_laplacian(laplMatrix.cast<double>());
      Eigen::EigenSolver<Eigen::MatrixXd> es_connection(sub_connected.cast<double>());
      //std::vector<double> eigen_laplace(laplMatrix.rows());
      std::vector<int> eigen_laplace_int(laplMatrix.rows());

      eigens_file << "Connection Matrix: " << std::endl;
      eigens_file << sub_connected << std::endl;
      eigens_file << "Eigen Values of Connection matrix: " << std::endl;
      eigens_file << es_connection.eigenvalues() << std::endl;
      eigens_file << "Laplacian Matrix:" << std::endl;
      eigens_file << laplMatrix << std::endl;
      eigens_file << "Eigen Values of Laplacian M" << std::endl;
      eigens_file << es_laplacian.eigenvalues() << endl;


      for(int i = 0; i < laplMatrix.rows(); ++i)
      {
        complex<double> lambda = es_laplacian.eigenvalues()[i];
        eigen_laplace_int[i] = round(lambda.real());
      }
      std::sort(eigen_laplace_int.begin(), eigen_laplace_int.end());

      eigens_file << "Real parts of Eigen values" << std::endl;
      for(unsigned int k = 0; k < eigen_laplace_int.size(); ++k)
      {
        eigens_file << "-" << eigen_laplace_int[k] << "-";
      }
      eigens_file << std::endl;
      if(eigen_laplace_int[0] != 0 || (eigen_laplace_int.size() > 1 && eigen_laplace_int[1] == 0))
      {
        eigens_file << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
      }
      eigens_file << "--------------" << std::endl;
      //eigens_file.close();
      // Print out and compute arangement key

      std::vector<int> grade_list(nr_component);
      int i = 0;
      for(int j = 0; j < possible_groups[g].group_.rows(); j++)
      {
        if(possible_groups[g].group_[j])
        {
          grade_list[i] = sub_connected.row(i).sum();
          i++;
        }
      }
      std::sort(grade_list.begin(), grade_list.end());
      possible_groups[g].CalculateArrangementKey(grade_list);
      eigens_file.close();
    }
  }

  double getRGB(float r, float g, float b)
  {
    int res = (int (r * 255) << 16) | (int (g * 255) << 8) | int (b * 255);
    double rgb = *(float *)(&res);
    return (rgb);
  }

  void printOut(const std::vector<ObjectGroup> &possible_groups)
  {
    int tmp_size1 = possible_groups.size();
    cerr << "Unique component groups: " << (int)possible_groups.size() << "/" << tmp_size1 << endl;
    for(unsigned g = 0; g < possible_groups.size(); g++)
    {

      cerr << "Group " << g << "/" << possible_groups.size() << " contains:";
      for(int j = 0; j < possible_groups[g].group_.rows(); j++)
      {
        if(possible_groups[g].group_[j])
        {
          cout << " " << j;
        }
      }
      //std::sort (grade_list.begin(), grade_list.end());
      cout << " Key: " << possible_groups[g].arrangement_key_ << " Descriptor: " << possible_groups[g].descriptor_.transpose()
           << " Grown form index nr " << possible_groups[g].grown_form;
      cout << endl;
    }
  }


  //  void computeC3HLAC(PointTPtr input_cloud, const boost::shared_ptr<const pcl::PointIndices > indices, Eigen::VectorXf &descriptor)
  //  {
  //   pcl::PointCloud<pcl::PointXYZRGB>::Ptr copy_cloud(new pcl::PointCloud<pcl::PointXYZRGB>) ;
  //    pcl::copyPointCloud(*input_cloud, *copy_cloud);
  //
  //    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr segment(new pcl::PointCloud<pcl::PointXYZRGB>);
  //    extract.setInputCloud(copy_cloud);
  //    extract.setIndices(indices);
  //    extract.setNegative(false);
  //    extract.filter(*segment);
  //
  //    const double voxel_size = 0.02;
  //    pcl::VoxelGrid<pcl::PointXYZRGB> grid;
  //    pcl::PointCloud<pcl::PointXYZRGB> cloud_downsampled;
  //    getVoxelGrid(grid, *segment, cloud_downsampled, voxel_size);
  //
  //    std::vector<float> c3_hlac;
  //    extractC3HLACSignature117(grid, cloud_downsampled, c3_hlac, 127, 127, 127, voxel_size);
  //    descriptor.resize(c3_hlac.size());
  //    for (unsigned int i=0; i<c3_hlac.size();++i)
  //    {
  //      descriptor[i] = c3_hlac[i];
  //    }
  //  }

  void computeGRSD2(/*pcl::PointCloud<PointT>::Ptr*/PointTPtr input_cloud, pcl::PointCloud<pcl::Normal>::Ptr cloud_normals,
      const boost::shared_ptr<pcl::PointIndices > indices, Eigen::VectorXf &descriptor)
  {
    double min_radius_plane = 0.100;
    double max_radius_noise = 0.015;
    double min_radius_cylinder = 0.175;
    double max_min_radius_diff = 0.050;

    //double min_radius_edge_ = 0.030;

    //static int i=1;
    //std::cerr<<"Nr:"<<i++<<endl;
    double t1;
    double downsample_leaf = 0.02;
    double rsd_radius_search = 0.03;
    std::cerr << "ObjectHasher::computeGRSD::start()" << std::endl;
    //pcl::PointCloud<PointT>::Ptr segment(new pcl::PointCloud<PointT> ());
    PointTPtr segment(new pcl::PointCloud<PointT> ());
    pcl::ExtractIndices<PointT> ei;
    ei.setInputCloud(input_cloud);
    ei.setIndices(indices);
    ei.setNegative(false);
    ei.filter(*segment);
    std::cerr << "ObjectHasher::computeGRSD::cluster size = " << segment->points.size() << std::endl;

    //std::cerr<<"SEGMENT SIZE:"<<segment->size()<<std::endl;
    // Create the voxel grid
    // pcl::PointCloud<PointT>::Ptr cloud_downsampled (new pcl::PointCloud<PointT> ());
    std::cerr << "ObjectHasher::computeGRSD:: downsamling" << std::endl;
    PointTPtr cloud_downsampled(new pcl::PointCloud<PointT> ());
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize(downsample_leaf, downsample_leaf, downsample_leaf);
    grid.setInputCloud(segment/*input_cloud*/);
    grid.setSaveLeafLayout(true);  // TODO avoid this using nearest neighbor search
    grid.filter(*cloud_downsampled);
    std::cerr << "ObjectHasher::computeGRSD::downsampled cloud size = " << cloud_downsampled->size() << std::endl;

    //    std::stringstream s1;
    //    s1<<"data/segmented/"<<indices->indices.size()<<"_segment.pcd";
    //    //    pcl::io::savePCDFile (s1.str(), *segment);
    //
    //    std::stringstream s2;
    //    s2<<"data/segmented/"<<indices->indices.size()<<"_downsampled.pcd";
    //    //  pcl::io::savePCDFile (s2.str(), *cloud_downsampled);

    // Compute RSD
    pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr radii(new pcl::PointCloud<pcl::PrincipalRadiiRSD> ());
    pcl::RSDEstimation <PointT, pcl::Normal, pcl::PrincipalRadiiRSD> rsd;
    rsd.setInputCloud(cloud_downsampled);
    rsd.setSearchSurface(input_cloud);
    rsd.setInputNormals(cloud_normals);
    rsd.setRadiusSearch(std::max(rsd_radius_search, sqrt(3) * downsample_leaf / 2));
    KdTreePtr tree = boost::make_shared<pcl::search::KdTree<PointT> > ();
    tree->setInputCloud(input_cloud);
    rsd.setSearchMethod(tree);

    t1 = my_clock();
    rsd.compute(*radii);
    ROS_INFO("RSD compute done in %f seconds.", my_clock() - t1);

    //pcl::PointCloud<pcl::PointNormalRADII> cloud_downsampled_radii;
    //pcl::concatenateFields (cloud_downsampled, radii, cloud_downsampled_radii);
    // t1 = my_clock();
    // Get rmin/rmax for adjacent 27 voxel

    Eigen::MatrixXi relative_coordinates(3, 13);
    Eigen::MatrixXi transition_matrix =  Eigen::MatrixXi::Zero(NR_CLASS + 1, NR_CLASS + 1);
    int idx = 0;
    // 0 - 8
    for(int i = -1; i < 2; i++)
    {
      for(int j = -1; j < 2; j++)
      {
        relative_coordinates(0, idx) = i;
        relative_coordinates(1, idx) = j;
        relative_coordinates(2, idx) = -1;
        idx++;
      }
    }
    // 9 - 11
    for(int i = -1; i < 2; i++)
    {
      relative_coordinates(0, idx) = i;
      relative_coordinates(1, idx) = -1;
      relative_coordinates(2, idx) = 0;
      idx++;
    }
    // 12
    relative_coordinates(0, idx) = -1;
    relative_coordinates(1, idx) = 0;
    relative_coordinates(2, idx) = 0;

    Eigen::MatrixXi relative_coordinates_all(3, 26);
    relative_coordinates_all.block<3, 13>(0, 0) = relative_coordinates;
    relative_coordinates_all.block<3, 13>(0, 13) = -relative_coordinates;
    // SAVE THE TYPE OF EACH POINT
    std::vector<int> types(radii->points.size());

    for(size_t idx = 0; idx < radii->points.size(); ++idx)
      types[idx] = get_type(radii->points[idx].r_min,
                            radii->points[idx].r_max,
                            min_radius_plane,
                            max_radius_noise,
                            min_radius_cylinder,
                            max_min_radius_diff);

    for(size_t idx = 0; idx < cloud_downsampled->points.size(); ++idx)
    {
      int source_type = types[idx];
      std::vector<int> neighbors = grid.getNeighborCentroidIndices(cloud_downsampled->points[idx], relative_coordinates_all);
      for(unsigned id_n = 0; id_n < neighbors.size(); id_n++)
      {
        int neighbor_type;
        if(neighbors[id_n] == -1)
        {
          neighbor_type = EMPTY;
        }
        else
        {
          neighbor_type = types[neighbors[id_n]];
        }

        transition_matrix(source_type, neighbor_type)++;
      }
    }
    // pcl::PointCloud<pcl::GRSDSignature21> cloud_grsd;
    // cloud_grsd.points.resize(1);
    descriptor.resize(21);
    int nrf = 0;
    for(int i = 0; i < NR_CLASS + 1; i++)
      for(int j = i; j < NR_CLASS + 1; j++)
      {
        descriptor[nrf++] = transition_matrix(i, j) + transition_matrix(j, i);
      }

    //uncomment if you want to see the computed GRSD-s

    //std::cerr << "transition matrix" << std::endl << transition_matrix << std::endl;
    //std::cerr << std::endl<<descriptor.transpose()<<std::endl;
    ROS_INFO("GRSD compute done in %f seconds.", my_clock() - t1);
    //return cloud_grsd;
  }

  template <typename Normal>
  void extractEuclideanClustersCurvature(
    const pcl::PointCloud<PointT> &cloud,
    const pcl::PointCloud<Normal> &normals,
    const boost::shared_ptr<pcl::search::KdTree<PointT> > &tree,
    float tolerance, std::vector<pcl::PointIndices::Ptr> &clusters, double eps_angle, double max_curvature,
    unsigned int min_pts_per_cluster = 1,
    unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max)())
  {
    //ptrdiff_t (*p_myrandom)(ptrdiff_t) = myrandom;
    //uncommnet srand if you want indices to be truely
    if(use_srand == 1)
    {
      srand(time(NULL));
    }

    // \note If the tree was created over <cloud, indices>, we guarantee a 1-1 mapping between what the tree returns
    //and indices[i]
    if(tree->getInputCloud()->points.size() != cloud.points.size())
    {
      ROS_ERROR("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset (%zu) than the input cloud (%zu)!", tree->getInputCloud()->points.size(), cloud.points.size());
      return;
    }
    /* if (tree->getIndices ()->size () != cloud.size ())
    {
      ROS_ERROR ("[pcl::extractEuclideanClusters] Tree built for a different set of indices (%zu) than the input set (%zu)!", tree->getIndices ()->size (), cloud.size ());
      return;
    }*/
    if(cloud.points.size() != normals.points.size())
    {
      ROS_ERROR("[pcl::extractEuclideanClusters] Number of points in the input point cloud (%zu) different than normals (%zu)!", cloud.points.size(), normals.points.size());
      return;
    }
    // Create a bool vector of processed point indices, and initialize it to false
    std::vector<bool> processed(cloud.size(), false);

    //create a random copy of indices
    std::vector<int> indices_rnd(cloud.size());
    for(unsigned int i = 0; i < indices_rnd.size(); ++i)
    {
      indices_rnd[i] = i;
    }
    //uncommnet myrandom part if you want indices to be truely
    if(use_srand == 1)
    {
      std::random_shuffle(indices_rnd.begin(), indices_rnd.end(), myrandom);
      ROS_ERROR("REAL RANDOM");
    }
    else
    {
      std::random_shuffle(indices_rnd.begin(), indices_rnd.end());
    }
    std::cerr << "Processed size: " << processed.size() << std::endl;
    std::vector<int> index_lookup(indices_rnd.size());
    for(unsigned int i = 0; i < indices_rnd.size(); ++i)
    {
      index_lookup[indices_rnd[i]] = i;
    }

    std::vector<int> nn_indices;
    std::vector<float> nn_distances;
    // Process all points in the indices vector
    for(size_t i = 0; i < indices_rnd.size(); ++i)
    {

      if(processed[i] || normals.points[indices_rnd[i]].curvature > max_curvature)
      {
        /*if(normals.points[indices_rnd[i]].curvature > max_curvature)
          std::cerr<<"Curvature of point skipped: "<<normals.points[indices_rnd[i]].curvature<<std::endl;*/
        continue;
      }
      pcl::PointIndices::Ptr seed_queue(new pcl::PointIndices());
      int sq_idx = 0;
      seed_queue->indices.push_back(indices_rnd[i]);

      processed[i] = true;

      while(sq_idx < (int)seed_queue->indices.size())
      {
        // Search for sq_idx
        if(!tree->radiusSearch(seed_queue->indices[sq_idx], tolerance, nn_indices, nn_distances))
        {
          sq_idx++;
          continue;
        }

        for(size_t j = 1; j < nn_indices.size(); ++j)               // nn_indices[0] should be sq_idx
        {
          // std::cerr<<nn_indices[j]<<std::endl;
          if(processed[index_lookup[nn_indices[j]]])                              // Has this point been processed before ?
          {
            continue;
          }

          // [-1;1]
          double dot_p =
            normals.points[indices_rnd[i]].normal[0] * normals.points[nn_indices[j]].normal[0] +
            normals.points[indices_rnd[i]].normal[1] * normals.points[nn_indices[j]].normal[1] +
            normals.points[indices_rnd[i]].normal[2] * normals.points[nn_indices[j]].normal[2];
          if(fabs(acos(dot_p)) < eps_angle)
          {
            processed[index_lookup[nn_indices[j]]] = true;
            seed_queue->indices.push_back(nn_indices[j]);
          }
        }
        sq_idx++;
      }

      // If this queue is satisfactory, add to the clusters
      if(seed_queue->indices.size() >= min_pts_per_cluster && seed_queue->indices.size() <= max_pts_per_cluster)
      {
        seed_queue->header = cloud.header;
        clusters.push_back(seed_queue);
      }
    }
    int unprocessed_counter = 0;
    for(unsigned int i = 0; i < processed.size(); ++i)
    {
      if(processed[i] == false)
      {
        //std::cerr<<"Indice not processed at " <<i<<" : "<<indices_rnd[i]<<std::endl;
        unprocessed_counter++;
      }
    }
    //std::cerr<<"Number of unprocessed indices: "<<unprocessed_counter<<std::endl;

  }

  inline void randRGB(double *rgb, double min = 0.1, double max = 2.9)
  {
    // double* rgb = new double[3];
    double sum;
    unsigned stepRGBA = 150;
    do
    {
      sum = 0;
      rgb[0] = (rand() % stepRGBA) / (double)stepRGBA;
      while((rgb[1] = (rand() % stepRGBA) / (double)stepRGBA) == rgb[0]);
      while(((rgb[2] = (rand() % stepRGBA) / (double)stepRGBA) == rgb[0]) && (rgb[2] == rgb[1]));
      sum = rgb[0] + rgb[1] + rgb[2];
    }
    while(sum <= min || sum >= max);
  }

  Eigen::MatrixXi calcLaplacianMatrix(Eigen::MatrixXi graph)
  {
    Eigen::MatrixXi laplacian = Eigen::MatrixXi::Zero(graph.rows(), graph.cols());
    //std::cerr<<"Size of Laplacian Matrix: rows= "<<laplacian.rows()<<" cols= "<<laplacian.cols()<<std::endl;
    graph = graph + Eigen::MatrixXi::Identity(graph.rows(), graph.cols());
    for(int i = 0; i < laplacian.rows(); ++i)
    {
      for(int j = 0; j < laplacian.cols(); ++j)
      {
        if(j == i)
        {
          laplacian(i, j) = graph.row(i).sum() - 1;
        }
        else if(graph(i, j) == 1)
        {
          laplacian(i, j) = -1;
        }
      }
    }
    return laplacian;
  }

  int get_type(float min_radius, float max_radius, double min_radius_plane, double max_radius_noise, double min_radius_cylinder, double max_min_radius_diff)
  {
    min_radius *= 1.1;
    max_radius *= 0.9;
    if(min_radius > max_radius)
    {
      const double t = min_radius;
      min_radius = max_radius;
      max_radius = t;
    }
    //                    0.100
    if(min_radius > min_radius_plane)
    {
      return PLANE;  // plane
    }
    //                    0.175
    else if(max_radius > min_radius_cylinder)
    {
      return CYLINDER;  // cylinder (rim)
    }
    //                    0.015
    else if(min_radius < max_radius_noise)
    {
      return NOISE;  // noise/corner
    }
    //                                    0.05
    else if(max_radius - min_radius < max_min_radius_diff)
    {
      return SPHERE;  // sphere/corner
    }
    else
    {
      return EDGE;  // edge
    }
  }

  double my_clock()
  {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (double)tv.tv_usec * 1e-6;
  }
};
#endif /* OBJECTPARTDECOMPOSITION_H_ */
