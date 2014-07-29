#include <object_hasher/segmentation/euclidean_clustering_curvature.h>

namespace oph{

ptrdiff_t myrandom(ptrdiff_t i)
{
  return rand() % i;
}

//empty constructor
EuclideanClusterExtractorCurvature::EuclideanClusterExtractorCurvature():normalsSet_(false),cloudSet_(false),
  maxPts_(std::numeric_limits<int>::max()),treeSet_(false),eps_angle_(45*M_PI/180),distTolerance_(0.020),
  useSrand_(false),maxCurvature_(0.01)
{

}

EuclideanClusterExtractorCurvature::~EuclideanClusterExtractorCurvature()
{
  //empty destructor
}

void EuclideanClusterExtractorCurvature::addCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud){
  cloud_=cloud;
  cloudSet_=true;
}

void EuclideanClusterExtractorCurvature::segment(std::vector<pcl::PointIndices::Ptr> &segments)
{
  //ptrdiff_t (*p_myrandom)(ptrdiff_t) = myrandom;
  //uncommnet srand if you want indices to be truely
  if(useSrand_)
  {
    srand(time(NULL));
  }
  if(!treeSet_)
  {
    std::cerr<<"No KdTree was set...creating"<<std::endl;
    createTree();
  }

  // \note If the tree was created over <cloud, indices>, we guarantee a 1-1 mapping between what the tree returns
  //and indices[i]
  if(tree_->getInputCloud()->points.size() != cloud_->size())
  {
    std::cerr<<"[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset ("
            << tree_->getInputCloud()->points.size()
            <<") than the input cloud ("<<cloud_->size()<<")!"<<std::endl;
    return;
  }
  if(!normalsSet_)
  {
    std::cerr<<"No Normals were set...calculating"<<std::endl;
    calculateNormals();
  }

  if(cloud_->size() != normals_->points.size())
  {
    std::cerr<<"[pcl::extractEuclideanClusters] Number of points in the input point cloud ("
            <<cloud_->size()<<") different than normals ("<<normals_->size()<<")!"<<std::endl;
    return;
  }

  // Create a bool vector of processed point indices, and initialize it to false
  std::vector<bool> processed(cloud_->size(), false);

  //create a random copy of indices
  std::vector<int> indices_rnd(cloud_->size());
  for(unsigned int i = 0; i < indices_rnd.size(); ++i)
  {
    indices_rnd[i] = i;
  }
  //uncommnet myrandom part if you want indices to be truely
  if(useSrand_)
  {
    std::random_shuffle(indices_rnd.begin(), indices_rnd.end(), myrandom);
    std::cerr<<"REAL RANDOM"<<std::endl;
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

    if(processed[i] || normals_->points[indices_rnd[i]].curvature > maxCurvature_)
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
      if(!tree_->radiusSearch(seed_queue->indices[sq_idx], distTolerance_, nn_indices, nn_distances))
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
            normals_->points[indices_rnd[i]].normal[0] * normals_->points[nn_indices[j]].normal[0] +
            normals_->points[indices_rnd[i]].normal[1] * normals_->points[nn_indices[j]].normal[1] +
            normals_->points[indices_rnd[i]].normal[2] * normals_->points[nn_indices[j]].normal[2];
        if(fabs(acos(dot_p)) < eps_angle_)
        {
          processed[index_lookup[nn_indices[j]]] = true;
          seed_queue->indices.push_back(nn_indices[j]);
        }
      }
      sq_idx++;
    }

    // If this queue is satisfactory, add to the clusters
    if(seed_queue->indices.size() >= minPts_ && seed_queue->indices.size() <= maxPts_)
    {
      seed_queue->header = cloud_->header;
      segments.push_back(seed_queue);
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
  std::cerr<<"Number of unprocessed indices: "<<unprocessed_counter<<std::endl;
}

void EuclideanClusterExtractorCurvature::calculateNormals()
{
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal> ());

  pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
  ne.setInputCloud(cloud_);
  ne.setSearchMethod(tree_);
  ne.setRadiusSearch(0.02);
  ne.compute(*normals);
  normals_ = normals;
  normalsSet_ = true;
}
void EuclideanClusterExtractorCurvature::createTree()
{
  pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr t(new pcl::search::KdTree<pcl::PointXYZRGBA> (cloud_));
  t->setInputCloud(cloud_);
  tree_ = t;
  treeSet_=true;
}

}
