
#ifndef EUCLIDEAN_CLUSTER_EXTRACTOR_CURVATURE_H
#define EUCLIDEAN_CLUSTER_EXTRACTOR_CURVATURE_H

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <Eigen/Core>
#include <object_hasher/segmentation/Segmentation.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <stdlib.h>
#include <random>
#include <algorithm>


namespace oph {

class EuclideanClusterExtractorCurvature: Segmentation

{
private:
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    bool normalsSet_,treeSet_,useSrand_,cloudSet_;

    //cluster size limits
    int maxPts_,minPts_;
    float distTolerance_,maxCurvature_,eps_angle_;

public:
    EuclideanClusterExtractorCurvature();
    ~EuclideanClusterExtractorCurvature();


    virtual void addCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud);

    inline void setNormals(const pcl::PointCloud<pcl::Normal>::Ptr &normals)
    {
        normals_=normals;
        normalsSet_=true;
    }
    inline void setKdTree(const pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree)
    {
        tree_=tree;
        treeSet_=true;
    }
    inline void setTrueRandom(const bool use_rand)
    {
        useSrand_=use_rand;
    }

    inline void setMinMaxPoints(int min, int max)
    {
        minPts_ = min;
        maxPts_ = max;
    }

    /**
     * @brief setEpsAngle set angle threshold between normals
     * @param eps angle in radians
     */
    inline void setEpsAngle (const float eps)
    {
        eps_angle_=eps;
    }
    /**
     * @brief setDistaneThreshold set the max distance between two point for region growing
     * @param dist distance in meteres
     */
    inline void setDistaneThreshold(const float dist)
    {
        distTolerance_=dist;
    }
    /**
     * @brief setMaxCurvature set the maximum curvature value for a seed point
     * @param c curvature value
     */
    inline void setMaxCurvature(const float c)
    {
        maxCurvature_ = c;
    }

    virtual void segment(std::vector<pcl::PointIndices::Ptr>&);
    void createTree();
    void calculateNormals();

};

}
#endif /* EUCLIDEAN_CLUSTER_EXTRACTOR_CURVATURE_H */

