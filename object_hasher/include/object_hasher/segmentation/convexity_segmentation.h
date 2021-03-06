
#ifndef ACTIVE_SEGMENTATION_H_
#define ACTIVE_SEGMENTATION_H_

#include <pcl/search/pcl_search.h>
#include <pcl/pcl_base.h>

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <pcl/features/boundary.h>

#include <pcl/features/normal_3d.h>

#include <Eigen/Core>
//using namespace pcl;
namespace oph
{
/////////////////////////////////////////////////////////////////////
/**
  * \brief stand alone method for doing active segmentation
  * \param[in] cloud_in input cloud
  * \param[in] boundary the boundary map
  * \param[in] normals normals for the input cloud
  * \param[in] tree the spatial locator used for radius search
  * \param[in] index of the fixation point
  * \param[in] search_radius radius of search
  * \param[out] indices_out the resulting segment as indices of the input cloud
  */
template <typename PointT> void
activeSegmentation(const pcl::PointCloud<PointT>                         &cloud_in,
                   const pcl::PointCloud<pcl::Boundary>                  &boundary,
                   const pcl::PointCloud<pcl::Normal>                    &normals,
                   const boost::shared_ptr<pcl::search::Search<PointT> > &tree,
                   int                                                   fp_index,
                   float                                                 search_radius,
                   pcl::PointIndices                                     &indices_out);

/**
  * \brief
  * \author Ferenc Balint-Benczedi
  * \ingroup segmentation
  * \description Active segmentation or segmentation around a fixation point as the authors call it,
  *  extracts a region enclosed by a boundary based on a point inside this.
  *
  * \note If you use this code in any academic work, please cite:
  *
  *      - Ajay Mishra, Yiannis Aloimonos, Cornelia Fermuller
  *      Active Segmentation for Robotics
  *      In 2009 IEEERSJ International Conference on Intelligent Robots and Systems (2009)
  *
  */
template<typename PointT, typename NormalT>
class ActiveSegmentation : public pcl::PCLBase<PointT>
{
  typedef pcl::search::Search<PointT> KdTree;
  typedef typename KdTree::Ptr KdTreePtr;
  typedef pcl::PointCloud<pcl::Boundary> Boundary;
  typedef typename Boundary::Ptr BoundaryPtr;

  typedef pcl::PointCloud<NormalT> Normal;
  typedef typename Normal::Ptr NormalPtr;

  using pcl::PCLBase<PointT>::input_;
  using pcl::PCLBase<PointT>::indices_;

  typedef pcl::PointCloud<PointT> PointCloud;

public:
  /* \brief empty constructor */
  ActiveSegmentation() :
    tree_(), normals_(), boundary_(), fixation_point_(), fp_index_(), search_radius_(0.02)
  {
  }

  /* \brief empty destructor */
  virtual ~ActiveSegmentation()
  {
  }

  /** \brief Set the fixation point.
    * \param[in] p the fixation point
    */
  void
  setFixationPoint(const PointT &p);

  /** \brief Set the fixation point.
    * \param[in] x the X coordinate of the fixation point
    * \param[in] y the Y coordinate of the fixation point
    * \param[in] z the Z coordinate of the fixation point
    */
  inline void
  setFixationPoint(float x, float y, float z)
  {
    PointT p;
    p.x = x;
    p.y = y;
    p.z = z;
    setFixationPoint(p);
  }

  /* \brief Returns the fixation point as a Point struct. */
  PointT
  getFixationPoint()
  {
    return (fixation_point_);
  }

  /** \brief Set the fixation point as an index in the input cloud.
    * \param[in] index the index of the point in the input cloud to use
    */
  inline void
  setFixationPoint(int index)
  {
    fixation_point_ = input_->points[index];
    fp_index_ = index;
  }

  /* \brief Returns the fixation point index. */
  int
  getFixationPointIndex()
  {
    return (fp_index_);
  }

  /** \brief Provide a pointer to the search object.
    * \param[in] tree a pointer to the spatial search object.
    */
  inline void
  setSearchMethod(const KdTreePtr &tree)
  {
    tree_ = tree;
  }

  /** \brief returns a pointer to the search method used. */
  inline KdTreePtr
  getSearchMethod() const
  {
    return (tree_);
  }

  /** \brief Set the boundary map of the input cloud
    * \param[in] boundary a pointer to the boundary cloud
    */
  inline void
  setBoundaryMap(const BoundaryPtr &boundary)
  {
    boundary_ = boundary;
  }

  /* \brief returns a pointer to the boundary map currently set. */
  inline BoundaryPtr
  getBoundaryMap() const
  {
    return (boundary_);
  }

  /** \brief Set search radius for the region growing
    * \param[in] r the radius used
    */
  inline void
  setSearchRadius(double r)
  {
    search_radius_ = r;
  }

  /** \brief Set the input normals to be used for the segmentation
    * \param[in] norm the normals to be used
    */
  inline void
  setInputNormals(const NormalPtr &norm)
  {
    normals_ = norm;
  }

  /** \brief returns a pointer to the normals */
  inline NormalPtr
  getInputNormals()
  {
    return normals_;
  }

  /**
    * \brief Method for segmenting the object that contains the fixation point
    * \param[out] indices_out
    */
  void
  segment(pcl::PointIndices &indices_out);

private:

  /** \brief A pointer to the spatial search object. */
  KdTreePtr tree_;

  /** \brief A pointer to the normals of the object. */
  NormalPtr normals_;

  /**\brief A pointer to the boundary map associated with the cloud*/
  BoundaryPtr boundary_;

  /**\brief fixation point as a pcl:struct*/
  PointT fixation_point_;

  /** \brief fixation point as an index*/
  int fp_index_;

  /**radius of search for region growing*/
  double search_radius_;

  /** \brief Checks if a point should be added to the segment
    * \return true if point can be added to segment
    * \param[in] index of point to be verified
    * \param[in] seed point index
    * \param[out] output var true if point can be a seed
    * \param[out] output var true if point belongs to a boundary
    */
  bool
  isPointValid(int v_point, int seed, bool &is_seed, bool &is_boundary);
};

}
#endif /* ACTIVE_SEGMENTATION_H_ */
