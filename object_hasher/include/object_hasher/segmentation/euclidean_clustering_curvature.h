
/*
 * Software License Agreement (BSD License)
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2009-2011, Willow Garage, Inc.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the following
 *   disclaimer in the documentation and/or other materials provided
 *   with the distribution.
 * * Neither the name of Willow Garage, Inc. nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef EUCLIDEAN_CLUSTER_EXTRACTOR_CURVATURE_H
#define EUCLIDEAN_CLUSTER_EXTRACTOR_CURVATURE_H

#include <pcl/search/pcl_search.h>
#include <pcl/pcl_base.h>

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <pcl/features/boundary.h>

#include <pcl/features/normal_3d.h>

#include <Eigen/Core>



namespace oph {

template<typename PointT, typename NormalT>
class EuclideanClusterExtractorCurvature : public pcl::PCLBase<PointT>
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
  EuclideanClusterExtractorCurvature() :
    tree_(), normals_(), boundary_(), fixation_point_(), fp_index_(), search_radius_(0.02)
  {
  }

  /* \brief empty destructor */
  virtual ~EuclideanClusterExtractorCurvature()
  {
  }

  /** \brief Set the fixation point.
    * \param[in] p the fixation point
    */
  void
  setFixationPoint(const PointT &p);

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
  void segment(pcl::PointIndices &indices_out);

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

};

}
#endif /* EUCLIDEAN_CLUSTER_EXTRACTOR_CURVATURE_H */

