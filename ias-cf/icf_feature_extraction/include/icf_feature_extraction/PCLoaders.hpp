/*
 * Copyright (c) 2012,
 * Zoltan-Csaba Marton <marton@cs.tum.edu>,
 * Ferenc Balint-Benczedi <balintb.ferenc@gmail.com>,
 * Florian Seidel <seidel.florian@gmail.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Intelligent Autonomous Systems Group/
 *       Technische Universitaet Muenchen nor the names of its contributors
 *       may be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef PCLOADERS_H
#define PCLOADERS_H

#include <icf_feature_extraction/HierarchicalPCLoader.hpp>

//stl
#include <string>
#include <vector>
#include <ctime>

//ros
#include <ros/ros.h>

//boost
#define BOOST_FILESYSTEM_VERSION 2
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/make_shared.hpp>
#include <boost/math/special_functions/fpclassify.hpp> // isnan
//pcl
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
//#include <pcl/features/rsd.h>
#include <pcl/features/feature.h>
#include <pcl/features/vfh.h>
#include <pcl/features/pfh.h>
#include <pcl/features/pfhrgb.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/features/spin_image.h>
#include <pcl/filters/passthrough.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

//#include <object_part_decomposition/point_type.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
//#include <c3_hlac/c3_hlac_tools.h>

using namespace pcl;
using namespace boost;

#define NR_CLASS 5
#define NOISE 0
#define PLANE 1
#define CYLINDER 2
#define SPHERE 3
#define EDGE 4
#define EMPTY 5

namespace icf
{

template<typename PT>
  class KMeansDownsampleFilter
  {
  public:
    unsigned int k;
    KMeansDownsampleFilter(unsigned int k) :
        k(k)
    {
      assert(k >= 1);
    }

    virtual ~KMeansDownsampleFilter()
    {

    }

    void filter(typename PointCloud<PT>::Ptr cloud, PointCloud<Normal>::Ptr normals, PointCloud<PT>& filtered,
                PointCloud<Normal>& filteredNormals)
    {
      assert(cloud->points.size()>0);
      if (k >= cloud->points.size())
      {
        filtered = *cloud;
        filteredNormals = *normals;
        return;
      }
      //create matrix filled with points
      cv::Mat_<float> points(cloud->points.size(), 3);
      for (unsigned int i = 0; i < cloud->points.size(); i++)
      {
        float * row = points[i];
        PT point = cloud->points.at(i);
        row[0] = point.x;
        row[1] = point.y;
        row[2] = point.z;
      }

      cv::Mat_<float> centers;
      cv::Mat_<int> labels;
      cv::kmeans(points, k, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER, 50, 0.0), 1, cv::KMEANS_PP_CENTERS,
                 centers);
      for (size_t _k = 0; _k < k; _k++)
      {
        double best_dist = DBL_MAX;
        int best_index = -1;
        float * center = centers[_k];
        for (int i = 0; i < labels.rows; i++)
        {
          float * row = points[i];
          if ((size_t)(labels[i][0]) == _k)
          {
            double dist = (row[0] - center[0]) * (row[0] - center[0]);
            dist += (row[1] - center[1]) * (row[1] - center[1]);
            dist += (row[2] - center[2]) * (row[2] - center[2]);
            if (dist < best_dist)
            {
              best_dist = dist;
              best_index = i;
            }
          }
        }
        assert(best_index!=-1);
        filtered.points.push_back(cloud->points.at(best_index));
        filteredNormals.points.push_back(normals->points.at(best_index));
        best_dist = DBL_MAX;
      }
      filteredNormals.width = filteredNormals.points.size();
      filteredNormals.height = 1;
      filtered.width = filtered.points.size();
      filtered.height = 1;
#ifdef DEBUG_KMEANS_DOWNSAMPLE
      PCDWriter writer;
      writer.write("filtered.pcd", filtered);
      writer.write("original.pcd", *cloud);
      PCDReader reader;
      std::cout<<"asfas"<<std::endl;
      assert(reader.read("filtered.pcd",filtered)!=-1);
#endif
    }
  };

class MatFileLoader : public FileLoader
{
public:
  typedef boost::shared_ptr<MatFileLoader> Ptr;
  virtual cv::Mat_<float>
  getFeatures()=0;
};

template<typename H>
  class HistogramReader
  {
    virtual ~HistogramReader();
  public:
    virtual float
    readBin(H& histogram, int i)=0;
  };

template<typename H>
  class FloatArrayReader : public HistogramReader<H>
  {
    virtual ~FloatArrayReader();
  public:
    virtual float readBin(H& histogram, int i)
    {
      return (reinterpret_cast<float*>(&histogram))[i];
    }
  };

class SHOTReader : public HistogramReader<SHOT352>
{
public:
  virtual float readBin(SHOT352& histogram, int i)
  {
    //assert(histogram.descriptor.size()==352);
    return histogram.descriptor[i];
  }
};

template<typename PT, typename H, typename FE>
  class LocalFeatureLoader : public MatFileLoader
  {
  public:

    LocalFeatureLoader(int maxFeaturesPerFile, int k, double radius, int fk, double fradius, double leafSize = 0.005,
                       int skip = 5, int histogramLength = -1, HistogramReader<H>* hr = new FloatArrayReader<H>()) :
        maxFeaturesPerFile(maxFeaturesPerFile), k(k), radius(radius), fk(k), fradius(fradius), leafSize(leafSize), skip(
            skip), skipcount(1), hr(hr)
    {
      if (histogramLength == -1)
      {
        features = cv::Mat_<float>(0, sizeof(H) / sizeof(float));
      }
      else
      {
        features = cv::Mat_<float>(0, histogramLength);
      }
      FileLoader::featureEstimationTiming = 0.0;
      FileLoader::nrPointsProcessed = 0;
    }

    virtual ~LocalFeatureLoader()
    {

    }

    cv::Mat_<float> getFeatures()
    {
      return features;
    }

    void clear()
    {
      features.resize(0, features.cols);
    }

    virtual void loadFile(const path& p, const std::vector<int>& labels)
    {
      if (p.extension() == ".pcd")
      {
        if (skipcount % skip != 0)
        {
          std::cout << "Skipping " << p.string() << std::endl;
          skipcount++;
          return;
        }
        skipcount = 1;
        typename PointCloud<PT>::Ptr cloud(new PointCloud<PT>());

        if (reader.read(p.string(), *cloud) == -1)
        {
          std::stringstream str;
          str << "Can't load cloud from file: " << p.filename();
          throw PCDLoaderException(str.str());
        }
        else
        {
          std::cout << "Extracting from file: " << p.filename() << std::endl;
        }
        typename PointCloud<PT>::Ptr surfaceForNormalEstimation(new PointCloud<PT>()); //estimate normals for those points
        typename PointCloud<PT>::Ptr surfaceForFeatureEstimation(new PointCloud<PT>()); //estimate features for those points
        PointCloud<pcl::Normal>::Ptr normals(new PointCloud<pcl::Normal>());

        typename PointCloud<PT>::Ptr filtered(new PointCloud<PT>());
        typename PointCloud<PT>::Ptr filtered2(new PointCloud<PT>());

        PassThrough<PT> ptFilter;
        ptFilter.setInputCloud(cloud);
        ptFilter.filter(*filtered);

        StatisticalOutlierRemoval<PT> sor;
        sor.setInputCloud(filtered);
        sor.setMeanK(50); //What's a good value for this
        sor.setStddevMulThresh(3.0); //What's a good value for this?
        sor.filter(*filtered2);

        pcl::VoxelGrid<PT> grid;
        grid.setInputCloud(filtered2);
        grid.setLeafSize(leafSize, leafSize, leafSize);
        grid.filter(*surfaceForNormalEstimation); //For every point in this calculate the normals
        //Find min(points,maxFeaturesPerFile) evenly spaced points in the cloud

        typename pcl::search::KdTree<PT>::Ptr originalSurface = make_shared<pcl::search::KdTree<PT> >();

        NormalEstimation<PT, pcl::Normal> normalEstimator;
        normalEstimator.setInputCloud(surfaceForNormalEstimation);
        normalEstimator.setSearchSurface(cloud);
        normalEstimator.setSearchMethod(originalSurface);
        normalEstimator.setKSearch(k);
        normalEstimator.setRadiusSearch(radius);
        normalEstimator.compute(*normals);

        PointCloud<Normal>::Ptr featurePointNormals(new PointCloud<Normal>());
        KMeansDownsampleFilter<PT> filter(maxFeaturesPerFile);
        filter.filter(surfaceForNormalEstimation, normals, *surfaceForFeatureEstimation, *featurePointNormals);

        typename pcl::search::KdTree<PT>::Ptr normalsSurface = make_shared<pcl::search::KdTree<PT> >();
        //normalsSurface->setInputCloud(surfaceForNormalEstimation);
        //Feature extraction
        PointCloud<H> *f = new PointCloud<H>();
        clock_t start = clock();
        FE estimator;
        setUpFeatureEstimation(estimator, surfaceForFeatureEstimation, featurePointNormals, surfaceForNormalEstimation,
                               normalsSurface, normals);
        estimator.compute(*f);
        double featuresTime = ((clock() - start) / (double)CLOCKS_PER_SEC);
        FileLoader::featureEstimationTiming += featuresTime;
        FileLoader::nrPointsProcessed += surfaceForNormalEstimation->points.size();
        cv::Mat_<float> cloudFeatures(f->points.size(), features.cols);
        for (unsigned int p = 0; p < f->points.size(); p++)
        {
          float * row = cloudFeatures[p];
          bool containsNaNs = false;
          for (int i = 0; i < features.cols; i++)
          {
            if (isnan(hr->readBin(f->points.at(p), i))
                || hr->readBin(f->points.at(p), i) == std::numeric_limits<double>::infinity())
            {
              std::cout << "Warning: one of the bins was NaN or infinity; will not be used" << std::endl;
              containsNaNs = true;
              break;
            }
          }
          if (!containsNaNs)
            for (int i = 0; i < features.cols; i++)
            {
              row[i] = hr->readBin(f->points.at(p), i);
            }
        }
        features.push_back(cloudFeatures);
        delete f;
      }
    }
  protected:

    virtual void setUpFeatureEstimation(FE& estimator, typename PointCloud<PT>::Ptr surfaceForFeatureEstimation,
                                        typename PointCloud<Normal>::Ptr featurePointNormals,
                                        typename PointCloud<PT>::Ptr surfaceForNormalEstimation,
                                        typename pcl::search::KdTree<PT>::Ptr normalsSurface,
                                        typename PointCloud<Normal>::Ptr normals)
    {
      estimator.setInputCloud(surfaceForFeatureEstimation);
      estimator.setSearchSurface(surfaceForNormalEstimation);
      estimator.setSearchMethod(normalsSurface);
      estimator.setInputNormals(normals);
      estimator.setKSearch(fk);
      estimator.setRadiusSearch(fradius);
    }

    pcl::IndicesConstPtr createIndices(int size)
    {
      pcl::IndicesPtr indices(new std::vector<int, std::allocator<int> >());
      for (int i = 0; i < size; i++)
      {
        indices->push_back(i);
      }
      return indices;
    }
    ;
    cv::Mat_<float> features;
    int maxFeaturesPerFile;
    int k;
    double radius;
    int fk;
    double fradius;
    bool downsample;
    float leafSize;
    int skip;
    int skipcount;
    int histogramLength;
    HistogramReader<H>* hr;
    PCDReader reader;
  };

template<typename PT>
  class SpinImageLoader :
      public LocalFeatureLoader<PT, Histogram<153>, SpinImageEstimation<PT, Normal, Histogram<153> > >
  {
  public:

    SpinImageLoader(int maxFeaturesPerFile, int k, double radius, int fk, double fradius, double leafSize = 0.005,
                    int skip = 5) :
        LocalFeatureLoader<PT, Histogram<153>, SpinImageEstimation<PT, Normal, Histogram<153> > >(
            maxFeaturesPerFile, k, radius, fk, fradius, leafSize, skip, 153, new FloatArrayReader<Histogram<153> >())
    {

    }
  protected:
    virtual void setUpFeatureEstimation(SpinImageEstimation<PT, Normal, Histogram<153> >& estimator,
                                        typename PointCloud<PT>::Ptr surfaceForFeatureEstimation,
                                        typename PointCloud<Normal>::Ptr featurePointNormals,
                                        typename PointCloud<PT>::Ptr surfaceForNormalEstimation,
                                        typename pcl::search::KdTree<PT>::Ptr normalsSurface,
                                        typename PointCloud<Normal>::Ptr normals)
    {
      estimator.setInputWithNormals(surfaceForFeatureEstimation, featurePointNormals);
      estimator.setSearchMethod(normalsSurface);
      estimator.setSearchSurfaceWithNormals(surfaceForNormalEstimation, normals);
      estimator.setKSearch(
          LocalFeatureLoader<PT, Histogram<153>, SpinImageEstimation<PT, Normal, Histogram<153> > >::fk);
      estimator.setRadiusSearch(
          LocalFeatureLoader<PT, Histogram<153>, SpinImageEstimation<PT, Normal, Histogram<153> > >::fradius);
    }
  };

template<typename PT, typename H, typename FE>
  class BoWExtractor : public MatFileLoader
  {
  public:

    BoWExtractor(cv::Mat_<float> codebook, LocalFeatureLoader<PT, H, FE>* localFeatureLoader) :
        scaled(false), codebook(codebook), localFeatureLoader(localFeatureLoader), matcher(
            DescriptorMatcher::create("FlannBased"))
    {
      matcher->add(vector<Mat>(1, codebook));
      bowFeatures = cv::Mat_<float>(0, codebook.rows);
    }

    virtual ~BoWExtractor()
    {
      if (localFeatureLoader != NULL)
      {
        delete localFeatureLoader;
      }
    }

    virtual void loadFile(const path& p, const std::vector<int>& labels)
    {
      localFeatureLoader->loadFile(p, labels);
      Mat_<float> features = localFeatureLoader->getFeatures();
      if (features.rows > 0)
      {
        vector<DMatch> matches;
        matcher->match(features, matches);
        Mat_<float> imgDescriptor(1, codebook.rows, 0.0f);
        float *dptr = (float*)imgDescriptor.data;
        for (size_t i = 0; i < matches.size(); i++)
        {
          int queryIdx = matches[i].queryIdx;
          int trainIdx = matches[i].trainIdx; // cluster index
          assert( queryIdx == (int)i);
          dptr[trainIdx] = dptr[trainIdx] + 1.f;
        }
        imgDescriptor /= features.rows;
        bowFeatures.push_back(imgDescriptor);
        localFeatureLoader->clear();
        assert(localFeatureLoader->getFeatures().rows==0);
      }
    }

    virtual void scaleBoWFeatures()
    {
      for (int c = 0; c < bowFeatures.cols; c++)
      {
        float min = FLT_MAX;
        float max = -FLT_MAX;
        for (int r = 0; r < bowFeatures.rows; r++)
        {
          float value = bowFeatures(r, c);
          if (value > max)
          {
            max = value;
          }
          else if (value < min)
          {
            min = value;
          }
        }
        if (max != 0.0)
          for (int r = 0; r < bowFeatures.rows; r++)
          {
            bowFeatures(r, c) = (bowFeatures(r, c) - min) / max;
          }
      }
    }

    virtual cv::Mat_<float> getFeatures()
    {
      if (!scaled)
      {
        //scaleBoWFeatures();
        scaled = true;
      }
      return bowFeatures;
    }

  private:
    bool scaled;
    cv::Mat_<float> codebook;
    LocalFeatureLoader<PT, H, FE>* localFeatureLoader;
    cv::Mat_<float> bowFeatures;
    cv::Ptr<DescriptorMatcher> matcher;
  };

template<typename IN, typename N, typename OUT>
  class NormalEstimator : public FileLoader
  {
  private:
    float nRadius;
    float leafSize;
    bool downsample;
    PCDReader reader;
  public:
    NormalEstimator(float nRadius, float leafSize, bool downsample) :
        nRadius(nRadius), leafSize(leafSize), downsample(downsample)
    {

    }

    void loadFile(const path& p, const std::vector<int>& labels)
    {
      if (p.extension() == ".pcd")
      {
        typename PointCloud<IN>::Ptr cloud(new PointCloud<IN>());
        typename PointCloud<N>::Ptr normals(new PointCloud<N>());

        if (reader.read(p.string(), *cloud) != -1)
        {

        }
        else
        {
          std::stringstream str;
          str << "Can't load cloud from file: " << p.filename();
          throw PCDLoaderException(str.str());

        }

        typename PointCloud<IN>::Ptr input;
        typename PointCloud<IN>::Ptr surface;
        typename PointCloud<IN>::Ptr filtered(new PointCloud<IN>());

        PassThrough<IN> filter;
        filter.setInputCloud(cloud);
        filter.filter(*filtered);

        StatisticalOutlierRemoval<IN> sor;
        sor.setInputCloud(filtered);
        sor.setMeanK(50);
        sor.setStddevMulThresh(3.0);
        sor.filter(*cloud);

        if (downsample)
        {
          //std::cout << "Downsampling" << std::endl;
          pcl::VoxelGrid<IN> grid;
          grid.setInputCloud(cloud);
          grid.setLeafSize(leafSize, leafSize, leafSize);
          grid.filter(*filtered);
          input = filtered;
          surface = cloud;
        }
        else
        {
          //
          //std::cout << "not downsampling" << std::endl;
          input = cloud;
          surface = cloud;
          //
        }

        typename pcl::search::KdTree<IN>::Ptr kdTree = make_shared<pcl::search::KdTree<IN> >();
        //pcl::IndicesConstPtr indices = createIndices(filtered->size());

        std::cout << "Estimating normals" << std::endl;
        NormalEstimation<IN, N> normalEstimator;
        normalEstimator.setInputCloud(input);
        normalEstimator.setSearchSurface(surface);
        normalEstimator.setSearchMethod(kdTree);
        //normalEstimator.setIndices(indices);
        normalEstimator.setKSearch(0);
        normalEstimator.setRadiusSearch(nRadius);
        clock_t start = clock();
        normalEstimator.compute(*normals);

        typename pcl::PointCloud<OUT>::Ptr outCloud(new pcl::PointCloud<OUT>());
        assert(input->points.size()==normals->points.size());
        pcl::concatenateFields(*input, *normals, *outCloud);
        string fileName = p.string().substr(0, p.string().length() - 4) + "-normals.pcd";
        pcl::io::savePCDFileASCII(fileName, *outCloud);
        double featuresTime = ((clock() - start) / (double)CLOCKS_PER_SEC);
        FileLoader::featureEstimationTiming += featuresTime;
        FileLoader::nrPointsProcessed += surface->points.size();

      }
    }

  };

template<typename PT, typename NT>
  class FeatureLoader : public FileLoader
  {

  protected:
    pcl::IndicesPtr createIndices(int size)
    {
      pcl::IndicesPtr indices(new std::vector<int, std::allocator<int> >());
      for (int i = 0; i < size; i++)
      {
        indices->push_back(i);
      }
      return indices;
    }
    ;
  public:

    FeatureLoader(DataSet<double>::Matrix& m, float nRadius, bool downsample = true, float leafSize = 0.005, int skip =
                      5,
                  bool readNormals = false, std::string fileEnding = ".pcd") :
        m(m), row(0), nRadius(nRadius), downsample(downsample), leafSize(leafSize), skip(skip), readNormals(
            readNormals), skipcount(1), fileEnding(fileEnding)
    {

    }

    void increaseSize()
    {
      DataSet<double>::Matrix newm(m.rows() * 2, m.cols());
      for (int i = 0; i < m.rows(); i++)
      {
        for (int j = 0; j < m.cols(); j++)
        {
          newm(i, j) = m(i, j);
        }
      }
      m = newm;
    }

    virtual void postProcessing()
    {
      pruneSize();
    }

    DataSet<double>::Matrix&
    getMatrix()
    {
      return m;
    }

    void pruneSize()
    {
      DataSet<double>::Matrix newm(row, m.cols());
      for (int i = 0; i < row; i++)
      {
        for (int j = 0; j < m.cols(); j++)
        {
          newm(i, j) = m(i, j);
        }
      }
      m = newm;
    }

    virtual void
    calcFeature(typename PointCloud<PT>::Ptr input, typename PointCloud<PT>::Ptr surface,
                typename pcl::search::KdTree<PT>::Ptr kdTree, typename PointCloud<NT>::Ptr normals)=0;

    void loadFile(const path& p, const std::vector<int>& labels)
    {
      if (hasEnding(p.string(), fileEnding))
      {
        if (skipcount % skip != 0)
        {
          std::cout << "Skipping " << p.string() << std::endl;
          skipcount++;
          return;
        }
        skipcount = 1;
        typename PointCloud<PT>::Ptr cloud(new PointCloud<PT>());
        typename PointCloud<NT>::Ptr normals(new PointCloud<NT>());

        if (reader.read(p.string(), *cloud) != -1)
        {

        }
        else
        {
          std::stringstream str;
          str << "Can't load cloud from file: " << p.filename();
          throw PCDLoaderException(str.str());
        }

        typename PointCloud<PT>::Ptr input;
        typename PointCloud<PT>::Ptr surface;
        typename PointCloud<PT>::Ptr filtered(new PointCloud<PT>());

        PassThrough<PT> filter;
        filter.setInputCloud(cloud);
        filter.filter(*filtered);

        StatisticalOutlierRemoval<PT> sor;
        sor.setInputCloud(filtered);
        sor.setMeanK(50);
        sor.setStddevMulThresh(3.0);
        sor.filter(*cloud);

        if (downsample)
        {
          //std::cout << "Downsampling" << std::endl;
          pcl::VoxelGrid<PT> grid;
          grid.setInputCloud(cloud);
          grid.setLeafSize(leafSize, leafSize, leafSize);
          grid.filter(*filtered);
          input = filtered;
          surface = cloud;
        }
        else
        {
          //
          //std::cout << "not downsampling" << std::endl;
          input = cloud;
          surface = cloud;
          //
        }

        typename pcl::search::KdTree<PT>::Ptr kdTree = make_shared<pcl::search::KdTree<PT> >();

        //TODO: doesn't make sense will be overwritten by setSearchSurface
        kdTree->setInputCloud(input);

        //pcl::IndicesConstPtr indices = createIndices(filtered->size());
        if (!readNormals)
        {
          std::cout << "Estimating normals" << std::endl;
          NormalEstimation<PT, NT> normalEstimator;
          normalEstimator.setInputCloud(input);

          normalEstimator.setSearchSurface(surface);
          normalEstimator.setSearchMethod(kdTree);
          //normalEstimator.setIndices(indices);
          normalEstimator.setKSearch(0);
          normalEstimator.setRadiusSearch(nRadius);
          normalEstimator.compute(*normals);

        }
        if (row == m.rows())
        {
          increaseSize();
        }
        clock_t start = clock();
        //TODO: remove kdTree argument
        if (readNormals)
        {
        	//TODO: find a way to do this nicely
          calcFeature(input, surface, kdTree, *reinterpret_cast<typename PointCloud<NT>::Ptr*>(&input));
        }
        else
        {
          calcFeature(input, surface, kdTree, normals);
        }
        double featuresTime = ((clock() - start) / (double)CLOCKS_PER_SEC);
        FileLoader::featureEstimationTiming += featuresTime;
        FileLoader::nrPointsProcessed += surface->points.size();
        row++;

      }
    }
    DataSet<double>::Matrix& m;
    int row;
    float nRadius;
    bool downsample;
    float leafSize;
    PCDReader reader;
    int skip;
    bool readNormals;
    int skipcount;
    std::string fileEnding;

  };

/*template<typename PT, typename NT>
 class GRSDLoader : public FeatureLoader<PT, NT>
 {

 private:
 float rsdRadius;
 public:

 GRSDLoader(DataSet<double>::Matrix& m, float nRadius, float rsdRadius = 0.03, bool downsample = true,
 float leafSize = 0.01, int skip = 5, bool readNormals = false, std::string fileEnding = ".pcd") :
 FeatureLoader<PT, NT>(m, nRadius, downsample, leafSize, skip, readNormals, fileEnding), rsdRadius(rsdRadius)
 {
 }

 int get_type(float min_radius, float max_radius, double min_radius_plane, double max_radius_noise,
 double min_radius_cylinder, double max_min_radius_diff)
 {

 @NOTE: this should be in PCL already
 min_radius *= 1.1;
 max_radius *= 0.9;
 if (min_radius > max_radius) {
 const double t = min_radius;
 min_radius = max_radius;
 max_radius = t;
 }


 //                    0.100
 if (min_radius > min_radius_plane)
 return PLANE; // plane
 //                    0.175
 else if (max_radius > min_radius_cylinder)
 return CYLINDER; // cylinder (rim)
 //                    0.015
 else if (min_radius < max_radius_noise)
 return NOISE; // noise/corner
 //                                    0.05
 else if (max_radius - min_radius < max_min_radius_diff)
 return SPHERE; // sphere/corner
 else
 return EDGE; // edge
 }

 //
 //	virtual void calcFeature(PointCloud<pcl::PointXYZ>::Ptr input_cloudPtr,
 //			PointCloud<pcl::PointXYZ>::Ptr surface,
 //			KdTreeFLANN<pcl::PointXYZ>::Ptr kdTree,
 //			PointCloud<pcl::Normal>::Ptr cloud_normalsPtr) {

 void calcFeature(typename pcl::PointCloud<PT>::Ptr input_cloudPtr, typename PointCloud<PT>::Ptr surface,
 typename pcl::search::KdTree<PT>::Ptr kdTree, typename pcl::PointCloud<NT>::Ptr cloud_normalsPtr)
 {
 double min_radius_plane = 0.100;
 double max_radius_noise = 0.015;
 double min_radius_cylinder = 0.175;
 double max_min_radius_diff = 0.050;
 //double min_radius_edge_ = 0.030;

 //static int i=1;
 //std::cerr<<"Nr:"<<i++<<endl;
 double downsample_leaf = FeatureLoader<PT, NT>::leafSize;
 double rsd_radius_search = rsdRadius;
 // Create the voxel grid
 typename pcl::PointCloud<PT>::Ptr cloud_downsampled(new pcl::PointCloud<PT>());
 pcl::VoxelGrid<PT> grid;
 grid.setLeafSize(downsample_leaf, downsample_leaf, downsample_leaf);
 grid.setInputCloud(input_cloudPtr);
 grid.setSaveLeafLayout(true); // TODO avoid this using nearest neighbor search
 grid.filter(*cloud_downsampled);

 // Compute RSD
 pcl::PointCloud<pcl::PrincipalRadiiRSD>::Ptr radii(new pcl::PointCloud<pcl::PrincipalRadiiRSD>());
 pcl::RSDEstimation<PT, NT, pcl::PrincipalRadiiRSD> rsd;
 rsd.setInputCloud(cloud_downsampled);
 rsd.setSearchSurface(input_cloudPtr);
 rsd.setInputNormals(cloud_normalsPtr);
 rsd.setRadiusSearch(std::max(rsd_radius_search, sqrt(3.0) * downsample_leaf / 2));
 typename pcl::search::KdTree<PT>::Ptr tree = boost::make_shared<pcl::search::KdTree<PT> >();
 tree->setInputCloud(input_cloudPtr);
 rsd.setSearchMethod(tree);

 rsd.compute(*radii);
 //ROS_INFO("RSD compute done in %f seconds.", my_clock()-t1);

 //pcl::PointCloud<pcl::PointNormalRADII> cloud_downsampled_radii;
 //pcl::concatenateFields (cloud_downsampled, radii, cloud_downsampled_radii);

 // Get rmin/rmax for adjacent 27 voxel
 Eigen::MatrixXi relative_coordinates(3, 13);

 Eigen::MatrixXi transition_matrix = Eigen::MatrixXi::Zero(NR_CLASS + 1, NR_CLASS + 1);

 int idx = 0;

 // 0 - 8
 for (int i = -1; i < 2; i++)
 {
 for (int j = -1; j < 2; j++)
 {
 relative_coordinates(0, idx) = i;
 relative_coordinates(1, idx) = j;
 relative_coordinates(2, idx) = -1;
 idx++;
 }
 }
 // 9 - 11
 for (int i = -1; i < 2; i++)
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
 relative_coordinates_all.block < 3, 13 > (0, 0) = relative_coordinates;
 relative_coordinates_all.block < 3, 13 > (0, 13) = -relative_coordinates;

 // SAVE THE TYPE OF EACH POINT
 std::vector<int> types(radii->points.size());

 for (size_t idx = 0; idx < radii->points.size(); ++idx)
 types[idx] = get_type(radii->points[idx].r_min, radii->points[idx].r_max, min_radius_plane, max_radius_noise,
 min_radius_cylinder, max_min_radius_diff);

 for (size_t idx = 0; idx < cloud_downsampled->points.size(); ++idx)
 {
 int source_type = types[idx];
 std::vector<int> neighbors = grid.getNeighborCentroidIndices(cloud_downsampled->points[idx],
 relative_coordinates_all);
 for (unsigned id_n = 0; id_n < neighbors.size(); id_n++)
 {
 int neighbor_type;
 if (neighbors[id_n] == -1)
 neighbor_type = EMPTY;
 else
 neighbor_type = types[neighbors[id_n]];

 transition_matrix(source_type, neighbor_type)++;}
 }
 // pcl::PointCloud<pcl::GRSDSignature21> cloud_grsd;
 // cloud_grsd.points.resize(1);

 int      nrf = 0;

 for (int i = 0; i < NR_CLASS + 1; i++)
 {
 for (int j = i; j < NR_CLASS + 1; j++)
 {
 //std::cout << transition_matrix(i, j) << " ";
 FeatureLoader<PT, NT>::m(FeatureLoader<PT, NT>::row, nrf++) = transition_matrix(i, j)
 + transition_matrix(j, i);
 }
 }

 //uncomment if you want to see the computed GRSD-s

 //std::cerr << "transition matrix" << std::endl << transition_matrix << std::endl;
 //std::cerr << std::endl<<descriptor.transpose()<<std::endl;
 //ROS_INFO("GRSD compute done in %f seconds.", my_clock()-t1);
 //return cloud_grsd;
 }

 };*/

template<typename PT, typename NT>
  class GlobalSHOTLoader : public FeatureLoader<PT, NT>
  {

  public:

    GlobalSHOTLoader(DataSet<double>::Matrix& m, float nRadius, float radius = // paramater not used, it is always that value
                         10000000 /* thx Aitor :P */,
                     bool downsample = true, float leafSize = 0.005, int skip = 5, bool readNormals = false,
                     std::string fileEnding = ".pcd") :
        FeatureLoader<PT, NT>(m, nRadius, downsample, leafSize, skip, readNormals, fileEnding)
    {
    }

    //
    //	virtual void calcFeature(PointCloud<pcl::PointXYZ>::Ptr input_cloudPtr,
    //			PointCloud<pcl::PointXYZ>::Ptr surface,
    //			KdTreeFLANN<pcl::PointXYZ>::Ptr kdTree,
    //			PointCloud<pcl::Normal>::Ptr cloud_normalsPtr) {

    void calcFeature(typename pcl::PointCloud<PT>::Ptr input_cloudPtr, typename PointCloud<PT>::Ptr surface,
                     typename pcl::search::KdTree<PT>::Ptr kdTree, typename pcl::PointCloud<NT>::Ptr cloud_normalsPtr)
    {

      Eigen::Vector4f centroid4f;
      pcl::compute3DCentroid(*input_cloudPtr, centroid4f);

      typedef pcl::PointNormal PointOutT;
      typename pcl::PointCloud<PointOutT>::Ptr input_cloud(new pcl::PointCloud<PointOutT>(/* *cloud_normals */));
      pcl::concatenateFields(*input_cloudPtr, *cloud_normalsPtr, *input_cloud);
      input_cloud->points.resize(input_cloud->points.size() + 1);
      input_cloud->width = input_cloud->points.size();
      input_cloud->height = 1;

      centroid4f[3] = 0;
      Eigen::Vector4f max_pt(0, 0, 0, 0);
      pcl::getMaxDistance(*input_cloudPtr, centroid4f, max_pt);

      //create fake point at the centroid...
      input_cloud->points[input_cloud->points.size() - 1].getVector4fMap() = centroid4f;
      input_cloud->points[input_cloud->points.size() - 1].getNormalVector4fMap() = -centroid4f;

      boost::shared_ptr<std::vector<int> > indices(new std::vector<int>());
      indices->push_back(input_cloud->points.size() - 1);

      typedef typename pcl::SHOTEstimation<PointOutT, PointOutT, pcl::SHOT352> SHOTEstimation;
      SHOTEstimation shot_estimate;
      //typename pcl::search::KdTree<PointOutT>::Ptr tree (new pcl::search::KdTree<PointOutT>);
      typename pcl::search::KdTree<PointOutT>::Ptr tree(new pcl::search::KdTree<PointOutT>);

      shot_estimate.setSearchMethod(tree);
      shot_estimate.setIndices(indices);
      shot_estimate.setInputCloud(input_cloud);
      shot_estimate.setInputNormals(input_cloud);
      shot_estimate.setRadiusSearch(10000000); //(centroid4f - max_pt).norm ()*10);

      pcl::PointCloud<pcl::SHOT352>::Ptr shots(new pcl::PointCloud<pcl::SHOT352>);
      shot_estimate.compute(*shots);

      //std::cerr << "SHOT: " << shots->points[0].descriptor.size() << std::endl;

      //std::cout << "#";
      int nrf = 0;
      for (size_t i = 0; i < 352/*shots->points[0].descriptor.size()*/; ++i)
      {
        std::cout << " " << shots->points[0].descriptor[nrf];
        if (std::isfinite(shots->points[0].descriptor[nrf]) && !std::isnan(shots->points[0].descriptor[nrf])
            && !boost::math::isnan(shots->points[0].descriptor[nrf]))
          FeatureLoader<PT, NT>::m(FeatureLoader<PT, NT>::row, nrf++) = shots->points[0].descriptor[nrf];
        else
        {
          //std::cerr << "NaN in poistion " << nrf << std::endl;
          //std::cerr << shots->points[0].descriptor[nrf] << std::endl;
          FeatureLoader<PT, NT>::m(FeatureLoader<PT, NT>::row, nrf++) = 0;
        }
      }
      //std::cout << std::endl;

    }

  };
template<typename PT, typename NT>
  class C3HLACLoader : public FeatureLoader<PT, NT>
  {
  private:
    float rsdRadius;
  public:

    C3HLACLoader(DataSet<double>::Matrix& m, float nRadius, float rsdRadius = 0.03, bool downsample = true,
                 float leafSize = 0.02, int skip = 5, bool readNormals = false, std::string fileEnding = ".pcd") :
        FeatureLoader<PT, NT>(m, nRadius, downsample, leafSize, skip, readNormals, fileEnding), rsdRadius(rsdRadius)
    {
    }

    //
    //      virtual void calcFeature(PointCloud<pcl::PointXYZ>::Ptr input_cloudPtr,
    //                      PointCloud<pcl::PointXYZ>::Ptr surface,
    //                      KdTreeFLANN<pcl::PointXYZ>::Ptr kdTree,
    //                      PointCloud<pcl::Normal>::Ptr cloud_normalsPtr) {

    virtual void calcFeature(typename PointCloud<PT>::Ptr input_cloudPtr, typename PointCloud<PT>::Ptr surface,
                             typename pcl::search::KdTree<PT>::Ptr kdTree, typename PointCloud<NT>::Ptr normals)
    {

      typename pcl::PointCloud<PT>::Ptr copy_cloud(new pcl::PointCloud<PT>);
      pcl::copyPointCloud(*input_cloudPtr, *copy_cloud);

      typename pcl::VoxelGrid<PT> grid;
      typename pcl::PointCloud<PT> cloud_downsampled;
      getVoxelGrid(grid, *copy_cloud, cloud_downsampled, 0.02);

      std::vector<float> c3_hlac;
      extractC3HLACSignature117(grid, cloud_downsampled, c3_hlac, 127, 127, 127, 0.02);
      //descriptor.resize(c3_hlac.size());
      int nrf = 0;
      for (size_t i = 0; i < c3_hlac.size(); ++i)
      {
        std::cerr << c3_hlac[i];
        FeatureLoader<PT, NT>::m(FeatureLoader<PT, NT>::row, i) = c3_hlac[i];
      }
      std::cerr << std::endl;
    }

  };

template<typename PT, typename NT, typename FE, typename H>
  class FFNLoader : public FeatureLoader<PT, NT>
  {
  public:
    FFNLoader(DataSet<double>::Matrix& m, float radius = 0.03, float nRadius = 0.02, int k = 0, bool downSample = true,
              float leafSize = 0.005, int skip = 1, bool readNormals = false, bool useOnePoint = false,
              std::string fileEnding = ".pcd") :
        FeatureLoader<PT, NT>(m, nRadius, downSample, leafSize, skip, readNormals, fileEnding), radius(radius), k(k), useOnePoint(
            useOnePoint)
    {

    }

    virtual void calcFeature(typename PointCloud<PT>::Ptr input, typename PointCloud<PT>::Ptr surface,
                             typename pcl::search::KdTree<PT>::Ptr kdTree, typename PointCloud<NT>::Ptr normals)
    {

      PointCloud<H> *f = new PointCloud<H>();
      f->width = 1;
      f->height = 1;
      f->points.resize(1);
      typename pcl::search::KdTree<PT>::Ptr kdTree_ = make_shared<pcl::search::KdTree<PT> >();
      kdTree->setInputCloud(input);

      FE estimator;

      setUpFeatureEstimation(kdTree_, estimator, input, normals);

      try
      {
        estimator.compute(*f);

        for (int i = 0; i < FeatureLoader<PT, NT>::m.cols(); i++)
        {
          std::cout << f->points.at(0).histogram[i] << " ";
          FeatureLoader<PT, NT>::m(FeatureLoader<PT, NT>::row, i) = f->points.at(0).histogram[i];

        }
        std::cout << std::endl;
      }
      catch (...)
      {
        for (int i = 0; i < FeatureLoader<PT, NT>::m.cols(); i++)
        {
          FeatureLoader<PT, NT>::m(FeatureLoader<PT, NT>::row, i) = 0.0;
        }
        //std::cout << std::endl;
        std::cerr << "Exception thrown!" << std::endl;
      }
      delete f;
    }
  protected:
    virtual void setUpFeatureEstimation(typename pcl::search::KdTree<PT>::Ptr kdTree, FE& estimator,
                                        typename PointCloud<PT>::Ptr input, typename PointCloud<NT>::Ptr normals)
    {
      estimator.setInputCloud(input);
      estimator.setSearchMethod(kdTree);
      estimator.setInputNormals(normals);
      if (!useOnePoint)
      {
        estimator.setIndices(FeatureLoader<PT, NT>::createIndices(input->size()));
        estimator.setKSearch(k);
        estimator.setRadiusSearch(radius);
      }
      else
      {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*input, centroid);
        double min_dist = DBL_MAX;
        int index = 0;
        for (size_t i = 0; i < input->points.size(); i++)
        {
          double dist = 0.0;
          double d = input->points[i].x - centroid(0);
          dist = d * d;
          d = input->points[i].y - centroid(1);
          dist += d * d;
          d = input->points[i].x - centroid(2);
          dist += d * d;
          if (dist < min_dist)
          {
            index = i;
            min_dist = dist;
          }
        }
        pcl::IndicesPtr indices(new std::vector<int, std::allocator<int> >());
        indices->push_back(index);
        estimator.setIndices(indices);
        estimator.setKSearch(input->size() - 1);
      }
    }
    float radius;
    int k;
    bool useOnePoint;

  };

  template<typename PT, typename NT, typename H>
  class FFNLoader<PT,NT,pcl::VFHEstimation<PT, NT, pcl::VFHSignature308>,H > : public FeatureLoader<PT, NT>
  {
  public:
    FFNLoader(DataSet<double>::Matrix& m, float radius = 0.03, float nRadius = 0.02, int k = 0, bool downSample = true,
              float leafSize = 0.005, int skip = 1, bool readNormals = false, bool useOnePoint = false,
              std::string fileEnding = ".pcd") :
        FeatureLoader<PT, NT>(m, nRadius, downSample, leafSize, skip, readNormals, fileEnding), radius(radius), k(k), useOnePoint(
            useOnePoint)
    {

    }

    virtual void calcFeature(typename PointCloud<PT>::Ptr input, typename PointCloud<PT>::Ptr surface,
                             typename pcl::search::KdTree<PT>::Ptr kdTree, typename PointCloud<NT>::Ptr normals)
    {

      PointCloud<H> *f = new PointCloud<H>();
      f->width = 1;
      f->height = 1;
      f->points.resize(1);
      typename pcl::search::KdTree<PT>::Ptr kdTree_ = make_shared<pcl::search::KdTree<PT> >();
      kdTree->setInputCloud(input);

      pcl::VFHEstimation<PT, NT, pcl::VFHSignature308> estimator;

      setUpFeatureEstimation(kdTree_, estimator, input, normals);

      try
      {
        estimator.compute(*f);
        for (int i = 0; i < FeatureLoader<PT, NT>::m.cols(); i++)
        {
          std::cout << f->points.at(0).histogram[i] << " ";
          FeatureLoader<PT, NT>::m(FeatureLoader<PT, NT>::row, i) = f->points.at(0).histogram[i];

        }
        std::cout << std::endl;
      }
      catch (...)
      {
        for (int i = 0; i < FeatureLoader<PT, NT>::m.cols(); i++)
        {
          FeatureLoader<PT, NT>::m(FeatureLoader<PT, NT>::row, i) = 0.0;
        }
        //std::cout << std::endl;
        std::cerr << "Exception thrown!" << std::endl;
      }
      delete f;
    }
  protected:
    virtual void setUpFeatureEstimation(typename pcl::search::KdTree<PT>::Ptr kdTree, pcl::VFHEstimation<PT, NT, pcl::VFHSignature308>& estimator,
                                        typename PointCloud<PT>::Ptr input, typename PointCloud<NT>::Ptr normals)
    {
      estimator.setInputCloud(input);
      estimator.setSearchMethod(kdTree);
      estimator.setInputNormals(normals);
      if (!useOnePoint)
      {
        estimator.setIndices(FeatureLoader<PT, NT>::createIndices(input->size()));
        estimator.setKSearch(k);
        estimator.setRadiusSearch(radius);
      }
      else
      {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*input, centroid);
        double min_dist = DBL_MAX;
        int index = 0;
        for (size_t i = 0; i < input->points.size(); i++)
        {
          double dist = 0.0;
          double d = input->points[i].x - centroid(0);
          dist = d * d;
          d = input->points[i].y - centroid(1);
          dist += d * d;
          d = input->points[i].x - centroid(2);
          dist += d * d;
          if (dist < min_dist)
          {
            index = i;
            min_dist = dist;
          }
        }
        pcl::IndicesPtr indices(new std::vector<int, std::allocator<int> >());
        indices->push_back(index);
        estimator.setIndices(indices);
        estimator.setKSearch(input->size() - 1);
      }
      estimator.setViewPoint(input->sensor_origin_.x(),input->sensor_origin_.y(),input->sensor_origin_.z());
    }
    float radius;
    int k;
    bool useOnePoint;
  };

}

#endif
