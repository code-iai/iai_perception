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

#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

#include <tclap/CmdLine.h>
#include <icf_feature_extraction/PCLoaders.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

using namespace icf;
using namespace boost;
using namespace std;
using namespace TCLAP;
using namespace pcl;

TEST(FeatureExtractionTest,TestVFH)
{
  // init dataset
  DS::Matrix extractedFeatures;
  extractedFeatures = DS::Matrix(1, 308);

  // initialize objects
  float not_used = 0;
  FileLoader::Ptr fl;
  fl =
      FileLoader::Ptr(
          new FFNLoader<PointNormal, PointNormal, VFHEstimation<PointNormal, PointNormal, VFHSignature308>,
              VFHSignature308>(extractedFeatures, not_used, not_used, (int)not_used, false, not_used, 1, true, false,
                               "normals.pcd"));
  HierarchicalPCDLoader loader(fl, true);

  // go over list of files and estimate features
  path basePath = path("data/training/");
  loader.load(basePath);

  // check features
  EXPECT_EQ(extractedFeatures.rows(), 5);
  EXPECT_EQ(extractedFeatures.cols(), 308);
  EXPECT_NEAR(extractedFeatures(0,0), 0.767085, 1e-5);
  EXPECT_NEAR(extractedFeatures(4,307), 0.0, 1e-5);
  Eigen::VectorXd sums = extractedFeatures.rowwise().sum();
  for (int i = 0; i < sums.size(); i++)
    EXPECT_NEAR(sums(i), 400.0, 1.0);
}

/*TEST(FeatureExtractionTest,TestGRSD)
 {
 // init dataset
 DS::Matrix extractedFeatures;
 extractedFeatures=DS::Matrix(1, 21);

 // initialize objects
 float not_used = 0;
 FileLoader::Ptr fl;
 fl = FileLoader::Ptr(new GRSDLoader<PointNormal, PointNormal>
 (extractedFeatures, not_used, 0.01, false, 0.01, 1, true, "normals.pcd"));
 HierarchicalPCDLoader loader(fl, true);

 // go over list of files and estimate features
 path basePath = path("data/training/");
 loader.load(basePath);

 // check features
 EXPECT_EQ(extractedFeatures.rows(), 5);
 EXPECT_EQ(extractedFeatures.cols(), 21);
 EXPECT_EQ(extractedFeatures(0,0), 184.0);
 EXPECT_EQ(extractedFeatures(4,20), 0.0);
 Eigen::VectorXd sums = extractedFeatures.rowwise().sum();
 EXPECT_EQ(sums(0), 7462);
 EXPECT_EQ(sums(4), 16293);


 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
 if (pcl::io::loadPCDFile<pcl::PointXYZ> ("data/training/1_obj050_mug3_green/mug3_0000-ascii-normals.pcd", *cloud) == -1) // load the file
 {
 PCL_ERROR ("Couldn't read file data/training/1_obj050_mug3_green/mug3_0000-ascii-normals.pcd \n");
 }
 const double voxel_size = 0.01;
 pcl::VoxelGrid<pcl::PointXYZ> grid;
 grid.setInputCloud(cloud);
 grid.setLeafSize(voxel_size,voxel_size,voxel_size);
 grid.filter (*cloud);
 EXPECT_EQ(cloud->points.size(), 7462/26);

 cloud->clear();
 if (pcl::io::loadPCDFile<pcl::PointXYZ> ("data/training/2_obj037_cereals/honey_bsss_cereal_0090-ascii-normals.pcd", *cloud) == -1) // load the file
 {
 PCL_ERROR ("Couldn't read file data/training/1_obj050_mug3_green/mug3_0000-ascii-normals.pcd \n");
 }
 grid.setInputCloud(cloud);
 grid.filter (*cloud);
 EXPECT_EQ(cloud->points.size(), 16293/26);
 }

 TEST(FeatureExtractionTest,TestLabels)
 {
 LabelLoader *ll = new LabelLoader(1,".pcd");
 FileLoader::Ptr fl(ll);
 HierarchicalPCDLoader loader(fl, true);
 path basePath = path("data/training/");

 loader.load(basePath);
 unsigned int longest = 0;
 vector<vector<int> > labels = ll->getLabels();
 for (unsigned int i = 0; i < labels.size(); i++)
 {
 unsigned int depth = labels.at(i).size();
 if (depth > longest)
 longest = depth;
 }
 DS::Matrix extractedLabels;
 extractedLabels=DS::Matrix(labels.size(), longest);
 for (unsigned int i = 0; i < labels.size(); i++)
 {
 for (unsigned int j = 0; j < longest; j++)
 {
 if (j >= labels.at(i).size())
 {
 extractedLabels(i, j) = -1.0;
 }
 else
 {
 extractedLabels(i, j) = (double)labels.at(i).at(j);
 }
 }
 }
 EXPECT_EQ(extractedLabels.rows(), 5);
 EXPECT_EQ(extractedLabels.cols(), 1);
 EXPECT_EQ(extractedLabels(0,0), 1);
 EXPECT_EQ(extractedLabels(4,0), 2);
 }*/

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

