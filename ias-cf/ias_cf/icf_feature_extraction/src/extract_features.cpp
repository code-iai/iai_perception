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

#include <tclap/CmdLine.h>
#include <icf_feature_extraction/PCLoaders.hpp>
#include <fstream>

using namespace icf;
using namespace boost;
using namespace std;
using namespace TCLAP;
using namespace pcl;

//#define INSTANTIATE_LOADER(P,FE,H) 	fl = FileLoader::Ptr(new FeatureLoader<P,FE,H>())

//#define GET_FEATURES(P,F) 	((FeatureLoader<P,F>*) fl.get())->getMatrix()

int main(int argc, char ** argv)
{
  //-------------------- Set up command line parser-----------------------------------
  //Command line parser
  CmdLine cmdLine("rospcloader", ' ', "dev");
  //Feature
  vector<string> availableFeatures;
  availableFeatures.push_back("vfh");
  availableFeatures.push_back("pfh");
  availableFeatures.push_back("fpfh");
  //availableFeatures.push_back("grsd2");
  availableFeatures.push_back("gshot");
  availableFeatures.push_back("spin");

  availableFeatures.push_back("labels");
  availableFeatures.push_back("names");

  //availableFeatures.push_back("c3hlac");

  ValuesConstraint<string> availableFeaturesConstraint(availableFeatures);
  ValueArg<string> featuresArg("f", "feature", "Then name of a feature to be extracted", true, "vfh",
                               &availableFeaturesConstraint);

  //base dir
  ValueArg<string> baseDirArg("b", "baseDir", "the name of the base folder to be scanned", false, ".", "string");
  SwitchArg foldersProvideLabelsArg("l", "folderLabel", "first characters of folder name are class numbers");

  //Scale?
  SwitchArg scaleArg("s", "scale", "scale each feature to [-1;1]");

  ValueArg<int> skipArg("", "skip", "skip every nth sample", false, 1, "int>0");

  //name?
  ValueArg<string> nameArg("n", "name", "name of the feature", true, "x", "string");

  ValueArg<string> outArg("o", "out", "The name of the output file", false, "out", "string");

  ValueArg<float> leafSizeArg("d", "leafSize", "Leaf size for downsampling", false, 0.01, "float");

  ValueArg<int> kArg("k", "knn", "use k nearest neighbours in feature estimation", false, 0, "int");

  ValueArg<float> rArg("r", "radiusSearch", "use radius r in radius search", false, 0.02, "float");

  ValueArg<std::string> eArg("e", "ending", "use files ending with this", false, ".pcd", "string");

  ValueArg<float> outliersKArg("", "outliers-k", "analyze knn for outputlier removal", false, 50, "int");

  ValueArg<int> outliersSDArg("", "outliers-sd", "standard deviations for removal or outliers", false, 2.0, "float");

  ValueArg<float> normalEstArg("", "normal-r", "normal estimation radius", false, 0.02, "float");

  SwitchArg readNormalsArg("", "normals", "point cloud contains normals");

  cmdLine.add(outliersKArg);
  cmdLine.add(outliersSDArg);
  cmdLine.add(kArg);
  cmdLine.add(rArg);
  cmdLine.add(leafSizeArg);
  cmdLine.add(featuresArg);
  cmdLine.add(normalEstArg);
  cmdLine.add(baseDirArg);
  cmdLine.add(foldersProvideLabelsArg);
  cmdLine.add(scaleArg);
  cmdLine.add(nameArg);
  cmdLine.add(skipArg);
  cmdLine.add(outArg);
  cmdLine.add(eArg);
  cmdLine.add(readNormalsArg);
  cmdLine.parse(argc, argv);

  DataSet<double> ds;
  DS::Matrix extractedFeatures;

  std::cout<<featuresArg.getValue()<<std::endl;

  if (featuresArg.getValue() != "labels" && featuresArg.getValue() != "names")
  {
    FileLoader::Ptr fl;
    if (readNormalsArg.getValue())
    {
      if (featuresArg.getValue() == "vfh")
      {
        extractedFeatures = DS::Matrix(1, 308);
        fl = FileLoader::Ptr(
            new FFNLoader<PointNormal, PointNormal, VFHEstimation<PointNormal, PointNormal, VFHSignature308>,
                VFHSignature308>(extractedFeatures, rArg.getValue(), normalEstArg.getValue(), kArg.getValue(),
                                 leafSizeArg.isSet(), leafSizeArg.getValue(), skipArg.getValue(),
                                 readNormalsArg.getValue(), false, eArg.getValue()));

      }
      else if (featuresArg.getValue() == "fpfh")
      {
        extractedFeatures = DS::Matrix(1, 33);
        fl = FileLoader::Ptr(
            new FFNLoader<PointNormal, PointNormal, FPFHEstimation<PointNormal, PointNormal, FPFHSignature33>,
                FPFHSignature33>(extractedFeatures, rArg.getValue(), normalEstArg.getValue(), kArg.getValue(),
                                 leafSizeArg.isSet(), leafSizeArg.getValue(), skipArg.getValue(),
                                 readNormalsArg.getValue(), true, eArg.getValue()));
      }
      else if (featuresArg.getValue() == "gshot")
      {
        extractedFeatures = DS::Matrix(1, 352);
        fl = FileLoader::Ptr(
            new GlobalSHOTLoader<PointNormal, PointNormal>(extractedFeatures, normalEstArg.getValue(), rArg.getValue(),
                                                           false, leafSizeArg.getValue(), skipArg.getValue(),
                                                           readNormalsArg.getValue(), eArg.getValue()));
      }
      else if (featuresArg.getValue() == "spin")
      {
        extractedFeatures = DS::Matrix(1, 153);
        fl = FileLoader::Ptr(
            new FFNLoader<PointNormal, PointNormal, SpinImageEstimation<PointNormal, PointNormal, Histogram<153> >,
                Histogram<153> >(extractedFeatures, rArg.getValue(), normalEstArg.getValue(), kArg.getValue(),
                                 leafSizeArg.isSet(), leafSizeArg.getValue(), skipArg.getValue(),
                                 readNormalsArg.getValue(), false, eArg.getValue()));
      }
      else if (featuresArg.getValue() == "pfh")
      {
        extractedFeatures = DS::Matrix(1, 125);
        fl = FileLoader::Ptr(
            new FFNLoader<PointNormal, PointNormal, PFHEstimation<PointNormal, PointNormal, PFHSignature125>,
                PFHSignature125>(extractedFeatures, rArg.getValue(), normalEstArg.getValue(), kArg.getValue(),
                                 leafSizeArg.isSet(), leafSizeArg.getValue(), skipArg.getValue(),
                                 readNormalsArg.getValue(), true, eArg.getValue()));
      }
    }
    else //TODO this is the case the files do not contain any normals
    {
          if (featuresArg.getValue() == "vfh")
          {
            extractedFeatures = DS::Matrix(1, 308);
            fl = FileLoader::Ptr(
                new FFNLoader<PointXYZ, Normal, VFHEstimation<PointXYZ, Normal, VFHSignature308>,
                    VFHSignature308>(extractedFeatures, rArg.getValue(), normalEstArg.getValue(), kArg.getValue(),
                                     leafSizeArg.isSet(), leafSizeArg.getValue(), skipArg.getValue(),
                                     readNormalsArg.getValue(), false, eArg.getValue()));

          }
          else if (featuresArg.getValue() == "fpfh")
          {
            extractedFeatures = DS::Matrix(1, 33);
            fl = FileLoader::Ptr(
                new FFNLoader<PointXYZ, Normal, FPFHEstimation<PointXYZ, Normal, FPFHSignature33>,
                    FPFHSignature33>(extractedFeatures, rArg.getValue(), normalEstArg.getValue(), kArg.getValue(),
                                     leafSizeArg.isSet(), leafSizeArg.getValue(), skipArg.getValue(),
                                     readNormalsArg.getValue(), true, eArg.getValue()));
          }
          else if (featuresArg.getValue() == "gshot")
          {
            extractedFeatures = DS::Matrix(1, 352);
            fl = FileLoader::Ptr(
                new GlobalSHOTLoader<PointXYZ, Normal>(extractedFeatures, normalEstArg.getValue(), rArg.getValue(),
                                                               false, leafSizeArg.getValue(), skipArg.getValue(),
                                                               readNormalsArg.getValue(), eArg.getValue()));
          }
          else if (featuresArg.getValue() == "spin")
          {
            extractedFeatures = DS::Matrix(1, 153);
            fl = FileLoader::Ptr(
                new FFNLoader<PointXYZ, Normal, SpinImageEstimation<PointXYZ, Normal, Histogram<153> >,
                    Histogram<153> >(extractedFeatures, rArg.getValue(), normalEstArg.getValue(), kArg.getValue(),
                                     leafSizeArg.isSet(), leafSizeArg.getValue(), skipArg.getValue(),
                                     readNormalsArg.getValue(), false, eArg.getValue()));
          }
          else if (featuresArg.getValue() == "pfh")
          {
            extractedFeatures = DS::Matrix(1, 125);
            fl = FileLoader::Ptr(
                new FFNLoader<PointXYZ, Normal, PFHEstimation<PointXYZ, Normal, PFHSignature125>,
                    PFHSignature125>(extractedFeatures, rArg.getValue(), normalEstArg.getValue(), kArg.getValue(),
                                     leafSizeArg.isSet(), leafSizeArg.getValue(), skipArg.getValue(),
                                     readNormalsArg.getValue(), true, eArg.getValue()));
          }
    }

    HierarchicalPCDLoader loader(fl, foldersProvideLabelsArg.getValue());

    vector<int> labels;
    path basePath = path(baseDirArg.getValue());
    //TODO: check if base path exist and is dir
    loader.load(basePath);

    ds.setFeatureMatrix(extractedFeatures, nameArg.getValue());

    if (scaleArg.isSet())
    {
      for (int i = 0; i < extractedFeatures.cols(); i++)
      {
        double min = extractedFeatures.col(i).minCoeff();
        for (int j = 0; j < extractedFeatures.rows(); j++)
        {
          extractedFeatures(j, i) -= min;
        }
        double max = extractedFeatures.col(i).maxCoeff();
        if (max != 0.0)
          extractedFeatures.col(i) /= max;
      }
    }
    cout << "Estimating features took on average: " << fl->getMsPerInputPointFeatureEstimationOnly()
        << " seconds per point in the downsampled cloud" << endl;

  }
  else if (featuresArg.getValue() == "labels")
  {
    LabelLoader *ll = new LabelLoader(skipArg.getValue(), eArg.getValue());
    FileLoader::Ptr fl(ll);
    HierarchicalPCDLoader loader(fl, foldersProvideLabelsArg.getValue());
    path basePath = path(baseDirArg.getValue());
    //TODO: check if base path exist and is dir
    loader.load(basePath);
    unsigned int longest = 0;
    vector<vector<int> > labels = ll->getLabels();
    for (unsigned int i = 0; i < labels.size(); i++)
    {
      unsigned int depth = labels.at(i).size();
      if (depth > longest)
        longest = depth;
    }

    DS::Matrix lMatrix(labels.size(), longest);
    for (unsigned int i = 0; i < labels.size(); i++)
    {
      for (unsigned int j = 0; j < longest; j++)
      {
        if (j >= labels.at(i).size())
        {
          lMatrix(i, j) = -1.0;
        }
        else
        {
          lMatrix(i, j) = (double)labels.at(i).at(j);
        }
      }
    }

    ds.setFeatureMatrix(lMatrix, nameArg.getValue());
  }
  else if (featuresArg.getValue() == "names")
  {
    LabelNameLoader *ll = new LabelNameLoader();
    FileLoader::Ptr fl(ll);
    HierarchicalPCDLoader loader(fl, foldersProvideLabelsArg.getValue());
    path basePath = path(baseDirArg.getValue());
    //TODO: check if base path exist and is dir
    loader.load(basePath);
    vector<string> names = ll->getLabels();

    if (!outArg.isSet())
    {
      for (vector<string>::iterator iter = names.begin(); iter != names.end(); iter++)
      {
        cout << *iter << endl;
      }
    }
    else
    {
      fstream out;
      out.open(outArg.getValue().c_str(), ios_base::out);
      for (vector<string>::iterator iter = names.begin(); iter != names.end(); iter++)
      {
        out << *iter << endl;
      }
      out.close();
    }
  }
  else
  {
    cerr << "Unsupported feature type" << endl;
    exit(-1);
  }
  if (!outArg.isSet())
  {
    cout << ds << flush;
  }
  else
  {
    fstream out;
    out.open(outArg.getValue().c_str(), ios_base::out);
    out << ds << flush;
  }
  return 0;
}
