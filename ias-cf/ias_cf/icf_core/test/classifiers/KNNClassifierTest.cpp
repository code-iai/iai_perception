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

#include <icf_dataset/DataSet.hpp>
#include <icf_core/base/ICFExceptionErrors.h>
#include <icf_core/classifiers/KNNClassifier.h>
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

using namespace std;

TEST(KNNClassifierTest,testImpl1)
{

  flann::Matrix<double> train(new double[6], 3, 2);

  train[0][0] = 0.5;
  train[0][1] = 0.5;
  train[1][0] = 0.2;
  train[1][1] = 0.2;
  train[2][0] = 0.7;
  train[2][1] = 0.7;

  boost::shared_ptr<std::vector<int> > labels(new std::vector<int>());
  labels->push_back(0);
  labels->push_back(1);
  labels->push_back(1);

  boost::shared_ptr<flann::Index<flann::L2<double> > > index(
      new flann::Index<flann::L2<double> >(train, flann::KDTreeSingleIndexParams()));
  index->buildIndex();

  icf::KNNImpl<flann::L2<double> > * classifier = new icf::KNNImpl<flann::L2<double> >(3, index, labels, false);

  flann::Matrix<double> test(new double[4], 2, 2);
  test[0][0] = 0.2;
  test[0][1] = 0.2;
  test[1][0] = 0.7;
  test[1][1] = 0.7;

  icf::KNNImpl<flann::L2<double> >::KNNResult predictions = classifier->classify(test);

  EXPECT_TRUE((size_t) 2== predictions.get<0> ()->size());
  EXPECT_TRUE(1== predictions.get<0> ()->at(0));
  EXPECT_TRUE(1== predictions.get<0> ()->at(1));

}

TEST(KNNClassifierTest,testWrapperInterfaceForEvaluation)
{
  icf::DS::Ptr trainDS(new icf::DS("data/tmpinmem", true, true, 1024, false));
  icf::DS::Matrix train(3, 2);

  train(0, 0) = 0.5;
  train(0, 1) = 0.5;
  train(1, 0) = 0.2;
  train(1, 1) = 0.2;
  train(2, 0) = 0.7;
  train(2, 1) = 0.7;

  icf::DS::Matrix labels(3, 1);
  labels(0, 0) = 0;
  labels(1, 0) = 1;
  labels(2, 0) = 1;

  trainDS->setFeatureMatrix(labels, "/y");
  trainDS->setFeatureMatrix(train, "/x");

  icf::DS::Ptr evalDS(new icf::DS("data/tmpinmem2", true, true, 1024, false));
  icf::DS::Matrix eval(2, 2);
  eval(0, 0) = 0.2;
  eval(0, 1) = 0.2;
  eval(1, 0) = 0.7;
  eval(1, 1) = 0.7;
  evalDS->setFeatureMatrix(eval, "/x");

  icf::DS::Matrix evalLabels(2, 1);
  evalLabels(0, 0) = 0;
  evalLabels(1, 0) = 1;

  evalDS->setFeatureMatrix(evalLabels, "/y");

  string params = "-k 3 -m L2";
  icf::KNNClassifier<double> classifier(params);

  classifier.setDataset(trainDS, "train");
  classifier.setDataset(evalDS, "eval");

  classifier.buildModel("");

  icf::EvaluationResult result(classifier.evaluate());

  EXPECT_TRUE(result.getErrorRate()==0.5);

}

TEST(KNNClassifierTest,testWrapperInterfaceForClassification)
{
  icf::DS::Ptr trainDS(new icf::DS("data/tmpinmem", true, true, 1024, false));
  icf::DS::Matrix train(3, 2);

  train(0, 0) = 0.5;
  train(0, 1) = 0.5;
  train(1, 0) = 0.2;
  train(1, 1) = 0.2;
  train(2, 0) = 0.7;
  train(2, 1) = 0.7;

  icf::DS::Matrix labels(3, 1);
  labels(0, 0) = 0;
  labels(1, 0) = 1;
  labels(2, 0) = 1;

  trainDS->setFeatureMatrix(labels, "/y");
  trainDS->setFeatureMatrix(train, "/x");

  icf::DS::Ptr testDS(new icf::DS("data/tmpinmem2", true, true, 1024, false));
  icf::DS::Matrix test(2, 2);
  test(0, 0) = 0.2;
  test(0, 1) = 0.2;
  test(1, 0) = 0.7;
  test(1, 1) = 0.7;
  testDS->setFeatureMatrix(test, "/x");

  string params = "-k 3 -m L2";
  icf::KNNClassifier<double> classifier(params);

  classifier.setDataset(trainDS, "train");
  classifier.setDataset(testDS, "classify");

  classifier.buildModel("");

  icf::ClassificationResult cr(classifier.classify());

  EXPECT_TRUE(cr.results->at(0)==1);
  EXPECT_TRUE(cr.results->at(1)==1);

}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
