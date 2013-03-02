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
#include <icf_core/classifiers/BoostOAA.h>
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <cstdio>

using namespace std;

TEST(BoostOAAClassifierTest,testWrapperInterfaceDSFromFile)
{

  icf::DS ds("data/test.h5", false);

  icf::DS::MatrixPtr train = ds.getFeatureMatrix("/train");
  icf::DS::MatrixPtr train_labels = ds.getFeatureMatrix("/train_labels");

  icf::DS::MatrixPtr test = ds.getFeatureMatrix("/test");
  icf::DS::MatrixPtr test_labels = ds.getFeatureMatrix("/test_labels");

  icf::DS::Ptr train_ds(new icf::DS("train_ds", true, true, 1024, false));
  icf::DS::Ptr test_ds(new icf::DS("test_ds", true, true, 1024, false));

  train_ds->setFeatureMatrix(train, "/x");
  train_ds->setFeatureMatrix(train_labels, "/y");

  test_ds->setFeatureMatrix(test, "/x");
  test_ds->setFeatureMatrix(test_labels, "/y");

  string params = "-t 0 -s --weak_count 5 --max_depth 1";

  icf::BoostOAA classifier(params);

  classifier.setDataset(train_ds, "train");
  classifier.setDataset(test_ds, "eval");

  classifier.buildModel("");

  icf::EvaluationResult result(classifier.evaluate());

  //EXPECT_EQ(result.getErrorRate(),0.7);

}

TEST(BoostOAAClassifierTest,testWrapperInterfaceDSFromFile2)
{

  icf::DS ds("data/test.h5", false);

  icf::DS::MatrixPtr train = ds.getFeatureMatrix("/train");
  icf::DS::MatrixPtr train_labels = ds.getFeatureMatrix("/train_labels");

  icf::DS::MatrixPtr test = ds.getFeatureMatrix("/test");
  icf::DS::MatrixPtr test_labels = ds.getFeatureMatrix("/test_labels");

  icf::DS::Ptr train_ds(new icf::DS("train_ds", true, true, 1024, false));
  icf::DS::Ptr test_ds(new icf::DS("test_ds", true, true, 1024, false));

  train_ds->setFeatureMatrix(train, "/x");
  train_ds->setFeatureMatrix(train_labels, "/y");

  test_ds->setFeatureMatrix(test, "/x");
  test_ds->setFeatureMatrix(test_labels, "/y");

  string params = "-t 1 -s --weak_count 2000 --max_depth 1";

  icf::BoostOAA classifier(params);

  classifier.setDataset(train_ds, "train");
  classifier.setDataset(test_ds, "classify");

  classifier.buildModel("");

  icf::ClassificationResult result(classifier.classify());

  /*
   EXPECT_EQ(result.results->at(0),0);
   EXPECT_EQ(result.results->at(1),0);
   EXPECT_EQ(result.results->at(2),1);
   EXPECT_EQ(result.results->at(3),0);
   EXPECT_EQ(result.results->at(4),1);
   EXPECT_EQ(result.results->at(5),1);
   EXPECT_EQ(result.results->at(6),0);
   EXPECT_EQ(result.results->at(7),0);
   EXPECT_EQ(result.results->at(8),1);
   EXPECT_EQ(result.results->at(9),0);
   */

}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
