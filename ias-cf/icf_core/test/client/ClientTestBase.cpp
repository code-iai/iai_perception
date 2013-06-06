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
#include <iostream>
#include <icf_core/client/Client.h>
#include <gtest/gtest.h>
#include <ros/ros.h>
#include <boost/thread.hpp>
#include <icf_core/base/EvaluationResult.hpp>
#include <icf_core/base/ConfusionMatrix.hpp>
#include <icf_core/base/ClassificationResult.hpp>

using namespace icf;

#define DESKTOP_TEST

bool g_stop = false;
bool g_hasStopped;

ros::NodeHandle * g_pNodeHandle;
std::string g_managerName;

ClassifierManager * g_manager;

#define TRAIN_DS 0
#define EVAL_DS 1

boost::tuple<DS, DS> createLinearilySeparableDatasetWithAxisAlignedSplit(const std::string& featureName,
                                                                         const std::string& labelName, int size = 50)
{
  DS trainingData;

  DS::Matrix trainFeatures(50, 2);
  DS::Matrix trainLabels(50, 2);

  int i;
  for (i = 0; i < size / 2; i++)
  {
    trainFeatures(i, 0) = rand() / (2.0 * RAND_MAX) - 8.01;
    trainFeatures(i, 1) = (double)rand() / RAND_MAX - 500.0;
    trainLabels(i, 0) = 1;
  }
  for (; i < size; i++)
  {
    trainFeatures(i, 0) = 8.6 + rand() / (2.0 * RAND_MAX);
    trainFeatures(i, 1) = (double)rand() / (2.0 * RAND_MAX) + 500.0;
    trainLabels(i, 0) = 2;
  }
  //std::cout << trainFeatures << std::endl;
  //std::cout << trainLabels << std::endl;
  trainingData.setFeatureMatrix(trainFeatures, featureName);
  trainingData.setFeatureMatrix(trainLabels, labelName);
  return boost::make_tuple<DS, DS>(trainingData, trainingData);

}

boost::tuple<DS, DS> createLinearilySeparableDataset(const std::string& featureName, const std::string& labelName)
{
  DS trainingData;

  DS::Matrix trainFeatures(12, 2);
  DS::Matrix trainLabels(12, 1);

  //2 classes one below y=-x +1, one above
  trainFeatures(0, 0) = 0.7;
  trainFeatures(0, 1) = 0.7;
  trainFeatures(1, 0) = 0.6;
  trainFeatures(1, 1) = 0.8;
  trainFeatures(2, 0) = 0.55;
  trainFeatures(2, 1) = 0.67;
  trainFeatures(3, 0) = 0.67;
  trainFeatures(3, 1) = 0.55;
  trainFeatures(4, 0) = 0.3;
  trainFeatures(4, 1) = 0.9;
  trainFeatures(5, 0) = 0.99;
  trainFeatures(5, 1) = 0.1;
  trainFeatures(6, 0) = 0.1;
  trainFeatures(6, 1) = 0.0;
  trainFeatures(7, 0) = 0.2;
  trainFeatures(7, 1) = 0.1;
  trainFeatures(8, 0) = 0.2;
  trainFeatures(8, 1) = 0.7;
  trainFeatures(9, 0) = 0.9;
  trainFeatures(9, 1) = 0.01;
  trainFeatures(10, 0) = 0.1;
  trainFeatures(10, 1) = 0.11;
  trainFeatures(11, 0) = 0.2;
  trainFeatures(11, 1) = 0.25;

  trainLabels(0) = 1;
  trainLabels(1) = 1;
  trainLabels(2) = 1;
  trainLabels(3) = 1;
  trainLabels(4) = 1;
  trainLabels(5) = 1;
  trainLabels(6) = 2;
  trainLabels(7) = 2;
  trainLabels(8) = 2;
  trainLabels(9) = 2;
  trainLabels(10) = 2;
  trainLabels(11) = 2;

  trainingData.setFeatureMatrix(trainFeatures, featureName);
  trainingData.setFeatureMatrix(trainLabels, labelName);

  return boost::make_tuple<DS, DS>(trainingData, trainingData);
}

TEST(ClientTestBase, transmissionOfErrorMessages)
{
  ServerSideRepo dr(*g_pNodeHandle, g_managerName);
  //upload training data and ground truth
  bool errorThrown = false;
  try
  {
    dr.uploadData("/path/does/not/exist", "test");
  }
  catch (boost::exception& e)
  {
    //std::cout<<boost::diagnostic_information(e)<<std::endl;
    //std::cout<<dr.getLastResponse()<<std::endl;
    //EXPECT_EQ(dr.getLastResponse(),"Error opening file");
    errorThrown = true;
  }
  EXPECT_TRUE(errorThrown);
}

TEST(ClientTestBase, testServiceNotAvailable)
{
  DS test1;
  DS test2;
  bool exceptionsThrown = false;
  try
  {
    ServerSideRepo dr(*g_pNodeHandle, "wrong_manager_name");
  }
  catch (boost::exception& e)
  {
    if (boost::get_error_info<service_unavailable_collection>(e) == NULL)
    {
      EXPECT_TRUE(false);
    }
    else
    {
      EXPECT_EQ(4, boost::get_error_info<service_unavailable_collection >(e)->size());
      std::vector<service_unavailable_error>* err = boost::get_error_info<service_unavailable_collection>(e);
      EXPECT_EQ("/wrong_manager_name/add_dataset", err->at(0).value());
      EXPECT_EQ("/wrong_manager_name/set_dataset", err->at(1).value());
      EXPECT_EQ("/wrong_manager_name/read_dataset", err->at(2).value());
      EXPECT_EQ("/wrong_manager_name/remove_dataset", err->at(3).value());
      exceptionsThrown = true;
    }
  }
  EXPECT_TRUE(exceptionsThrown);
}

TEST(ClientTestBase, testDataUpload)
{
  try
  {
    ServerSideRepo dr(*g_pNodeHandle, g_managerName);
    boost::tuple<DS, DS> datasets = createLinearilySeparableDataset("/x", "/y");

    //upload training data and ground truth
    dr.uploadData(datasets.get<TRAIN_DS>(), "train1");
    EXPECT_TRUE(g_manager->datasets.find("train1")!=g_manager->datasets.end());
    EXPECT_EQ(*datasets.get<TRAIN_DS>().getFeatureMatrix("/x"), *g_manager->datasets["train1"]->getFeatureMatrix("/x"));
    EXPECT_EQ(*datasets.get<TRAIN_DS>().getFeatureMatrix("/y"), *g_manager->datasets["train1"]->getFeatureMatrix("/y"));

    boost::tuple<DS, DS> datasets2 = createLinearilySeparableDataset("/xx", "/yy");

    //upload and rename xx -> x and yy -> y, the standard names for data and ground truth
    dr.uploadData(datasets2.get<TRAIN_DS>(), "train2", "/xx", "/yy");
    EXPECT_TRUE(g_manager->datasets.find("train2")!=g_manager->datasets.end());
    EXPECT_TRUE(g_manager->datasets["train2"]->contains("/y"));
    EXPECT_TRUE(g_manager->datasets["train2"]->contains("/x"));

    //delete the data
    dr.removeData("train1");
    dr.removeData("train2");
    EXPECT_TRUE(g_manager->datasets.find("train1")==g_manager->datasets.end());
    EXPECT_TRUE(g_manager->datasets.find("train2")==g_manager->datasets.end());
  }
  catch (boost::exception& e)
  {
    std::cerr << boost::diagnostic_information(e);
    EXPECT_TRUE(false);
  }
}

void checkClassificationResult(boost::tuple<DS, DS> & datasets, ClassificationResult & classificationResult)
{
  DS::MatrixPtr labels = datasets.get<EVAL_DS>().getY();
  for (int i = 0; i < datasets.get<EVAL_DS>().getX()->rows(); i++)
  {
    EXPECT_EQ((*labels)(i,0), classificationResult.results->at(i));
    //TODO: Check consistency of confidences with classification result
    //EXPECT_TRUE(classificationResult.confidenceFor(i,classificationResult.results->at(i))>=classificationResult.confidenceFor(i,);
  }
}

void testClassifierInterfaceConformance(std::string classifier, std::string params, boost::tuple<DS, DS> datasets,
                                        const std::string& classifierName, bool check_err_rate = true)
{
  //Test data assignment, training and evaluation
  ClassifierClient client(*g_pNodeHandle, g_managerName, classifier, params);
  try
  {
    ServerSideRepo dr(*g_pNodeHandle, g_managerName);
    //upload training data and ground truth
    dr.uploadData(datasets.get<TRAIN_DS>(), "train");
    EXPECT_TRUE(g_manager->datasets.find("train")!=g_manager->datasets.end());
    //TODO: This should be enforced later. => send binary data
    //EXPECT_EQ(*datasets.get<TRAIN_DS>().getFeatureMatrix("/x"),*g_manager->datasets["train"]->getFeatureMatrix("/x"));
    //EXPECT_EQ(*datasets.get<TRAIN_DS>().getFeatureMatrix("/y"),*g_manager->datasets["train"]->getFeatureMatrix("/y"));

    dr.uploadData(datasets.get<EVAL_DS>(), "eval");
    EXPECT_TRUE(g_manager->datasets.find("eval")!=g_manager->datasets.end());
    //TODO: This should be enforced later. => send binary data
    //EXPECT_EQ(*datasets.get<EVAL_DS>().getFeatureMatrix("/x"),*g_manager->datasets["eval"]->getFeatureMatrix("/x"));
    //EXPECT_EQ(*datasets.get<EVAL_DS>().getFeatureMatrix("/y"),*g_manager->datasets["eval"]->getFeatureMatrix("/y"));

    client.assignData("train", icf::Train);
    client.assignData("eval", icf::Eval);
    client.assignData("eval", icf::Classify);
    client.train();

    //Test saving and loading without having evaluated the classifier
    client.save("data/" + classifierName + "loadsavewithouteval");

    ClassifierClient client3(*g_pNodeHandle, g_managerName, classifier, params);
    client3.load("data/" + classifierName + "loadsavewithouteval");
    EXPECT_TRUE(!client3.getConfusionMatrix());
    client3.free();

    EvaluationResult result = client.evaluate();
    //		std::cout << "Evaluation result" << std::endl;
    //		std::cout << result << std::endl;
    //		std::cout << "Confusion Matrix" << std::endl;
    //		std::cout << result.getConfusionMatrix()->getCM();
    //std::cout << "error rate" << result.getErrorRate() << std::endl;
    if (check_err_rate)
    {
      EXPECT_EQ(0.0, result.getErrorRate());
      if (result.getConfusionMatrix().get() != NULL)
      {
        EXPECT_EQ(6, result.getConfusionMatrix()->operator ()(1,1));
        EXPECT_EQ(6, result.getConfusionMatrix()->operator ()(2,2));
        EXPECT_EQ(0.0, result.getConfusionMatrix()->operator ()(1,2));
        EXPECT_EQ(0.0, result.getConfusionMatrix()->operator ()(2,1));
      }
      else
      {
        EXPECT_TRUE(false);
      }
    }
    ClassificationResult classificationResult = client.classify();
    if (check_err_rate)
      checkClassificationResult(datasets, classificationResult);
    client.save("data/" + classifierName);
    //Load saved model
    ClassifierClient client2(*g_pNodeHandle, g_managerName, classifier, params);
    client2.load("data/" + classifierName);
    client2.assignData("eval", icf::Classify);
    //std::cout << "Confusion matrix after save-load" << std::endl;
    //std::cout << client2.getConfusionMatrix()->serialize() << std::endl;
    EXPECT_EQ(result.getConfusionMatrix()->serialize(), client2.getConfusionMatrix()->serialize());
    classificationResult = client2.classify(datasets.get<EVAL_DS>());
    if (check_err_rate)
      checkClassificationResult(datasets, classificationResult);
    //delete the data
    dr.removeData("train");
    dr.removeData("eval");
    EXPECT_TRUE(g_manager->datasets.find("train")==g_manager->datasets.end());
    EXPECT_TRUE(g_manager->datasets.find("eval")==g_manager->datasets.end());

    //delete the classifiers
    client.free();
    client2.free();

    EXPECT_EQ(0, g_manager->classifier_collection.size());

  }
  catch (ICFException& e)
  {
    std::cerr << boost::diagnostic_information(e);
    std::vector<service_unavailable_error>* err = boost::get_error_info<service_unavailable_collection>(e);
    if (err != NULL)
    {
      for (std::vector<service_unavailable_error>::iterator iter = err->begin(); iter != err->end(); iter++)
      {
        std::cerr << "Error: " << iter->value() << std::endl;
      }
    }
    EXPECT_TRUE(false);
    sleep(1);
  }
}

TEST(ClientTestBase, testSVMClassifier)
{
  testClassifierInterfaceConformance("svm", "-t 0 -c 4096", createLinearilySeparableDataset("/x", "/y"), "svm"); //high value for C parameter: Fit training data perfectly, please.
}

TEST(ClientTestBase, testKNNClassifier)
{
  testClassifierInterfaceConformance("knn", "-m L2 -k 1", createLinearilySeparableDataset("/x", "/y"), "knn");
}

TEST(ClientTestBase, testBoostClassifier)
{
  testClassifierInterfaceConformance("boost", "--weak_count 2000 --max_depth 1 -t 1",
                                     createLinearilySeparableDataset("/x", "/y"), "boost", false);
}

void managerThread()
{
  ros::Rate r(10);
  while (!g_stop)
  {
    ros::spinOnce();
    r.sleep();
  }
  g_hasStopped = true;
}

void startManagerThread()
{
  boost::thread thread(&managerThread);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  g_managerName = "ias_classifier_manager";
  ros::init(argc, argv, g_managerName); //name of node
  ros::NodeHandle n("~");
  g_pNodeHandle = &n;
  g_manager = new icf::ClassifierManager(n);
  startManagerThread();
  sleep(1);
  std::cerr << "done sleeping" << std::endl;
  int result = RUN_ALL_TESTS();
  g_stop = true;
  ros::Rate r(10);
  while (!g_hasStopped)
  {
    r.sleep();
  }
  delete g_manager;
  return result;
}
