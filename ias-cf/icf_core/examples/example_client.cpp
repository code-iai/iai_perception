/*
 * Copyright (c) 2012,
 * Zoltan-Csaba Marton <marton@cs.tum.edu>,
 * Ferenc Balint-Benczedi <balintb.ferenc@gmail.com>,
 * Florian Seidel <seidel.florian@gmail.com
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
#include <ros/ros.h>
#include <boost/thread.hpp>
#include <icf_core/client/Client.h>
#include <icf_core/base/EvaluationResult.hpp>
#include <icf_core/base/ConfusionMatrix.hpp>
#include <icf_core/base/ClassificationResult.hpp>

using namespace icf;

bool g_stop = false;
bool g_hasStopped;

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

int main(int argc, char** argv)
{
  // init node
  std::string manager_name = "ias_classifier_manager";
  ros::init(argc, argv, manager_name);
  ros::NodeHandle n("~");

  // init manager in a different thread (or start it up separately)
  ClassifierManager g_manager(n);
  startManagerThread();
  sleep(1);

  try
  {
    // load datasets "read only" style into matrices with elements of type double (could use DS as a type)
    // NOTE: will use data from the same file using some tricks as an example
    DS ds("data/test.h5", false, true);

    // in case the features and labels are not named /x and /y in the DS as required by ICF, they can be
    // renamed, aliased, or uploaded by specifying correct name (see examples for all three cases below)
    if (!ds.contains("/x"))
      ds.renameFeatureMatrix("/train", "/x"); // rename, permanent if flushed to disk (see constructor parameters)
    if (!ds.contains("/y"))
      ds.alias("/train_labels", "/y"); // alias; "/train_labels" can now also be accessed as "/y"

    // print out data
    bool print_training_data = false;
    if (print_training_data)
    {
      DS::MatrixPtr features = ds.getFeatureMatrix("/x");
      std::cout << "train features: " << features->rows() << "x" << features->cols() << std::endl;
      std::cout << (*features) << std::endl;
      DS::MatrixPtr labels = ds.getFeatureMatrix("/y");
      std::cout << "train labels: " << labels->rows() << "x" << labels->cols() << std::endl;
      std::cout << (*labels) << std::endl;
    }
    bool print_testing_data = false;
    if (print_testing_data)
    {
      DS::MatrixPtr features = ds.getFeatureMatrix("/test");
      std::cout << "test features: " << features->rows() << "x" << features->cols() << std::endl;
      std::cout << (*features) << std::endl;
      DS::MatrixPtr labels = ds.getFeatureMatrix("/test_labels");
      std::cout << "test labels: " << labels->rows() << "x" << labels->cols() << std::endl;
      std::cout << (*labels) << std::endl;
    }

    // upload training data and ground truth with a string as identifier
    ServerSideRepo data_store(n, manager_name);
    data_store.uploadData(ds, "train");
    data_store.uploadData(ds, "test", "/test", "/test_labels"); // uploading different features and labels for testing than /x and /y

    // start client for a classifier
    ClassifierClient client(n, manager_name, "knn", "-m L2 -k 1");
    //ClassifierClient client(n, manager_name, "svm", "-a 2 -t 2 -v 5 -5:2:15 -15:2:3");
    //ClassifierClient client(n, manager_name, "boost", "--weak_count 2000 --max_depth 1 -t 1");

    // assign uploaded data to the classifier and train it
    client.assignData("train", icf::Train); // training data
    client.assignData("test", icf::Eval); // OPTIONAL: evaluation data, must have labels!
    client.assignData("test", icf::Classify); // testing data (here same as the evaluation data)
    client.train();
    // it is possible to save the trained classifier to a file and load later or with a different client
    // if saving is done after evaluation, the confusion matrix is saved/loaded as well:
    // client.save("path/to.file");
    // client.load("path/to.file");

    // evaluate classifier
    EvaluationResult result = client.evaluate();
    bool print_complete_response = false;
    if (print_complete_response)
    {
      std::cout << "Evaluation result" << std::endl;
      std::cout << result << std::endl;
    }
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << result.getConfusionMatrix()->getCM() << std::endl;
    std::cout << "Error rate = " << result.getErrorRate() << std::endl;

    // classify data and check result for each feature point
    ClassificationResult classificationResult = client.classify();
    DataSet<double>::MatrixPtr features = ds.getX();
    DataSet<double>::MatrixPtr labels = ds.getY(); // OPTIONAL: compare to expected results manually
    for (int i = 0; i < features->rows(); i++)
      std::cout << "Expected " << (*labels)(i, 0) << " and got " << classificationResult.results->at(i)
          << " for the data point " << features->row(i) << std::endl;
  }
  catch (ICFException& e)
  {
    std::cerr << boost::diagnostic_information(e);
    std::vector<service_unavailable_error>* err = boost::get_error_info<service_unavailable_collection>(e);
    if (err != NULL)
    {
      for (std::vector<service_unavailable_error>::iterator iter = err->begin(); iter != err->end(); iter++)
        std::cerr << "Error: " << iter->value() << std::endl;
    }
    else
      std::cerr << "No service availability related errors" << std::endl;
  }

  // stop manager
  g_stop = true;
  ros::Rate r(10);
  while (!g_hasStopped)
  {
    r.sleep();
  }
  return 0;
}

