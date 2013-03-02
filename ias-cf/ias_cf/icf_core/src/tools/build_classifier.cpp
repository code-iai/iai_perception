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
#include <tclap/CmdLine.h>
#include <icf_core/client/Client.h>
#include <icf_core/base/EvaluationResult.hpp>
#include <icf_core/base/ConfusionMatrix.hpp>
#include <icf_core/base/ClassificationResult.hpp>

using namespace icf;
using namespace TCLAP;

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
  //Command line parser
  CmdLine cmdLine("build_classifier", ' ', "dev");
  ValueArg<std::string> cArg("c", "classifier", "the used classifier type", true, "c", "string");
  ValueArg<std::string> tArg("t", "train", "the HDF5 file for training", true, "t", "string");
  ValueArg<std::string> eArg("e", "eval", "the HDF5 file for evaluation (default: same as -t)", false, "", "string");
  ValueArg<std::string> nArg("n", "name", "a name for the dataset", false, "build_classifier", "string");
  ValueArg<std::string> fArg("f", "features", "name of the matrix containing the features (default: /x), use with -l", false, "/x", "string");
  ValueArg<std::string> lArg("l", "labels", "name of the vector containing the labels (default: /y), use with -f", false, "/y", "string");
  ValueArg<std::string> oArg("o", "output", "base file name for the saved model and confusion matrix", true, "o", "string");
  ValueArg<std::string> mArg("m", "manager", "name of the classification service manager node (default: ias_classifier_manager)", false, "ias_classifier_manager", "string");
  cmdLine.add(mArg);
  cmdLine.add(oArg);
  cmdLine.add(lArg);
  cmdLine.add(fArg);
  cmdLine.add(nArg);
  cmdLine.add(eArg);
  cmdLine.add(tArg);
  cmdLine.add(cArg);
  cmdLine.parse(argc, argv);

  // init evaluation file name
  std::string eval_file = eArg.getValue();
  if (eval_file == "")
    eval_file = tArg.getValue();

  // init classifer params: TODO use args --
  std::string params = "";
  if (cArg.getValue() == "knn")
    params = "-m L2 -k 1";
  else if (cArg.getValue() == "svm")
    params = "-a 2 -t 2 -v 5 -5:2:15 -15:2:3";
  else if (cArg.getValue() == "boost")
    params = "--weak_count 2000 --max_depth 1 -t 1";

  // init node
  std::string manager_name = mArg.getValue();
  ros::init(argc, argv, manager_name);
  ros::NodeHandle n("~");

  // init manager in a different thread (or start it up separately)
  ClassifierManager g_manager(n);
  startManagerThread();
  sleep(1);

  try
  {
    // load datasets "read only" style into matrices with elements of type double (could use DS as a type)
    DS ds_train(tArg.getValue(), false, true);
    DS ds_eval(eval_file, false, true);

    // print out data
    std::cout << "training data: " << tArg.getValue() << std::endl;
    std::cout << "evaluation data: " << eval_file << std::endl;
    std::cout << "features: " << fArg.getValue() << std::endl;
    std::cout << "labels: " << lArg.getValue() << std::endl;
    bool print_training_data = true;
    if (print_training_data)
    {
      DS::MatrixPtr features = ds_train.getFeatureMatrix(fArg.getValue());
      std::cout << "training features: " << features->rows() << "x" << features->cols() << std::endl;
      //std::cout << (*features) << std::endl;
      DS::MatrixPtr labels = ds_train.getFeatureMatrix(lArg.getValue());
      std::cout << "training labels: " << labels->rows() << "x" << labels->cols() << std::endl;
      //std::cout << (*labels) << std::endl;
    }
    bool print_testing_data = true;
    if (print_testing_data)
    {
      DS::MatrixPtr features = ds_eval.getFeatureMatrix(fArg.getValue());
      std::cout << "evaluation features: " << features->rows() << "x" << features->cols() << std::endl;
      //std::cout << (*features) << std::endl;
      DS::MatrixPtr labels = ds_eval.getFeatureMatrix(lArg.getValue());
      std::cout << "evaluation labels: " << labels->rows() << "x" << labels->cols() << std::endl;
      //std::cout << (*labels) << std::endl;
    }

    // upload training data and ground truth with a string as identifier
    ServerSideRepo data_store(n, manager_name);
    data_store.uploadData(ds_train, nArg.getValue()+"_train", fArg.getValue(), lArg.getValue());
    data_store.uploadData(ds_eval, nArg.getValue()+"_eval", fArg.getValue(), lArg.getValue());

    // start client for a classifier
    ClassifierClient client(n, manager_name, cArg.getValue(), params);

    // assign uploaded data to the classifier and train it
    client.assignData(nArg.getValue()+"_train", icf::Train);
    client.assignData(nArg.getValue()+"_eval", icf::Eval);
    std::cout << "building " << cArg.getValue() << ": " << params << std::endl;
    client.train();
    std::cout << "building done, starting evaluation" << std::endl;

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

    // save the trained classifier and the confusion matrix
    client.save(oArg.getValue());
    std::cout << "saved to " << oArg.getValue() << ".*" << std::endl;

    // TODO: test
    ClassifierClient client_test(n, manager_name, cArg.getValue(), params);
    client_test.load(oArg.getValue()); // TODO: better error messaging if missing
    client_test.assignData(nArg.getValue()+"_eval", icf::Classify);

    // classify data and check result for each feature point
    ClassificationResult classificationResult = client_test.classify();
    // TODO: get confusion matrix to get accuracy weighted result!
    DataSet<double>::MatrixPtr labels = ds_eval.getFeatureMatrix(lArg.getValue());
    double success = 0;
    for (int i = 0; i < labels->rows(); i++)
    {
      double expected = (*labels)(i, 0);
      double got = classificationResult.results->at(i);
      if (expected == got)
        success++;
      else
//      if (print_complete_response)
        std::cout << i << ": Expected " << expected << " and got " << got
                  << " with confidence " << classificationResult.confidenceFor(i, got)
                  << " vs " << classificationResult.confidenceFor(i, expected) << std::endl;
    }
    std::cout << "Expected success rate: " << (1-result.getErrorRate()) << " and got " << success/labels->rows() << std::endl;
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

