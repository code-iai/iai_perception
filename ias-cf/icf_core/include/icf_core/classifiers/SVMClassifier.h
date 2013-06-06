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

//iascf
#include <icf_dataset/DataSet.hpp>
#include "icf_core/base/ICFExceptionErrors.h"
#include <icf_core/base/Classifier.hpp>
#include <icf_core/base/EvaluationResult.hpp>
#include <icf_core/base/ClassificationResult.hpp>

//libsvm
#include <libsvm/svm.h>

//stl
#include <string>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <assert.h>
#include <sstream>
#include <cmath>
#include <cfloat>
#include <queue>
#include <map>
#include <ctime>

//boost
#include <boost/regex.hpp>
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>

#ifndef SVMCLASSIFIER_H_
#define SVMCLASSIFIER_H_
namespace icf
{

/**
 * \brief SVM classifier class. Inherits base functionality form abstract Classifier
 */
class SVMClassifier : public Classifier
{

public:

  /*
   * \brief constructor
   * \param parameters unrolled in a string
   */
  SVMClassifier(const std::string& command_line);

  /** \brief destructor...daaaaa */
  virtual ~SVMClassifier();

  /**
   *\brief method for adding training data
   *\param data to be interperted and added; cell separator: space; row seperator: \n
   *\param  each row one example; last column label;
   */
  virtual void addTrainingData(std::string data);

  /**
   *\brief method for adding data
   *\param data to be interperted and added; cell separator: space; row seperator: \n
   */
  virtual void addData(std::string data);

  /**
   * \brief method for adding evaluation data (use this only if you have ground truth)
   * \param data
   */
  virtual void addEvaluationData(std::string data);

  virtual std::string executeService(std::string service, std::string params);

  /**
   *\brief
   */
  virtual int buildModel(std::string data);
  void cvThread(std::queue<std::pair<double, double> >& workQueue, double& bestError, double& best_c, double& best_g,
                boost::mutex& queueMutex, boost::mutex& bestParamsMutex, svm_problem& problem);

  /**
   *\brief classifying/adding test data
   *\return result for given test data
   */
  virtual std::string classify();

  virtual void load(std::string filename);
  virtual void save(std::string filename);

  std::string evaluate();

private:

  std::string evaluationData;
  std::string classificationData;
  std::string trainingData;
  svm_parameter param;
  svm_model* model;
  LabelMap * labelMap;
  svm_problem* problem;

  bool cross_validation;
  int nr_fold;
  double gstart;
  double gstep;
  double gend;
  double cstart;
  double cstep;
  double cend;
  double best_error;
  int nrThreads;
  std::string ds_prefix;

  void parse_command_line(const std::string& paramsString);
  void parse_data(const std::string& data, svm_problem& model, bool containsLabels);
  void parse_data(DataSet<double>& ds, svm_problem& problem, bool containsLabels);
  void parse_data(std::string data);

};

}
#endif /* SVMCLASSIFIER_H_ */
