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

//icf_core
#include <icf_core/base/ConfusionMatrix.hpp>
#include <icf_core/base/EvaluationResult.hpp>
#include <icf_core/base/ClassificationResult.hpp>

//boost
#include <boost/algorithm/string.hpp>

//eigne
#include <Eigen/Core>

//stl
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <ctime>

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

namespace icf
{
/**
 * \brief lightweight abstract class defining basic model for a classifier
 */
class Classifier
{
public:

  /** \brief empty constructor */
  virtual ~Classifier()
  {
  }

  /**
   *\brief method for adding training data
   *\param data to be interpreted and added
   */
  virtual void addTrainingData(std::string) = 0;

  /**
   *\brief method for adding data
   *\param data to be interpreted and added
   */
  virtual void addData(std::string) = 0;

  /**
   *\brief method for adding evaluation data(use this when you data has ground truth)
   *\param data to be interpreted and added
   */
  virtual void addEvaluationData(std::string)=0;

  /**
   *\brief method for building the model for classifiers
   *\return 0 for success other for error
   *\param parameters of the classifier unrolled in a string
   */
  virtual int buildModel(std::string) = 0;

  /**
   *\brief classifying/adding test data use this function when you don't have ground truth
   *\return result for given test data
   *\param data to be tested
   */
  virtual std::string classify()=0;

  /** \brief basically the same as classify, use it when you have ground truth so confusion matrix can be created*/
  virtual std::string evaluate()= 0;

  /** \brief save the model. Throw a std::string if something goes wrong.*/
  virtual void save(std::string filename)=0;

  /** \brief load the model. Throw a std::string if something goes wrong.*/
  virtual void load(std::string filename)=0;

  /** \brief "start" timer */
  virtual void startTiming()
  {
    this->timingStart = clock();
  }

  /** \brief "stop" timer */
  virtual void stopTiming()
  {
    this->secLastOp = (clock() - this->timingStart) / (double)CLOCKS_PER_SEC;
  }

  /**
   * \brief next release
   */
  virtual std::string executeService(std::string service, std::string params)
  {
    if (service == "get_confidence_eval")
    {
      DS::MatrixPtr conf = evalResult->getConfidence();
      DS ds("get_confidence_eval", true, 1024, false);
      ds.setFeatureMatrix(*conf, "confidence");
      std::stringstream ss;
      ss << ds;
      return ss.str();
    }
    else if (service == "get_confidence_classify")
    {
      DS ds("get_confidence_classify", true, 1024, false);
      ds.setFeatureMatrix(*(classificationResult->confidences), "confidence");
      std::stringstream ss;
      ss << ds;
      return ss.str();
    }
    else if (service == "time_op")
    {
      std::stringstream ss;
      ss << this->secLastOp;
      return ss.str();
    }
    else if (service == "per_instance_classification_time")
    {
      std::stringstream ss;
      ss << this->secPerInstanceClassification;
      return ss.str();
    }
    else if (service == "help")
    {
      return "get_confidence_eval,get_confidence_classify,time_op,per_instance_classification_time";
    }
    return "Service unknown";
  }

  /**
   * \brief set dataset for training, classification or evaluation
   * \param[1] dataset
   * \param[2] slot--type of data
   */
  void setDataset(DS::Ptr ds, std::string slot)
  {
    if (slot == "train")
    {
      this->trainDS = ds;
    }
    else if (slot == "eval")
    {
      this->evalDS = ds;
    }
    else if (slot == "classify")
    {
      this->classifyDS = ds;
    }
    else
    {
      throw std::string("Invalid slot");
    }
  }

  /** \brief type of the classifier*/
  std::string type;
  /** \brief parameters of the classifier*/
  std::string params;
  /** \brief confusion matrix **/
  boost::shared_ptr<ConfusionMatrix> cm;
  /** \brief store evaluation results here*/
  boost::shared_ptr<EvaluationResult> evalResult;
  /** \brief store classification results here*/
  boost::shared_ptr<ClassificationResult> classificationResult;

  /** \brief datasets, for train eval classify **/
  DS::Ptr trainDS;
  DS::Ptr evalDS;
  DS::Ptr classifyDS;

  /** \brief timing members*/
  double secPerInstanceClassification;
  double secLastOp;
  clock_t timingStart;

  /* ---will see what happens to these guys, they should be obsolete
   Eigen::MatrixXf confusionMatrix;
   Eigen::MatrixXf result_matrix; //TODO find a more descriptive name for this:)
   Eigen::VectorXi tpr; //true positive rate
   */
};
}
#endif /* CLASSIFIER_H_ */
