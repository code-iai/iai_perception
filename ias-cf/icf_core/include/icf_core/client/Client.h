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

#ifndef CLIENT_H_
#define CLIENT_H_

#include <string>
#include <exception>

#include <ros/ros.h>

#include <icf_dataset/DataSet.hpp>
#include <icf_core/base/ICFExceptionErrors.h>
#include <icf_core/base/ClassificationResult.hpp>
#include <icf_core/base/EvaluationResult.hpp>

//services
#include <icf_core/NewClassifier.h>
#include <icf_core/BuildModel.h>
#include <icf_core/TrainData.h>
#include <icf_core/TestData.h>
#include <icf_core/Evaluate.h>
#include <icf_core/GetConfMatrix.h>
#include <icf_core/Classify.h>
#include <icf_core/EvalData.h>
#include <icf_core/Load.h>
#include <icf_core/Save.h>
#include <icf_core/AddDataset.h>
#include <icf_core/RemoveDataset.h>
#include <icf_core/SetDataset.h>
#include <icf_core/ReadDataset.h>

#include <icf_core/service/ClassifierManager.h>

#include <boost/exception/all.hpp>
#include <boost/tuple/tuple.hpp>

namespace icf
{

typedef DS::Matrix Matrix;
typedef DS::MatrixPtr MatrixPtr;

/**
 * Each classifier has three slots to which data can be assigned.
 * "Train" for the training data, "Classify" for the data that should be classified by the
 * trained classifier and "Eval" for the data used for evaluation.
 */
enum DataSetSlot
{
  Train, Classify, Eval
};

/**
 * Abstract base class for all classifiers.
 */
class ClassifierBase
{
protected:
  bool invalidState;
  /**
   * Call this at the beginning of every function to check if the object is in a consistent state
   * and the services are all reachable.
   */
  virtual void checkState() const;

public:
  ClassifierBase();
  virtual ~ClassifierBase();

  /**
   * After data has been assigned to the "Train" slot this method can
   * be called to train the classifier
   */
  virtual void train(const std::string& params = "")=0;

  /**
   * After data has been assigned to the "Evaluate" slot this method can be
   * called to evaluate the classifier
   * @returns the result of the model evaluation
   */
  virtual EvaluationResult evaluate()=0;

  /**
   * After data has been assigned to the "Classify" slot and the classifier has been trained
   * this method can be called to classify this data
   */
  virtual ClassificationResult classify()=0;

  /**
   * After the classifier has been trained this method can be called to save it to disk
   * The path is relative to the location from which the ICF service was started
   */
  virtual void save(const std::string& filename)=0;

  /**
   * After the classifier was saved to disk using the save method it can be
   * loaded using this method. The path is relative to the location from which the ICF service
   * was started.
   */
  virtual void load(const std::string& filename)=0;

  /**
   * Returns the confusion matrix if the classifier has been evaluated on evaluation data.
   * Otherwise NULL
   */
  virtual boost::shared_ptr<ConfusionMatrix> getConfusionMatrix()=0;
};

typedef boost::shared_ptr<ClassifierBase> CBPtr;

/**
 * Each class which uses the ICF service has to implement this interface
 */
class ServiceCallResultHolder
{
private:
  std::string lastResponse;
public:
  ServiceCallResultHolder();
  virtual ~ServiceCallResultHolder();
  /**
   * After having made a service call this will return the last response
   */
  virtual std::string getLastResponse();
  /**
   * Use this method to set the result of the last service call
   */
  virtual void setLastResponse(const std::string& lastResponse);

};

/**
 * Functionality needed for a basic classifier if it is hosted in the ICF service. See ClassifierBase for details
 */
class SimpleClassifier : public ClassifierBase
{
public:
  /**
   * Assign a dataset, uploaded using the ServerSideRepo.
   */
  virtual void assignData(const std::string& name, DataSetSlot type)=0;
  /**
   * The dataset needs to contain one matrix called "/x" holding the data and one matrix called "/y" holding the labels
   */
  virtual EvaluationResult evaluate(DS& dataset, const std::string& features = "", const std::string& gt = "")=0;
  /**
   * The dataset needs to contain one matrix called "/x" holding the data
   */
  virtual ClassificationResult classify(DS& dataset, const std::string& features = "")=0;
  /**
   * The dataset needs to contain one matrix called "/x" holding the data and one matrix called "/y" holding the labels
   */
  virtual void train(const std::string& params, DS& dataset, const std::string& features = "", const std::string& gt =
                         "")=0;
  virtual void train(const std::string& params = "")=0;

  virtual EvaluationResult evaluate()=0;

  virtual ClassificationResult classify()=0;

  virtual void save(const std::string& filename)=0;

  virtual void load(const std::string& filename)=0;

  virtual boost::shared_ptr<ConfusionMatrix> getConfusionMatrix()=0;
};

typedef boost::shared_ptr<SimpleClassifier> SCPtr;

class ServerSideRepo;
class DataRepository;
/**
 * Client stub for communicating with the classifiers managed by the classifier manager.
 */
class ClassifierClient : public ServiceCallResultHolder, public SimpleClassifier
{
private:
  void createClients();
  virtual void checkState() const;
protected:
  boost::shared_ptr<DataRepository> dataRepository;
  ros::NodeHandle nh;
  std::string _serviceBasePath;
  std::string classifierName;
  std::string parameters;
  int classifierId;
  std::string buildModelName;
  std::string evaluateName;
  std::string classifyName;
  std::string setDatasetName;
  std::string saveClassifierName;
  std::string addNewClassifierName;
  std::string loadClassifierName;
  std::string freeClassifierName;
  ros::ServiceClient buildModelClient;
  ros::ServiceClient evaluateClient;
  ros::ServiceClient classifyClient;
  ros::ServiceClient setDatasetClient;
  ros::ServiceClient addNewClassifierClient;
  ros::ServiceClient saveClassifierClient;
  ros::ServiceClient loadClassifierClient;
  ros::ServiceClient freeClassifierClient;
  boost::shared_ptr<ConfusionMatrix> confusionMatrix;
  bool ownRepo;
public:

  /**
   * Basic constructor, requires node handle, path to service and optionaly a reference to DataRepository
   */
  ClassifierClient(ros::NodeHandle& nh, const std::string& serviceBasePath, boost::shared_ptr<DataRepository> repo);
  /**
   * Constructor can be used to create a classifier on the server side during object construction
   */
  ClassifierClient(ros::NodeHandle& nh, const std::string& serviceBasePath, const std::string& classifierName,
                   const std::string& parameters = "",
                   boost::shared_ptr<DataRepository> repo = boost::shared_ptr<DataRepository>());
  /**
   * This constructor can be used to connect to a classifier on the server via its id
   */
  ClassifierClient(ros::NodeHandle& nh, const std::string& serviceBasePath, int classifierId,
                   boost::shared_ptr<DataRepository> repo = boost::shared_ptr<DataRepository>());

  virtual ~ClassifierClient();

  virtual void assignData(const std::string& name, DataSetSlot type);

  virtual void train(const std::string& params = "");

  virtual void train(const std::string& params, DS& dataset, const std::string& features = "", const std::string& gt =
                         "");

  virtual EvaluationResult evaluate();

  virtual EvaluationResult evaluate(DS& dataset, const std::string& features = "", const std::string& gt = "");

  virtual ClassificationResult classify();

  virtual ClassificationResult classify(DS& dataset, const std::string& features = "");

  virtual int getClassifierId() const;
  virtual void setClassifierId(int classifierId);

  virtual void save(const std::string& filename);

  virtual void load(const std::string& filename);

  /**
   * Call this to delete the server side part of the classifier
   */
  virtual void free();

  virtual boost::shared_ptr<ConfusionMatrix> getConfusionMatrix();

};

/**
 * Basic data repository functionality.
 */
class DataRepository
{
public:
  virtual void uploadData(const std::string& pathToDataset, const std::string& name)=0;
  virtual void uploadData(DS& dataset, const std::string& name, const std::string& features = "",
                          const std::string& gt = "")=0;
  virtual void removeData(const std::string& name)=0;

  virtual ~DataRepository()=0;
};

/**
 * A data repository on the server. Data uploaded to this repo is accessible by all classifiers
 */
class ServerSideRepo : public ServiceCallResultHolder, public DataRepository
{

protected:
  void checkState() const;
  ros::NodeHandle nh;
  std::string _serviceBasePath;
  std::string addDatasetName;
  std::string setDatasetName;
  std::string removeDatasetName;
  std::string readDatasetName;
  ros::ServiceClient addDatasetClient;
  ros::ServiceClient setDatasetClient;
  ros::ServiceClient removeDatasetClient;
  ros::ServiceClient readDatasetClient;
private:
  bool invalidState;
  void createClients();
public:
  ServerSideRepo(ros::NodeHandle& nh, const std::string& serviceBasePath);
  virtual ~ServerSideRepo();
  virtual void uploadData(const std::string& pathToDataset, const std::string& name);
  virtual void uploadData(DS& dataset, const std::string& name, const std::string& features = "",
                          const std::string& gt = "");
  virtual void removeData(const std::string& name);

};
std::string ensureServiceBasePathFormat(const std::string& serviceBasePath);

}
#endif /* CLIENT_H_ */
