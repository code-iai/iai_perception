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

#ifndef BOOSTDL_H_
#define BOOSTDL_H_

#include <tclap/CmdLine.h>

#include <icf_dataset/DataSet.hpp>
#include <icf_core/base/ICFExceptionErrors.h>
#include <icf_core/base/Classifier.hpp>
#include <icf_core/base/ClassificationResult.hpp>
#include <icf_core/base/EvaluationResult.hpp>
#include <icf_core/base/LabelMap.hpp>

#include <opencv2/opencv.hpp>

//stl
#include <sstream>
#include <istream>
#include <fstream>
#include <cmath>
#include <iostream>
#include <set>
#include <cmath>
#include <ctime>
#include <sys/stat.h>

#include <boost/thread.hpp>

namespace icf
{

class BoostOAA : public Classifier
{
public:
  BoostOAA(std::string params, int threadCount = 4);
  virtual ~BoostOAA();

  /**
   *\brief method for adding training data
   *\param data to be interperted and added
   */
  virtual void addTrainingData(std::string);

  virtual void addData(std::string);

  virtual void addEvaluationData(std::string);

  /**
   *\brief
   */
  virtual int buildModel(std::string);

  /**
   *\brief classifying/adding test data use this function when you don't have ground truth
   *\return result for given test data
   *\param data to be tested
   */
  virtual std::string classify();

  /**
   *\brief basically the same as classify, use it when you have ground truth so confusion matrix can be created
   */
  virtual std::string evaluate();

  /**
   * \brief save the model. Throw a std::string if something goes wrong.
   */
  virtual void save(std::string filename);

  /**
   * \brief load the model. Throw a std::string if something goes wrong.
   */
  virtual void load(std::string filename);

private:
  void parse_parameters(std::string);
  void relabel(cv::Mat& labels, float cls);
  std::set<int> countClasses(cv::Mat labels);
  bool useSquashingFunction;
  CvBoostParams params;
  int threadCount;
  LabelMap labelMap;
  std::vector<std::pair<int, boost::shared_ptr<CvBoost> > > classifiers;
  boost::mutex m;
  boost::mutex c;
  std::string ds_prefix;
};
}

#endif /* BOOSTDL_H_ */
