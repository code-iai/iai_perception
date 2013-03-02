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

#ifndef KNNCLASSIFIER_H_
#define KNNCLASSIFIER_H_

#include <icf_dataset/IOUtils.h>
#include <icf_dataset/DataSet.hpp>
#include "icf_core/base/ICFExceptionErrors.h"
#include <icf_core/base/Classifier.hpp>
#include <icf_core/base/ClassificationResult.hpp>
#include <icf_core/base/EvaluationResult.hpp>
#include <icf_core/base/LabelMap.hpp>
#include <icf_core/classifiers/KNNImpl.h>

#include <tclap/CmdLine.h>

#include <boost/algorithm/string.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

#include <sstream>
#include <string>
#include <cstdlib>
#include <iostream>
#include <istream>
#include <fstream>

namespace icf
{

template<class T>
  class KNNClassifier : public Classifier
  {
  public:

    KNNClassifier(std::string parameters);

    virtual ~KNNClassifier();
    /**
     *\brief method for adding training data
     *\param data to be interperted and added; cell separator: space; row seperator: \n
     *\param  each row one example; last column label;
     */
    virtual void addTrainingData(std::string data);

    virtual void addData(std::string);

    virtual void addEvaluationData(std::string);

    void save(std::string filename);
    void load(std::string filename);

    /**
     *\brief
     */
    virtual int buildModel(std::string data);

    /**
     *\brief classifying/adding test data
     *\return result for given test data
     *\param data to be tested
     */
    virtual std::string classify();

    //TODO implement
    virtual std::string evaluate();

    virtual std::string executeService(std::string service, std::string params);

  private:
    void preprocessTrainingData(DS& ds);
    bool parse_parameters(std::string parameters);

    std::string evaluationData;
    std::string classificationData;

    int k;
    std::string m;
    double mParam;
    IKNNImpl<T> * classifier;
    flann::Matrix<T> * data;
    boost::shared_ptr<std::vector<int> > classLabels;
    bool rebuildModel;
    int folds;
    bool w;
    LabelMap labelMap;
    DS::Ptr ds;
    std::string ds_prefix;

  };

}
#endif /* KNNCLASSIFIER_H_ */
