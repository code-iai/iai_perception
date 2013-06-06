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

#ifndef CLASSIFICATION_RESULT
#define CLASSIFICATION_RESULT

#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <istream>
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <icf_dataset/DataSet.hpp>
#include <icf_core/base/ICFExceptionErrors.h>
#include <icf_core/base/LabelMap.hpp>

namespace icf
{
/**
 * This class wrapps the results of a "classify" service call
 */
class ClassificationResult
{
public:

  /**
   * If you have a LabelMap use this one
   */
  ClassificationResult(const LabelMap& labelMap);

  /**
   * If you have a serialized classification result use this one
   */
  ClassificationResult(const std::string& serialized);

  /**
   * If you have a labelMap and results, use this one
   */
  ClassificationResult(const LabelMap& labelMap, boost::shared_ptr<std::vector<int> > results);

  /**
   * push a result to the list of results
   */
  void add(int result);

  boost::shared_ptr<std::vector<int> > results;

  /**
   * get the classifiers confidence for an instance and label
   */
  double confidenceFor(int instance, int label);

  DataSet<double>::MatrixPtr confidences;
  LabelMap labelMap;
};

std::istream& operator>>(std::istream& in, ClassificationResult& cr);

std::ostream& operator<<(std::ostream& out, const ClassificationResult& cr);

}

#endif
