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

#include <icf_core/base/EvaluationResult.hpp>

namespace icf
{

EvaluationResult::EvaluationResult(boost::shared_ptr<std::vector<int> > groundTruth,
                                   boost::shared_ptr<std::vector<int> > classification) :
    errorRate(-1)
{
  Matrix * gt = new Matrix(groundTruth->size(), 1);
  this->groundTruth = MatrixPtr(gt);
  Matrix * cl = new Matrix(classification->size(), 1);
  this->results = MatrixPtr(cl);
  for (unsigned int i = 0; i < groundTruth->size(); i++)
  {
    (*gt)(i, 0) = groundTruth->at(i);
    (*cl)(i, 0) = classification->at(i);
  }
  this->cm = boost::shared_ptr<ConfusionMatrix>(new ConfusionMatrix(this->groundTruth, this->results));
}

EvaluationResult::EvaluationResult(MatrixPtr groundTruth, boost::shared_ptr<std::vector<int> > classification) :
    groundTruth(groundTruth), errorRate(-1)
{

  assert(((size_t)groundTruth->rows())==classification->size());
  Matrix * cl = new Matrix(classification->size(), 1);
  this->results = MatrixPtr(cl);
  for (int i = 0; i < groundTruth->rows(); i++)
  {
    (*cl)(i, 0) = classification->at(i);
  }
  ConfusionMatrix * cm = new ConfusionMatrix(this->groundTruth, this->results);
  this->cm = boost::shared_ptr<ConfusionMatrix>(cm);

}

EvaluationResult::EvaluationResult(MatrixPtr groundTruth, MatrixPtr classification) :
    groundTruth(groundTruth), results(classification), cm(
        boost::shared_ptr<ConfusionMatrix>(new ConfusionMatrix(this->groundTruth, this->results))), errorRate(-1)
{

}

EvaluationResult::EvaluationResult(const std::string& serialized) :
    errorRate(-1)
{

  std::stringstream in(serialized);
  results = MatrixPtr(readMatrixASCII(in));
  groundTruth = MatrixPtr(readMatrixASCII(in));
  cm = boost::shared_ptr<ConfusionMatrix>(new ConfusionMatrix(groundTruth, results));
  if (!in.eof())
  {
    confidences = MatrixPtr(readMatrixASCII(in));
  }
}

const boost::shared_ptr<ConfusionMatrix> EvaluationResult::getConfusionMatrix() const
{
  return this->cm;
}

void EvaluationResult::setConfidence(MatrixPtr confidence)
{
  this->confidences = confidence;
}

EvaluationResult::MatrixPtr EvaluationResult::getConfidence() const
{
  return confidences;
}

double EvaluationResult::getErrorRate() const
{
  if (errorRate == -1)
  {
    Matrix gt = *groundTruth;
    Matrix cr = *results;
    errorRate = 0.0;
    for (int i = 0; i < gt.rows(); i++)
    {
      if (gt(i, 0) != cr(i, 0))
      {
        errorRate++;
      }
    }
    errorRate /= (double)gt.rows();
  }
  return errorRate;
}

std::ostream& operator<<(std::ostream& out, const EvaluationResult& er)
{
  writeMatrixASCII(*er.results, out);
  out << std::endl;
  writeMatrixASCII(*er.groundTruth, out);
  if (er.confidences.get() != NULL)
  {
    out << std::endl;
    writeMatrixASCII(*er.confidences, out);
  }
  return out;
}

}
