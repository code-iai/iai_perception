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

#include <icf_core/base/ClassificationResult.hpp>

namespace icf
{

ClassificationResult::ClassificationResult(const LabelMap& labelMap) :
    results(new std::vector<int>()), labelMap(labelMap)
{

}

ClassificationResult::ClassificationResult(const std::string& serialized) :
    results(new std::vector<int>())
{
  std::stringstream in(serialized);
  std::string line;
  std::getline(in, line);
  std::stringstream crIn(line);
  while (!crIn.eof())
  {
    std::string element;
    crIn >> element;
    results->push_back(atoi(element.c_str()));
  }
  std::getline(in, line);
  std::stringstream lmIn(line);
  labelMap = LabelMap(line);
  if (!in.eof())
  {
    int rows;
    int cols;
    in >> rows;
    in >> cols;
    DS::MatrixPtr matrix(new DataSet<double>::Matrix(rows, cols));
    double value;
    for (int row = 0; row < rows; row++)
    {
      for (int col = 0; col < cols; col++)
      {
        in >> value;
        (*matrix)(row, col) = value;
      }
    }
    confidences = matrix;
  }
}

ClassificationResult::ClassificationResult(const LabelMap& labelMap, boost::shared_ptr<std::vector<int> > results) :
    results(results), labelMap(labelMap)
{

}

void ClassificationResult::add(int result)
{
  this->results->push_back(result);
}

double ClassificationResult::confidenceFor(int instance, int label)
{
  if (confidences.get() == NULL)
  {
    throw ICFException() << invalid_state_error("No confidences!");
  }
  return (*confidences)(instance, labelMap.mapToIndex(label));
}

std::istream& operator>>(std::istream& in, ClassificationResult& cr)
{
  std::string line;
  std::getline(in, line);
  std::stringstream crIn(line);
  while (!crIn.eof())
  {
    std::string element;
    crIn >> element;
    cr.results->push_back(atoi(element.c_str()));
  }
  if (!in.eof())
  {
    int rows;
    int cols;
    in >> rows;
    in >> cols;
    DataSet<double>::MatrixPtr matrix(new DataSet<double>::Matrix(rows, cols));
    double value;
    for (int row = 0; row < rows; row++)
    {
      for (int col = 0; col < cols; col++)
      {
        in >> value;
        (*matrix)(row, col) = value;
      }
    }
    cr.confidences = matrix;
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const ClassificationResult& cr)
{
  for (unsigned int i = 0; i < cr.results->size(); i++)
  {
    out << cr.results->at(i);
    if (i != cr.results->size() - 1)
      out << " ";
    else
      out << std::endl;
  }
  out << cr.labelMap;
  if (cr.confidences.get() != NULL)
  {
    out << std::endl;
    out << cr.confidences->rows() << std::endl;
    out << cr.confidences->cols() << std::endl;
    out << *(cr.confidences);
  }
  return out;
}
}
