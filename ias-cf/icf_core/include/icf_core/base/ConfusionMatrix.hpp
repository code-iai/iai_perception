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

#ifndef CONFUSION_MATRIX_HPP
#define CONFUSION_MATRIX_HPP

#include <icf_dataset/IOUtils.h>
#include <icf_dataset/DataSet.hpp>
#include <icf_core/base/ICFExceptionErrors.h>
#include <icf_core/base/LabelMap.hpp>

namespace icf
{

DS::Matrix * createDummyConfidenceMatrix(LabelMap map, boost::shared_ptr<std::vector<int> > labels);

/**
 * Contains a contingency table
 */
class ConfusionMatrix
{
public:
  typedef DS::MatrixPtr MatrixPtr;
  typedef DS::Matrix Matrix;

  ConfusionMatrix();
  ConfusionMatrix(const std::string& filename);
  ConfusionMatrix(MatrixPtr groundTruth, MatrixPtr classification);
  virtual ~ConfusionMatrix();

  const LabelMap& getLabelMap() const;

  const Matrix& getCM() const;

  int operator ()(int i, int j);

  bool save(const std::string& filename) const;
  bool load(const std::string& filename);
  bool deserialize(const std::string& serialized);
  bool deserialize(std::istream& in);
  std::string serialize() const;
  bool operator ==(const ConfusionMatrix& other) const;
  bool operator !=(const ConfusionMatrix& other) const;
  const std::ostream& operator >>(const std::ostream& out) const;
  void normalizeRows();
  void normalizeCols();
private:
  LabelMap labelMap;
  MatrixPtr cm;

};

std::istream& operator >>(std::istream & in, ConfusionMatrix& cf);
const std::ostream& operator <<(const std::ostream& out, const ConfusionMatrix& cf);
std::ostream& operator <<(std::ostream& out, const ConfusionMatrix& cf);

}
#endif
