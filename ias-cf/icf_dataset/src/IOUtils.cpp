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

#include <icf_dataset/IOUtils.h>

namespace icf
{

DS::Matrix * readMatrixASCII(const std::string& matrixString)
{

  std::stringstream ss;
  ss << matrixString;
  return readMatrixASCII(ss);

}

/*DS::Matrix * createDummyConfidenceMatrix(LabelMap map,
 boost::shared_ptr<std::vector<int> > labels) {
 DS::Matrix * matrix = new DS::Matrix();
 matrix->setZero(labels->size(), map.indexToLabel.size());
 for (std::vector<int>::iterator iter = labels->begin();
 iter != labels->end(); iter++) {
 int row = iter - labels->begin();
 (*matrix)(row, map.mapToIndex(*iter)) = 1.0;
 }
 return matrix;
 }*/

DS::Matrix * readMatrixASCII(std::istream& in)
{

  int rows = 0;
  int cols = 0;
  in >> rows;
  in >> cols;
  DS::Matrix& matrix = *(new DS::Matrix(rows, cols));
  double value;
  for (int row = 0; row < rows; row++)
  {
    for (int col = 0; col < cols; col++)
    {
      in >> value;
      matrix(row, col) = value;
    }
  }
  return &matrix;
}

void writeMatrixASCII(const DS::Matrix& matrix, std::ostream& out)
{
  out << matrix.rows() << std::endl;
  out << matrix.cols() << std::endl;
  out << matrix;
}

void writeMatrixASCII(const DS::Matrix& matrix, const std::string& filename, bool append)
{

  std::_Ios_Openmode mode = std::ios_base::out;
  if (!append)
  {
    mode |= std::ios_base::trunc;
  }
  else
  {
    mode |= std::ios_base::app;
  }
  std::fstream out(filename.c_str(), mode);
  if (append)
  {
    out << std::endl;
  }
  writeMatrixASCII(matrix, out);
  out.close();
}

}