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

#include <icf_core/base/ConfusionMatrix.hpp>
namespace icf
{

DS::Matrix * createDummyConfidenceMatrix(LabelMap map, boost::shared_ptr<std::vector<int> > labels)
{
  DS::Matrix * matrix = new DS::Matrix();
  matrix->setZero(labels->size(), map.indexToLabel.size());
  for (std::vector<int>::iterator iter = labels->begin(); iter != labels->end(); iter++)
  {
    int row = iter - labels->begin();
    (*matrix)(row, map.mapToIndex(*iter)) = 1.0;
  }
  return matrix;
}

ConfusionMatrix::ConfusionMatrix()
{

}

ConfusionMatrix::ConfusionMatrix(const std::string& filename)

{
  load(filename);
}

/**
 * @param groundTruth as user labels
 * @param classification as user labels
 */
ConfusionMatrix::ConfusionMatrix(MatrixPtr groundTruth, MatrixPtr classification) :
    labelMap(LabelMap(groundTruth, classification))
{
  using namespace std;
  Matrix& gt = *groundTruth;
  Matrix& cl = *classification;
  cm = ConfusionMatrix::MatrixPtr(
      new ConfusionMatrix::Matrix((int)labelMap.indexToLabel.size(), (int)labelMap.indexToLabel.size()));
  cm->setZero();

  for (LabelMap::IndexToLabel::iterator iter = labelMap.indexToLabel.begin(); iter != labelMap.indexToLabel.end();
      iter++)
  {
    int index = iter->first;
    int label = iter->second;
    for (int i = 0; i < cl.rows(); i++)
    {
      if (gt(i, 0) == label)
      {
        (*cm)(index, labelMap.mapToIndex(cl(i, 0)))++;}

      }
    }

  }

ConfusionMatrix ::~ConfusionMatrix()
{

}

const LabelMap& ConfusionMatrix::getLabelMap() const
{
  return labelMap;
}

const ConfusionMatrix::Matrix& ConfusionMatrix::getCM() const
{
  return *cm;
}

bool ConfusionMatrix::save(const std::string& filename) const
{
  std::fstream out(filename.c_str(), std::ios_base::out);
  if (!out)
  {
    return false;
  }
  try
  {
    out << serialize();
  }
  catch (...)
  {
    out.close();
    throw;
  }
  out.close();
  return true;
}

bool ConfusionMatrix::load(const std::string& filename)
{
  std::fstream in(filename.c_str(), std::ios_base::in);
  if (!in)
  {
    return false;
  }
  cm = MatrixPtr(readMatrixASCII(in));
  labelMap.load(in);
  in.close();
  return true;
}

bool ConfusionMatrix::deserialize(const std::string& serialized)
{
  std::stringstream in(serialized);
  deserialize(in);
  return true;
}

bool ConfusionMatrix::deserialize(std::istream& in)
{
  cm = MatrixPtr(readMatrixASCII(in));
  labelMap.load(in);
  return true;
}

std::string ConfusionMatrix::serialize() const
{
  std::stringstream out;
  writeMatrixASCII(*cm, out);
  out << std::endl;
  labelMap.save(out);
  return out.str();
}

bool ConfusionMatrix::operator ==(const ConfusionMatrix& other) const
{
  bool lmeq = labelMap == other.getLabelMap();
  bool cmeq = *cm == other.getCM();
  return lmeq && cmeq;
}

bool ConfusionMatrix::operator !=(const ConfusionMatrix& other) const
{
  bool lmeq = labelMap == other.getLabelMap();
  bool cmeq = *cm == other.getCM();
  return !(lmeq && cmeq);
}

int ConfusionMatrix::operator ()(int i, int j)
{
  return (*cm)(labelMap.mapToIndex(i), labelMap.mapToIndex(j));
}

const std::ostream& ConfusionMatrix::operator >>(const std::ostream& out) const
{
  out << serialize();
  return out;
}

const std::ostream & operator <<(const std::ostream& out, const ConfusionMatrix& cf)
{
  out << cf.serialize();
  return out;
}

std::ostream& operator <<(std::ostream& out, const ConfusionMatrix& cf)
{
  out << cf.serialize();
  return out;
}

std::istream & operator >>(std::istream& in, ConfusionMatrix& cf)
{
  cf.deserialize(in);
  return in;
}

void ConfusionMatrix::normalizeRows()
{
	DS::Matrix& m = *cm;

	for(int i=0;i<m.rows();i++)
	{
		double sum=0.0;
		for(int j=0;j<m.cols();j++)
		{
			sum+=m(i,j);
		}
		for(int j=0;j<m.cols();j++)
		{
			m(i,j)/=sum;
		}
	}
}

void ConfusionMatrix::normalizeCols()
{
	DS::Matrix& m = *cm;

	for(int i=0;i<m.cols();i++)
	{
		double sum=0.0;
		for(int j=0;j<m.rows();j++)
		{
			sum+=m(j,i);
		}
		for(int j=0;j<m.rows();j++)
		{
			m(j,i)/=sum;
		}
	}
}

}
