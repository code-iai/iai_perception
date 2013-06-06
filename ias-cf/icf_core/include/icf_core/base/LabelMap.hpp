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

#ifndef LABELMAP_HPP_
#define LABELMAP_HPP_
#include <icf_dataset/DataSet.hpp>
#include <icf_core/base/ICFExceptionErrors.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <map>
#include <set>

namespace icf
{

/**
 * User supplied labels do not need to start with 0 or be consecutively labeled, but classifiers
 * and the confusion matrix class requires a mapping from indices [0,N-1] where N is the number of
 * labels to the user supplied labels. This class encapsulates the required functionality.
 */
class LabelMap
{
public:
  typedef DS::MatrixPtr MatrixPtr;
  typedef DS::Matrix Matrix;
  typedef int Index;
  typedef int Label;
  typedef std::map<Label, Index> LabelToIndex;
  typedef std::map<Index, Label> IndexToLabel;

  LabelMap()
  {

  }

  LabelMap(const std::string& serialized)
  {
    std::stringstream ss(serialized);
    Label lbl;
    Index index = 0;
    while (!ss.eof())
    {
      ss >> lbl;
      labelToIndex[lbl] = index;
      indexToLabel[index] = lbl;
      ++index;
    }
  }

  LabelMap(std::vector<int>& labels)
  {
    int index = 0;
    std::set<int> ll;
    for (unsigned int i = 0; i < labels.size(); i++)
    {
      ll.insert(labels[i]);
    }
    for (std::set<int>::iterator iter = ll.begin(); iter != ll.end(); iter++)
    {
      labelToIndex[*iter] = index;
      indexToLabel[index++] = *iter;
    }
  }

  LabelMap(MatrixPtr labels)
  {
    assert(labels.get()!=NULL);

    int index = 0;

    std::set<int> ll;
    for (int i = 0; i < labels->rows(); i++)
    {
      ll.insert((*labels)(i, 0));
    }

    for (std::set<int>::iterator iter = ll.begin(); iter != ll.end(); iter++)
    {
      labelToIndex[*iter] = index;
      indexToLabel[index++] = *iter;
    }
  }

  LabelMap(MatrixPtr labels, MatrixPtr labels2)
  {
    assert(labels.get()!=NULL);

    int index = 0;

    std::set<int> ll;
    for (int i = 0; i < labels->rows(); i++)
    {
      ll.insert((*labels)(i, 0));
    }
    for (int i = 0; i < labels2->rows(); i++)
    {
      ll.insert((*labels2)(i, 0));
    }
    for (std::set<int>::iterator iter = ll.begin(); iter != ll.end(); iter++)
    {
      labelToIndex[*iter] = index;
      indexToLabel[index++] = *iter;
    }
  }

  virtual ~LabelMap()
  {

  }

  bool operator ==(const LabelMap& other) const
  {
    return labelToIndex == other.labelToIndex && indexToLabel == other.indexToLabel;
  }

  int mapToIndex(int label) const
  {
    LabelToIndex::const_iterator iter = labelToIndex.find(label);
    if (iter == labelToIndex.end())
    {
      throw std::runtime_error("invalid label");
    }
    return iter->second;
  }

  int mapToLabel(int index) const
  {
    IndexToLabel::const_iterator iter = indexToLabel.find(index);
    if (iter == indexToLabel.end())
    {
      throw std::runtime_error("invalid index");
    }
    return iter->second;
  }

  void mapToIndex(std::vector<int>& labels)
  {
    for (size_t r = 0; r < labels.size(); r++)
    {
      LabelToIndex::const_iterator iter = labelToIndex.find(labels[r]);
      if (iter == labelToIndex.end())
      {
        throw std::runtime_error("invlid label");
      }
      labels[r] = iter->second;
    }
  }

  void mapToIndex(boost::shared_ptr<std::vector<int> > labels)
  {
    if (labels)
    {
      mapToIndex(*labels);
    }
  }

  void mapToIndex(Matrix& labels) const
  {
    for (int r = 0; r < labels.rows(); r++)
    {
      LabelToIndex::const_iterator iter = labelToIndex.find(labels(r, 0));
      if (iter == labelToIndex.end())
      {
        throw std::runtime_error("invlid label");
      }
      labels(r, 0) = iter->second;
    }
  }

  void mapToLabels(Matrix& index) const
  {
    for (int r = 0; r < index.rows(); r++)
    {
      IndexToLabel::const_iterator iter = indexToLabel.find(index(r, 0));
      if (iter == indexToLabel.end())
      {
        throw std::runtime_error("invalid index");
      }
      index(r, 0) = iter->second;
    }
  }

  void mapToLabels(boost::shared_ptr<std::vector<int> > indices)
  {
    if (indices)
    {
      mapToLabels(*indices);
    }
  }

  void mapToLabels(std::vector<int>& indices) const
  {
    for (size_t r = 0; r < indices.size(); r++)
    {
      IndexToLabel::const_iterator iter = indexToLabel.find(indices[r]);
      if (iter == indexToLabel.end())
      {
        throw std::runtime_error("invalid index");
      }
      indices[r] = iter->second;
    }
  }

  MatrixPtr mapToIndex(MatrixPtr labels) const
  {
    assert(labels.get()!=NULL);

    MatrixPtr mapped(new Matrix(labels->rows(), 1));
    double * mappedData = mapped->data();
    for (int r = 0; r < labels->rows(); r++)
    {
      LabelToIndex::const_iterator iter = labelToIndex.find((*labels)(r, 0));
      if (iter == labelToIndex.end())
      {
        throw std::runtime_error("invalid label");
      }
      mappedData[r] = iter->second;
    }
    return mapped;
  }

  MatrixPtr mapToLabel(MatrixPtr indices) const
  {
    assert(indices.get()!=NULL);

    MatrixPtr mapped(new Matrix(indices->rows(), 1));
    double * mappedData = mapped->data();
    for (int r = 0; r < indices->rows(); r++)
    {
      IndexToLabel::const_iterator iter = indexToLabel.find((*indices)(r, 0));
      if (iter != indexToLabel.end())
      {
        mappedData[r] = iter->second;
      }
      else
      {
        throw std::runtime_error("invalid index");
      }
    }
    return mapped;
  }

  bool save(std::ostream & out) const
  {
    for (LabelToIndex::const_iterator iter = labelToIndex.begin(); iter != labelToIndex.end();)
    {
      out << iter->first;
      ++iter;
      if (iter != labelToIndex.end())
      {
        out << " ";
      }
    }
    return true;
  }

  bool save(const std::string& filename) const
  {
    std::fstream out(filename.c_str(), std::ios_base::out);
    if (!out)
    {
      return false;
    }
    bool result = save(out);
    out.close();
    return result;
  }

  int numLabels() const
  {
    return labelToIndex.size();
  }

  bool load(std::istream& in)
  {
    Label lbl;
    Index index = 0;
    while (!in.eof())
    {
      in >> lbl;
      labelToIndex[lbl] = index;
      indexToLabel[index] = lbl;
      ++index;
    }
    return true;
  }

  bool load(const std::string& filename)
  {
    std::fstream in(filename.c_str(), std::ios_base::in);
    if (!in)
    {
      return false;
    }
    bool result = load(in);
    in.close();
    return result;
  }

  LabelToIndex labelToIndex;
  IndexToLabel indexToLabel;
};

std::ostream& operator<<(std::ostream& out, const LabelMap& lm);

}

#endif /* LABELMAP_HPP_ */
