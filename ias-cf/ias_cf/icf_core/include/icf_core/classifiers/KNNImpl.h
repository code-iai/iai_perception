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

#ifndef KNNIMP_H_
#define KNNIMP_H_

#include <flann/flann.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/control_structures.hpp>
#include <boost/lambda/closures.hpp>
#include <boost/lambda/casts.hpp>
#include <boost/lambda/algorithm.hpp>

#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdlib>

#include <icf_core/base/ICFExceptionErrors.h>

namespace icf
{

class KNNImplTest;

template<class K>
  std::vector<int> maxIndices(K start, K end);

template<class T>
  class IKNNImpl
  {
  public:
    typedef boost::tuple<boost::shared_ptr<std::vector<int> >, boost::shared_ptr<flann::Matrix<int> >,
        boost::shared_ptr<flann::Matrix<T> > > KNNResult;

    virtual KNNResult classify(typename flann::Matrix<T>& instances) const =0;

    virtual ~IKNNImpl()
    {

    }

    virtual void save(std::string baseName)=0;

    virtual boost::shared_ptr<std::vector<int> > getLabels()=0;

  };

template<class Metric>
  class KNNImpl : public IKNNImpl<typename Metric::ElementType>
  {
  public:
    friend class KNNImplTest;
    typedef typename IKNNImpl<typename Metric::ElementType>::KNNResult KNNResult;

    KNNImpl(unsigned int k, boost::shared_ptr<typename flann::Index<Metric> > m,
            boost::shared_ptr<std::vector<int> > labels, bool weighVotes = false);

    KNNImpl(const flann::Matrix<typename Metric::ElementType>& dataset, std::string fileName, unsigned int k,
            bool weighVotes = false, Metric m = Metric());

    virtual KNNResult classify(flann::Matrix<typename Metric::ElementType>& instances) const;

    virtual void save(std::string baseName);

    virtual boost::shared_ptr<std::vector<int> > getLabels()
    {
      return labels;
    }

    virtual ~KNNImpl();
  private:

    unsigned int k;
    boost::shared_ptr<flann::Index<Metric> > index;
    boost::shared_ptr<std::vector<int> > labels;
    bool weighVotes;

  };

}

#endif /* KNNIMP_H_ */
