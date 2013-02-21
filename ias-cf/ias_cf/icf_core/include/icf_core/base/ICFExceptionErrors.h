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

#ifndef ICFEXCEPTIONERRORS_H_
#define ICFEXCEPTIONERRORS_H_


#include <exception>
#include <boost/exception/all.hpp>
#include <icf_dataset/ICFException.h>

namespace icf
{

typedef boost::error_info<struct service_call_returned_false, boost::tuple<std::string, std::string> > service_call_returned_false_error;
inline std::string to_string(service_call_returned_false_error const & e)
{
  return "Service: " + e.value().get<0>() + "\nMessage: " + e.value().get<1>() + "\n";
}

typedef boost::error_info<struct service_unavailable, std::string> service_unavailable_error;
typedef boost::error_info<struct invalid_state, std::string> invalid_state_error;
typedef boost::error_info<struct invalid_classifier_id, int> invalid_classifier_id_error;
typedef boost::error_info<struct dataset_matrix_not_found, std::string> dataset_matrix_not_found_error;
typedef boost::error_info<struct not_implemented, std::string> not_implemented_error;
typedef boost::error_info<struct service_unavailable_collection_, std::vector<service_unavailable_error> > service_unavailable_collection;

inline std::string to_string(service_unavailable_collection const & e)
{
  std::string errorMsg = "";
  for (size_t i = 0; i < e.value().size(); i++)
  {
    errorMsg += to_string(e.value().at(i));
  }
  return errorMsg;
}

typedef boost::error_info<struct invalid_argument_, std::string> invalid_argument;

typedef boost::error_info<struct matrix_dimensions_do_not_match_, std::string> matrix_dimensions_do_not_match;

//typedef boost::error_info<struct hdf5_error_, std::string> hdf5_error;

//struct ICFException: virtual boost::exception {};

}
#endif /* ICFEXCEPTIONERRORS_H_ */
