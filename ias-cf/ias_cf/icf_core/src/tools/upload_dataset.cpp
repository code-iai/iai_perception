/*
 * Copyright (c) 2012,
 * Zoltan-Csaba Marton <marton@cs.tum.edu>,
 * Ference Balint-Benczedi <balintb.ferenc@gmail.com>,
 * Florian Seidel <seidel.florian@gmail.com
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
#include <iostream>
#include <ros/ros.h>
#include <tclap/CmdLine.h>
#include <icf_core/client/Client.h>

using namespace icf;
using namespace TCLAP;

int main(int argc, char **argv)
{
  //Command line parser
  CmdLine cmdLine("upload_dataset", ' ', "dev");
  ValueArg<std::string> iArg("i", "input", "the HDF5 file to be uploaded", true, "i", "string");
  ValueArg<std::string> nArg("n", "name", "a name for the dataset", true, "n", "string");
  ValueArg<std::string> fArg("f", "features", "name of the matrix containing the features (default: /x), use with -l", false, "", "string");
  ValueArg<std::string> lArg("l", "labels", "name of the vector containing the labels (default: /y), use with -f", false, "", "string");
  ValueArg<std::string> mArg("m", "manager", "name of the classification service manager node (default: ias_classifier_manager)", false, "ias_classifier_manager", "string");
  cmdLine.add(mArg);
  cmdLine.add(lArg);
  cmdLine.add(fArg);
  cmdLine.add(nArg);
  cmdLine.add(iArg);
  cmdLine.parse(argc, argv);

  // init node
  ros::init(argc, argv, "upload_dataset");
  ros::NodeHandle n("~");

  try
  {
    // open file
    DS ds(iArg.getValue(), false, true);

    // upload data
    ServerSideRepo data_store(n, mArg.getValue());
    if (fArg.getValue() == "" || lArg.getValue() == "")
      data_store.uploadData(ds, nArg.getValue());
    else
      data_store.uploadData(ds, nArg.getValue(), fArg.getValue(), lArg.getValue());
  }
  catch (ICFException& e)
  {
    std::cerr << boost::diagnostic_information(e);
    std::vector<service_unavailable_error>* err = boost::get_error_info<service_unavailable_collection>(e);
    if (err != NULL)
    {
      for (std::vector<service_unavailable_error>::iterator iter = err->begin(); iter != err->end(); iter++)
        std::cerr << "Error: " << iter->value() << std::endl;
    }
    else
      std::cerr << "No service availability related errors" << std::endl;
  }

  return 0;
}

