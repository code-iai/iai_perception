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

#ifndef DATASET_HPP_
#define DATASET_HPP_

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string.hpp>
#include <hdf5/hdf5.h>
#include <hdf5/hdf5_hl.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <icf_dataset/ICFException.h>

namespace icf
{

/**
 * This class is a wrapper for the HDF5 API. It supports a limited subset of the
 * HDF5 standard. <br/>
 * Supported are
 *
 */
template<class T, int StorageOrder = Eigen::RowMajor>
  class DataSet
  {
  private:
    std::string fileName;
    bool create;
    bool flushOnClose;
    hid_t fapl;
    hid_t file_id;
    mutable int *refCount;

  public:

    typedef boost::shared_ptr<DataSet<T> > Ptr;
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, StorageOrder> Matrix;
    typedef boost::shared_ptr<Matrix> MatrixPtr;
    typedef std::map<std::string, MatrixPtr> FeatureMatrices;

    /**
     * Create a dataset in memory which will not be written to disk after destruction
     */
    DataSet() :
        flushOnClose(false), file_id(-1)
    {
      refCount = new int;
      *refCount = 1;
      std::stringstream ss;
      ss << rand() << rand();
      fileName = "/" + ss.str();
      int blockSize = 1024;
      fapl = H5Pcreate(H5P_FILE_ACCESS);

      if (H5Pset_fapl_core(fapl, blockSize, flushOnClose) < 0)
      {
        H5Pclose(fapl);
        fapl = 0;
        throw ICFException() << hdf5_error("Error setting file access property list for in-memory file");
      }

      file_id = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

      if (file_id < 0)
      {
        H5Pclose(fapl);
        fapl = 0;
        throw ICFException() << hdf5_error("Error opening in-memory file");
      }

    }

    /**
     * Create a dataset in memory from a serialized dataset (image; you get one from calling serialize()) in "buffer" with size "buf_len".
     * Optionally flush the dataset out to disk after destruction of this object
     */
    DataSet(void * buffer, hsize_t buf_len, bool flushOnClose = false) :
        flushOnClose(flushOnClose), file_id(-1)
    {
      refCount = new int;
      *refCount = 1;
      std::stringstream ss;
      ss << rand() << rand();
      fileName = "/" + ss.str();
      fapl = H5Pcreate(H5P_FILE_ACCESS);
      if (H5Pset_fapl_core(fapl, 16 * 1024, flushOnClose) < 0)
      {
        H5Pclose(fapl);
        fapl = 0;
        throw ICFException() << hdf5_error("Error setting file access property list for in-memory file");
      }

      if (H5Pset_file_image(fapl, buffer, buf_len) < 0)
      {
        H5Pclose(fapl);
        fapl = 0;
        throw ICFException() << hdf5_error("Error setting file image");
      }

      file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDWR, fapl);

      if (file_id < 0)
      {
        throw ICFException() << hdf5_error("Can't deserialize");
      }
    }

    /**
     * Create a new dataset in memory  with name "fileName", from a dataset serialized in "buffer" with
     * lenght "buf_len".
     * Optionally flush upon destruction of the object.
     */
    DataSet(const std::string& fileName, void * buffer, hsize_t buf_len, bool flushOnClose = false) :
        fileName(fileName), flushOnClose(flushOnClose), file_id(-1)
    {
      refCount = new int;
      *refCount = 1;
      fapl = H5Pcreate(H5P_FILE_ACCESS);
      if (H5Pset_fapl_core(fapl, 16 * 1024, flushOnClose) < 0)
      {
        H5Pclose(fapl);
        fapl = 0;
        throw ICFException() << hdf5_error("Error setting file access property list for in-memory file");
      }

      if (H5Pset_file_image(fapl, buffer, buf_len) < 0)
      {
        H5Pclose(fapl);
        fapl = 0;
        throw ICFException() << hdf5_error("Error setting file image");
      }

      file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDWR, fapl);

      if (file_id < 0)
      {
        throw ICFException() << hdf5_error("Can't deserialize");
      }

    }

    /**
     * Open or create a dataset with name "fileName", optionally in memory, with block size "blockSize"
     * (see hdf5 documentation) and optionally flush out to hard-disk on close
     */
    DataSet(const std::string& fileName, bool create = true, bool inmem = false, int blockSize = 16 * 1024,
            bool flushOnClose = false) :
        fileName(fileName), flushOnClose(flushOnClose), fapl(0), file_id(-1)
    {
      refCount = new int;
      *refCount = 1;
      fapl = H5Pcreate(H5P_FILE_ACCESS);
      if (inmem)
      {
        if (H5Pset_fapl_core(fapl, blockSize, flushOnClose) < 0)
        {
          H5Pclose(fapl);
          fapl = 0;
          throw ICFException() << hdf5_error("Error setting file access property list for in-memory file");
        }
      }
      else
      {
        if (H5Pset_fapl_stdio(fapl) < 0)
        {
          H5Pclose(fapl);
          fapl = 0;
          throw ICFException() << hdf5_error("Error setting file access property list for stdio file");
        }
      }
      std::ifstream file(fileName.c_str());
      if (create && !file.good())
      {
        file_id = H5Fcreate(fileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
      }
      else
      {
        try
        {
          file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDWR, fapl);
        }
        catch (std::exception& e)
        {
          file_id = -1;
        }
      }
      if (file_id < 0)
      {
        H5Pclose(fapl);
        fapl = 0;
        throw ICFException() << hdf5_error("Error opening file");
      }
    }

    /**
     * Copy constructor. Does not copy data!
     */
    DataSet(const DataSet<T, StorageOrder>& dataset) :
        fileName(dataset.fileName), flushOnClose(dataset.flushOnClose), fapl(dataset.fapl), file_id(dataset.file_id), refCount(
            &++*dataset.refCount)
    {

    }

    /**
     * Copy assignment operator. Does not copy data!
     */
    DataSet<T, StorageOrder>& operator =(const DataSet<T, StorageOrder> & dataset)
    {
      this->fileName = dataset.fileName;
      this->fapl = dataset.fapl;
      this->file_id = dataset.file_id;
      this->refCount = &++*dataset.refCount;
      this->flushOnClose = dataset.flushOnClose;
      return *this;
    }

    /**
     * Closes underlying hdf5file if reference count down to 0
     */
    virtual ~DataSet()
    {
      if (--*refCount == 0 && fapl != 0)
      {
        delete refCount;
        H5Pclose(fapl);
        H5Fclose(file_id);
      }
    }

    /**
     * Serialize the underlying hdf5 file
     */
    uint8_t * serialize(hsize_t& size)
    {
      if (fapl != 0)
      {
        //H5Fget_filesize(file_id, size);
        H5Fflush(file_id, H5F_SCOPE_LOCAL);
        size = H5Fget_file_image(file_id, NULL, 0);

        uint8_t * buffer = new uint8_t[size];

        if (H5Fget_file_image(file_id, (void*)buffer, size) < 0)
        {
          delete[] buffer;
          throw ICFException() << hdf5_error("Serialization failed");
        }
        return buffer;
      }
      else
      {
        throw ICFException() << hdf5_error("Object in invalid state!");
      }
      return NULL;
    }

    /**
     * get the standard dataset "/x"
     */
    MatrixPtr getX()
    {
      return getFeatureMatrix("/x");
    }

    /**
     * get the standard ground truth dataset "/y"
     */
    MatrixPtr getY()
    {
      return getFeatureMatrix("/y");
    }

    /**
     * Add feature matrix from a buffer "data". THIS COPIES THE DATA IN "data"
     */
    void setFeatureMatrix(T * data, size_t rows, size_t cols, const std::string& name)
    {
      hid_t dataset_id;
      herr_t status;

      if (!contains(name) || (dataset_id = H5Dopen(file_id, name.c_str(), H5P_DEFAULT)) < 0)
      {
        hid_t dataspace_id;
        hid_t cparams;
        hsize_t dims[2] = {rows, cols};
        hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
        hsize_t chunk_dims[2] = {1024, cols};

        dataspace_id = H5Screate_simple(2, dims, maxdims);

        cparams = H5Pcreate(H5P_DATASET_CREATE);
        status = H5Pset_chunk(cparams, 2, chunk_dims);

        dataset_id = H5Dcreate(file_id, name.c_str(), H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, cparams,
                               H5P_DEFAULT);

        H5Pclose(cparams);
        H5Sclose(dataspace_id);

        if (dataset_id < 0)
        {
          throw ICFException() << hdf5_error("Error extending dataset" + name);
        }

      }
      else //make sure dataset has the right size to store matrix data
      {
        hsize_t dims[2] = {rows, cols};
        status = H5Dextend(dataset_id, dims);
        if (status < 0)
        {
          throw ICFException() << hdf5_error("Error extending dataset size" + name);
        }
      }

      status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

      H5Dclose(dataset_id);

      if (status < 0)
      {
        throw ICFException() << hdf5_error("Error writing to dataset: " + name);
      }
    }

    /**
     * Add feature matrix. THIS COPIES THE DATA IN MATRIX
     */
    void setFeatureMatrix(MatrixPtr matrix, const std::string& name)
    {
      setFeatureMatrix(*matrix, name);
    }

    /**
     * Add feature matrix. THIS COPIES THE DATA IN MATRIX
     */
    void setFeatureMatrix(Matrix matrix, const std::string& name)
    {
      setFeatureMatrix(matrix.data(), matrix.rows(), matrix.cols(), name);
    }

    /**
     * get a feature matrix by name (hdf5 path)
     */
    MatrixPtr getFeatureMatrix(const std::string& name)
    {
      hid_t dataspace;
      hid_t dataset;

      herr_t status;

      if (!contains(name))
      {
        throw ICFException() << hdf5_error("No dataset with name " + name + " in file");
      }

      dataset = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);
      if (dataset < 0)
      {
        throw ICFException() << hdf5_error("Error opening dataset for reading:" + name);
      }

      dataspace = H5Dget_space(dataset);
      hsize_t rank = H5Sget_simple_extent_ndims(dataspace);
      hsize_t dims_out[rank];
      status = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);

      if (status < 0)
      {
        throw ICFException() << hdf5_error("Error getting dataspace");
      }

      MatrixPtr matrix(new Matrix(dims_out[0], dims_out[1]));
      status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix->data());
      H5Sclose(dataspace);
      H5Dclose(dataset);
      if (status < 0)
      {
        throw ICFException() << hdf5_error("Error reading from dataset");
      }
      return matrix;
    }

    /**
     * Same as the other methods for adding data. Takes OpenCV matrix
     */
    void addMat(const cv::Mat_<T>& mat, const std::string& name)
    {
      setFeatureMatrix(const_cast<T*>(mat[0]), mat.rows, mat.cols, name);
    }

    /**
     * Same as the other methods for adding data. Takes OpenCV matrix
     */
    void addMat(const cv::Mat& mat, const std::string& name)
    {
      setFeatureMatrix((T*)mat.ptr(0), mat.rows, mat.cols, name);
    }

    /**
     * Same as the other methods for getting data. gets OpenCV matrix
     */
    template<typename TT>
      cv::Mat_<TT> getAsCvMat_(const std::string& name)
      {
        MatrixPtr m_ptr = getFeatureMatrix(name);
        Matrix& m = *m_ptr;
        cv::Mat_<TT> mat(m.rows(), m.cols());
        for (int i = 0; i < m.rows(); i++)
        {
          for (int j = 0; j < m.cols(); j++)
          {
            mat(i, j) = m(i, j);
          }
        }
        return mat;
      }

    /**
     * Same as the other methods for getting data. gets OpenCV matrix
     */
    template<typename TT>
      cv::Mat getAsCvMat(const std::string& name)
      {
        MatrixPtr m_ptr = getFeatureMatrix(name);
        Matrix& m = *m_ptr;
        cv::Mat mat(m.rows(), m.cols(), cv::DataType < TT > ::type);
        for (int i = 0; i < m.rows(); i++)
        {
          for (int j = 0; j < m.cols(); j++)
          {
            mat.at < TT > (i, j) = m(i, j);
          }
        }
        return mat;
      }
    /**
     * create a hard hdf5 link to a dataset. Mainly usefull for temporarily aliasing data
     * before uploading to the service. The inbuilt classifiers all require the data to be
     * namend "/x" and the ground truth to be named "/y". So if your data is in "/train" and
     * the ground truth in "/train_labels" then do something like this before uploading:
     * ds.alias("/train","/x");
     * ds.alias("/train_labels","/y");
     * serverSideRepo.uploadData(ds,"name");
     *
     * Alternatively you can use the optional uploadData parameters of serverSideRepo:
     * serverSideRepo.uploadData(ds,"/train","/train_labels");
     */
    void alias(const std::string& dataset, const std::string& alias)
    {
      if (file_id != -1 && fapl != 0)
      {
        if (!contains(dataset))
        {
          throw ICFException() << hdf5_error("Dataset " + dataset + " does not exist in file");
        }
        if (contains(alias))
        {
          H5Ldelete(file_id, alias.c_str(), H5P_DEFAULT);
        }
        if (H5Lcreate_hard(file_id, dataset.c_str(), file_id, alias.c_str(), H5P_DEFAULT, H5P_DEFAULT) < 0)
        {
          throw ICFException() << hdf5_error("Could not create alias " + alias + " of " + dataset);
        }
      }
      else
      {
        throw ICFException() << hdf5_error("Dataset in illegal state");
      }
    }

    /**
     * Rename a feature matrix
     */
    void renameFeatureMatrix(const std::string& from, const std::string& to)
    {
      if (file_id != -1 && fapl != 0)
      {
        if (!contains(from))
        {
          throw ICFException() << hdf5_error("Dataset " + from + " does not exist in file");
        }
        if (contains(to))
        {
          throw ICFException() << hdf5_error("Dataset " + to + " already exists in file");
        }
        if (H5Lmove(file_id, from.c_str(), file_id, to.c_str(), H5P_DEFAULT, H5P_DEFAULT) < 0)
        {
          throw ICFException() << hdf5_error("Could not rename " + from + " to " + to);
        }
      }
      else
      {
        throw ICFException() << hdf5_error("Dataset in illegal state");
      }
    }

    /**
     * Does the dataset contain a dataset under path path?
     */
    bool contains(const std::string& path)
    {
      if (file_id != -1 && fapl != 0)
      {
        int res = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
        if (res < 0)
          throw ICFException() << hdf5_error("Can not determine existance of " + path);
        return res != 0;
      }
      else
      {
        throw ICFException() << hdf5_error("Dataset in illegal state");
      }
    }

    /**
     * Flushes the dataset to underlying data storage
     */
    bool flush()
    {
      if (file_id != -1 && fapl != 0)
      {
        H5Fflush(file_id, H5F_SCOPE_LOCAL);
      }
      else
      {
        throw ICFException() << hdf5_error("Dataset in illegal state");
      }
    }
    std::string name;
  };

template<class T, int StorageOrder>
  std::ostream& operator<<(std::ostream& out, DataSet<T, StorageOrder>& ds)
  {
    hsize_t size;
    uint8_t * buffer;
    buffer = ds.serialize(size);
    out.write((char*)buffer, size / sizeof(char));
    delete[] buffer;
    return out;
  }

template<class T, int StorageOrder>
  std::string& operator<<(std::string& string, DataSet<T, StorageOrder>& ds)
  {
    std::stringstream ss;
    ss << ds;
    string = ss.str();
    return string;
  }

typedef DataSet<double> DS;

}

#endif /* DATASET_HPP_ */
