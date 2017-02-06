/******************************************************************************
 * Copyright (c) 2016 Thomas Faeulhammer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 ******************************************************************************/

#ifndef CAFFE_UTILS_H__
#define CAFFE_UTILS_H__

#include <iostream>
#include <vector>
#include <v4r/core/macros.h>

namespace
v4r
{
namespace caffe_utils
{

/**
 * @brief edits the number of outputs of a certain layer in a network defined by a prototxt file
 * @param filename of the prototxt file to be edited
 * @param layer of the network to be edited
 * @param the number of output this layer should have
 */
V4R_EXPORTS void editPrototxtOutput(const std::string &protoxt_filename, const std::string &output_layer_name, size_t num_outputs);

/**
 * @brief creates a lmdb database from a list of files where each line contains the filename and its associated label
 * @param filename of the list
 * @param filename of the database to be generated
 */
V4R_EXPORTS void createDatabase(const std::string &img2label_list, const std::string &db_path);

/**
 * @brief compute pixelwise mean values for the set of images in the database
 * @param database_filename
 * @param filepath to the file where to store the mean values
 */
V4R_EXPORTS void computeMean(const std::string &db_filename, const std::string &mean_file);

/**
 * @brief generate list for the createDatabase function
 * @param path to folders with pics
 * @param filepath to the file where to store the list
 * @param filepath to the file where to store class names
 */
V4R_EXPORTS void createTrainList(const std::string &root_path, const std::string &list_file, const std::string &class_name_file);

/**
 * @brief load model names from txt file
 * @param path to file
 * @return vector with model names
 */
V4R_EXPORTS std::vector<std::string> loadModelNames(const std::string &class_name_file);

/**
 * @brief generate list for the bias check
 * @param filepath to trainlist
 * @param filepath to the file where to store the std list
 * @param number of classes
 */
V4R_EXPORTS void createStdList(const std::string &train_list, const std::string &std_list, const int &class_num);

/**
 * @brief edits the number of iteration for training in the solver prototxt
 * @param filename of the prototxt file to be edited
 * @param the number of iterations
 */

V4R_EXPORTS void editTrainIter(const std::string &protoxt_filename, size_t num_iter);

/**
 * @brief returns the number of training iterations not finished yet
 * @param filename of the prototxt file
 * @return the number of iterations
 */

V4R_EXPORTS size_t getTrainIter(const std::string &protoxt_filename);

}

}

#endif
