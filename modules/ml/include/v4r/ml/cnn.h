/******************************************************************************
 * Copyright (c) 2015 Thomas Faeulhammer
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

#ifndef V4R_CNN_H__
#define V4R_CNN_H__

#include </home/martin/github/caffe/include/caffe/net.hpp>
#include </home/martin/github/caffe/include/caffe/solver.hpp>
#include <opencv/cv.h>
#include <v4r/core/macros.h>
#include <pcl/point_cloud.h>

#include <boost/program_options.hpp>
#include <glog/logging.h>

namespace po = boost::program_options;

namespace v4r
{

/**
 * @brief Convolutional Neural Network based on Berkeley's Caffe Framework.
 * Extracts the image from a RGB-D point cloud by the bounding box indicated from the object indices
 * @author Thomas Faeulhammer
 * @date Nov, 2015
 */
class V4R_EXPORTS CNN
{
public:
//    typedef std::pair<std::string, float> Prediction;
    typedef boost::shared_ptr< ::v4r::CNN > Ptr;
    typedef boost::shared_ptr< ::v4r::CNN const> ConstPtr;

    class V4R_EXPORTS Parameter
    {
    public:
        int device_id_;
        std::string device_name_;
        Parameter(
                int device_id = 0,
                std::string device_name = "CPU"
                )
            :
                device_id_ (device_id),
                device_name_ (device_name)
        {}


        /**
         * @brief init parameters
         * @param command_line_arguments (according to Boost program options library)
         * @return unused parameters (given parameters that were not used in this initialization call)
         */
        std::vector<std::string>
        init(int argc, char **argv)
        {
                std::vector<std::string> arguments(argv + 1, argv + argc);
                return init(arguments);
        }

        /**
         * @brief init parameters
         * @param command_line_arguments (according to Boost program options library)
         * @return unused parameters (given parameters that were not used in this initialization call)
         */
        std::vector<std::string>
        init(const std::vector<std::string> &command_line_arguments)
        {
            po::options_description desc("CNN parameters\n=====================");
            desc.add_options()
                    ("help,h", "produce help message")
                    ("device_name", po::value<std::string>(&device_name_)->default_value(device_name_), "")
                    ("device_id", po::value<int>(&device_id_)->default_value(device_id_), "")
                    ;
            po::variables_map vm;
            po::parsed_options parsed = po::command_line_parser(command_line_arguments).options(desc).allow_unregistered().run();
            std::vector<std::string> to_pass_further = po::collect_unrecognized(parsed.options, po::include_positional);
            po::store(parsed, vm);
            if (vm.count("help")) { std::cout << desc << std::endl; to_pass_further.push_back("-h"); }
            try { po::notify(vm); }
            catch(std::exception& e) {  std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl; }
            return to_pass_further;
        }
    } param_;

    void Train(const std::string &solver_file, const std::string& init_weight_file);
    void LoadSolver(const std::string &solver_file);
    void InitSolver(const std::string &init_weight_file);

    /**
     * @brief loads the initial weights and stores them into the corresponding layers of the network, where a correspondence is
     * fulfilled if the name of layers in the initial weight file is equal to the one of the network appended by the suffix
     * @param init_weight_file
     * @param suffix
     */
    void InitSolverUsingSuffixNames(const std::string &init_weight_file, const std::string &suffix);
    void Solve();
    void SaveNetwork(const std::string &output_trained_net);
    void LoadNetwork(const std::string &model_file, const std::string &trained_model, const std::string &mean_file);

    /**
     * @brief getOutput of a specific layer (use after calling classify)
     * @param[in] name of the output layer to be returned
     * @param[out] output vector
     * @return
     */
    bool getOutput(const std::string &output_layer_name, std::vector<float> &output);

    std::vector<int> Classify(const cv::Mat& img, int N=3); /// @brief return the top N predictions.
    static bool PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs);


    void CopyTrainedLayersFromBinaryProto(const std::string &trained_filename, const std::string &suffix);
    void CopyTrainedLayersFromAndSuffixLayerName(const caffe::NetParameter& param, const std::string &suffix);


    void changeNames();

    boost::shared_ptr<caffe::Net<float> >
    getCNN( ) const
    {
        return net_;
    }

    CNN()
    {
#ifdef CPU_ONLY
        caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
        caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
    }

private:
    boost::shared_ptr<caffe::Net<float> > net_;
    boost::shared_ptr<caffe::Solver<float> > solver_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
//    std::vector<std::string> labels_;

    void SetMean(const std::string& mean_file); /// @brief Load the mean file in binaryproto format.

    std::vector<float> Predict(const cv::Mat& img);

    static std::vector<int> Argmax(const std::vector<float>& v, int N); /// @brief Return the indices of the top N values of vector v.

    /**
     * @brief Wrap the input layer of the network in separate cv::Mat objects
     * (one per channel). This way we save one memcpy operation and we
     * don't need to rely on cudaMemcpy2D. The last preprocessing
     * operation will write the separate channels directly to the input
     * layer. */
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

};

}
#endif
