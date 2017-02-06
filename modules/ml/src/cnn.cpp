#include <v4r/ml/cnn.h>

#include <stdio.h>  // for snprintf
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <google/protobuf/text_format.h>

#include </home/martin/github/caffe/include/caffe/blob.hpp>
#include </home/martin/github/caffe/include/caffe/common.hpp>
//#include <caffe/proto/caffe.pb.h>
#include </home/martin/github/caffe/include/caffe/util/db.hpp>
#include </home/martin/github/caffe/include/caffe/util/io.hpp>
#include </home/martin/github/caffe/include/caffe/util/upgrade_proto.hpp>
#include </home/martin/github/caffe/include/caffe/layer.hpp>
#include </home/martin/github/caffe/include/caffe/layers/memory_data_layer.hpp>
#include </home/martin/github/caffe/include/caffe/solver.hpp>
#include </home/martin/github/caffe/include/caffe/sgd_solvers.hpp>
#include </home/martin/github/caffe/include/caffe/net.hpp>

#include <v4r/common/pcl_opencv.h>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;


namespace v4r
{

void CNN::changeNames()
{
//    const std::vector<boost::shared_ptr<caffe::Layer<float> > >& layers_c = net_->layers();
//    std::vector<boost::shared_ptr<caffe::Layer<float> > >& layers = const_cast<vector<boost::shared_ptr<caffe::Layer<float> > >&>(layers_c);

//    typedef std::vector<boost::shared_ptr<caffe::Layer<float> > >::iterator l_iter;

//    for(l_iter=layerss.begin(); l_iter!=layers.end(); ++l_iter)
//    {
//        caffe::Layer<float> &l = **l_iter;

//        caffe::LayerParameter& source_layer = param.layer(i);
//    }


//    int num_source_layers = param.layer_size();
//    for (int i = 0; i < num_source_layers; i++)
//    {
//      const caffe::LayerParameter& source_layer_c = param.layer(i);
//      caffe::LayerParameter& source_layer = const_cast<caffe::LayerParameter&>(source_layer_c);
//      const std::string& source_layer_name = source_layer_c.name();

//      std::string suffix = "_rgb";
//      source_layer.set_name(source_layer_name + suffix);

//    }
  //    source_layer.set_name( source_layer_name + suffix);
}

void
CNN::CopyTrainedLayersFromBinaryProto(const std::string &trained_filename, const std::string &suffix)
{
    caffe::NetParameter param;
    caffe::ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
    CopyTrainedLayersFromAndSuffixLayerName(param,suffix);
}


void CNN::CopyTrainedLayersFromAndSuffixLayerName(const caffe::NetParameter& param, const std::string &suffix) {
    int num_source_layers = param.layer_size();
    for (int i = 0; i < num_source_layers; i++) {
        const caffe::LayerParameter& source_layer = param.layer(i);
        const std::string& source_layer_name = source_layer.name();
        const std::string modified_string = source_layer.name() + suffix;

        const std::vector<std::string>& layer_names_ = net_->layer_names();
        const std::vector<boost::shared_ptr<caffe::Layer<float> > >& layers_c = net_->layers();

        std::vector<boost::shared_ptr<caffe::Layer<float> > >& layers = const_cast<std::vector<boost::shared_ptr<caffe::Layer<float> > >&>(layers_c);

        int target_layer_id = 0;
        while (target_layer_id != layer_names_.size() &&
               layer_names_[target_layer_id] != modified_string) {
            ++target_layer_id;
        }
        if (target_layer_id == layer_names_.size()) {
            LOG(INFO) << "Ignoring source layer " << source_layer_name;
            continue;
        }
        DLOG(INFO) << "Copying source layer " << source_layer_name;

        std::vector<boost::shared_ptr<caffe::Blob<float> > >& target_blobs =
                layers[target_layer_id]->blobs();
        CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
                << "Incompatible number of blobs for layer " << source_layer_name;
        for (int j = 0; j < target_blobs.size(); ++j) {
            if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
                caffe::Blob<float> source_blob;
                const bool kReshape = true;
                source_blob.FromProto(source_layer.blobs(j), kReshape);
                LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
                           << source_layer_name << "'; shape mismatch.  Source param shape is "
                           << source_blob.shape_string() << "; target param shape is "
                           << target_blobs[j]->shape_string() << ". "
                           << "To learn this layer's parameters from scratch rather than "
                           << "copying from a saved net, rename the layer.";
            }
            const bool kReshape = false;
            target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
        }
    }
}

bool CNN::PairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs) { return lhs.first > rhs.first; }

void CNN::Train(const std::string &solver_file, const std::string &init_weight_file)
{
    LoadSolver( solver_file );
    InitSolver( init_weight_file );
    Solve();
}

void CNN::LoadSolver(const std::string &solver_file)
{
    caffe::SolverParameter solverParams;
    caffe::ReadProtoFromTextFileOrDie(solver_file, &solverParams);
    solver_.reset( new caffe::SGDSolver<float>(solverParams) );
}

void CNN::InitSolver(const std::string &init_weight_file)
{
    solver_->net()->CopyTrainedLayersFrom( init_weight_file );
}

void CNN::InitSolverUsingSuffixNames(const std::string &init_weight_file, const std::string &suffix)
{
    net_ = solver_->net();
    CopyTrainedLayersFromBinaryProto(init_weight_file, suffix);
}

void CNN::Solve()
{
    CHECK (solver_) << "Either use  methodTrain or call LoadSolver and InitSolver first!";
    solver_->Solve();
    net_ = solver_->net();
}

void CNN::SaveNetwork(const std::string &output_trained_net)
{
    CHECK (net_ && solver_);

    caffe::NetParameter net_param;
    net_->ToProto(&net_param, solver_->param().snapshot_diff());
    caffe::WriteProtoToBinaryFile(net_param, output_trained_net);
}

void CNN::LoadNetwork(const std::string &model_file, const std::string &trained_model, const std::string &mean_file)
{
    solver_.reset();

    /* Load the network. */
    net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(trained_model);

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    /* Load the binaryproto mean file. */
    SetMean(mean_file);
}

/* Return the top N predictions. */
std::vector<int> CNN::Classify(const cv::Mat& img, int N)
{
    std::vector<float> output = Predict(img);

    N = std::min<int>(net_->output_blobs()[0]->count(), N);
    return Argmax(output, N);
//    std::vector<int> maxN = Argmax(output, N);
//    std::vector<Prediction> predictions;
//    for (int i = 0; i < N; ++i) {
//        int idx = maxN[i];
//        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
//    }

//    return predictions;
}

void CNN::SetMean(const std::string& mean_file)
{
    caffe::BlobProto blob_proto;
    caffe::ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    caffe::Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; i++)
    {
        /* Extract an individual channel. */
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}


std::vector<float> CNN::Predict(const cv::Mat& img)
{
    caffe::Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}


bool CNN::getOutput(const std::string &output_layer_name, std::vector<float> &output)
{
    if( ! net_->has_blob(output_layer_name) )
        return false;

    boost::shared_ptr<Blob<float> > output_layer = net_->blob_by_name(output_layer_name);
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    output = std::vector<float>(begin, end);
    return true;
}


std::vector<int> CNN::Argmax(const std::vector<float>& v, int N)
{
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); i++)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; i++)
        result.push_back(pairs[i].second);
    return result;
}

void CNN::WrapInputLayer(std::vector<cv::Mat>* input_channels)
{
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void CNN::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}
}

