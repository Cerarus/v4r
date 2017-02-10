#include <sstream>
#include <iostream>
#include <string>

#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/gil/extension/io/jpeg_dynamic_io.hpp>
#include <v4r/common/pcl_opencv.h>

#include <v4r/features/global_alexnet_cnn_estimator.h>




typedef boost::mpl::vector<boost::gil::gray8_image_t, boost::gil::gray16_image_t, boost::gil::rgb8_image_t, boost::gil::rgb16_image_t> my_img_types;
namespace po = boost::program_options;

int main(int argc, char** argv)
{
    std::string path;
    po::options_description desc("Calculates one point cloud for classification\n======================================\n**Allowed options");

    desc.add_options()
            ("help,h", "produce help message")
            ("path,p", po::value<std::string>(&path), "Path to folders with pics")

            ;

    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser(argc,argv).options(desc).allow_unregistered().run();
    po::store(parsed, vm);
    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return false;
    }

    try
    {
        po::notify(vm);
    }
    catch(std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
        return false;
    }

//    std::string pretrained_binary_proto = "/home/martin/github/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel";
//    std::string feature_extraction_proto = "/home/martin/github/caffe/models/bvlc_alexnet/deploy.prototxt";
    std::string pretrained_binary_proto = "/home/martin/github/caffe/models/bvlc_alexnet_own/caffe_alexnet_own_train_iter_81.caffemodel";
    std::string feature_extraction_proto = "/home/martin/github/caffe/models/bvlc_alexnet_own/deploy.prototxt";
    std::string mean_file = "/home/martin/github/caffe/data/ilsvrc12/imagenet_mean.binaryproto";
    std::vector<std::string> extract_feature_blob_names;
    extract_feature_blob_names.push_back("fc8_own");








    v4r::CNN_Feat_Extractor<pcl::PointXYZRGB,float>::Parameter estimator_param;
    //estimator_param.init(argc, argv);
    v4r::CNN_Feat_Extractor<pcl::PointXYZRGB,float>::Ptr estimator;
    estimator.reset(new v4r::CNN_Feat_Extractor<pcl::PointXYZRGB, float>(estimator_param));
    //estimator->setExtractFeatureBlobNames(extract_feature_blob_names);
    estimator->setFeatureExtractionProto(feature_extraction_proto);
    estimator->setPretrainedBinaryProto(pretrained_binary_proto);
    estimator->setMeanFile(mean_file);




    std::string fp="/media/martin/Diplomarbeit_Dat/imagenet_sub_resize/apple/n07739125_12.JPEG";
    cv::Mat image = cv::imread(fp);
    Eigen::MatrixXf signature;
    if(image.rows != 256 || image.cols != 256)
        cv::resize( image, image, cv::Size(256, 256));
    estimator->compute(image,signature);
//    all_model_signatures_.row(old_rows+i) = signature.row(0);
//    models.conservativeResize(models.rows()+1);
//    models(models.rows()-1) = o;



    std::cout << signature << std::endl;

}
