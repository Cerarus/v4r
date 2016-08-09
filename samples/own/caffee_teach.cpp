
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

#include <v4r/ml/svmWrapper.h>
#include <v4r/features/global_alexnet_cnn_estimator.h>
//#include <v4r/own/CNN.h>
//#include <v4r/features/global_estimator.h>
#include <pcl/io/pcd_io.h>
//#include <opencv2/opencv.hpp>

typedef boost::mpl::vector<boost::gil::gray8_image_t, boost::gil::gray16_image_t, boost::gil::rgb8_image_t, boost::gil::rgb16_image_t> my_img_types;
namespace po = boost::program_options;

int main(int argc, char** argv)
{
    std::string path;
    po::options_description desc("Calculates one point cloud for classification\n======================================\n**Allowed options");

    desc.add_options()
            ("help,h", "produce help message")
            ("path,p", po::value<std::string>(&path)->required(), "Path to folders with pics")

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
    std::string pretrained_binary_proto = "/home/martin/github/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";
    std::string feature_extraction_proto = "/home/martin/github/caffe/models/bvlc_reference_caffenet/deploy.prototxt";
    std::string mean_file = "/home/martin/github/caffe/data/ilsvrc12/imagenet_mean.binaryproto";
    //    std::vector<std::string> extract_feature_blob_names;
    //    extract_feature_blob_names.push_back("fc7");



    Eigen::MatrixXf all_model_signatures_, test;

    v4r::CNN_Feat_Extractor<pcl::PointXYZRGB,float>::Parameter estimator_param;
    //estimator_param.init(argc, argv);
    v4r::CNN_Feat_Extractor<pcl::PointXYZRGB,float>::Ptr estimator;
    //v4r::CNN_Feat_Extractor<pcl::PointXYZRGB,float> estimator;
    estimator.reset(new v4r::CNN_Feat_Extractor<pcl::PointXYZRGB, float>(estimator_param));
    //estimator.reset();
    //estimator->setExtractFeatureBlobNames(extract_feature_blob_names);
    estimator->setFeatureExtractionProto(feature_extraction_proto);
    estimator->setPretrainedBinaryProto(pretrained_binary_proto);
    estimator->setMeanFile(mean_file);
    //estimator->init();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::PointCloud<pcl::PointXYZRGB> test(new pcl::PointCloud<pcl::PointXYZRGB>);
    char end = path.back();
    if(end!='/')
        path.append("/");
    std::vector<std::string> objects(v4r::io::getFoldersInDirectory(path));

    objects.erase(find(objects.begin(),objects.end(),"svm"));
    std::vector<std::string> paths,Files;
    std::vector<std::string> Temp;
    std::string fs, fp, fo;
    Eigen::VectorXi models;
    std::vector<std::string> modelnames;
    std::vector<int> indices;


    v4r::svmClassifier::Parameter paramSVM;
    paramSVM.knn_=1;
    paramSVM.do_cross_validation_=0;
    v4r::svmClassifier classifier(paramSVM);

    if(!(v4r::io::existsFile(path+"svm/Signatures.txt")&&v4r::io::existsFile(path+"svm/Labels.txt"))){
        for(size_t o=0;o<objects.size();o++){
        //for(size_t o=0;o<1;o++){
            fo = path;
            fo.append(objects[o]);
            fo.append("/");
            paths.clear();
            paths = v4r::io::getFilesInDirectory(fo,".*.JPEG",false);
            modelnames.push_back(objects[o]);
            int old_rows = all_model_signatures_.rows();
            all_model_signatures_.conservativeResize(all_model_signatures_.rows()+paths.size(),4096);
            for(size_t i=0;i<paths.size();i++){
//          for(size_t i=0;i<3;i++){
                fp = fo;
                fp.append(paths[i]);
                std::cout << "Teaching File: " << fp << std::endl;


                //            int rows = image.rows;
                //            int cols = image.cols;

                //            int a,b;
                //            if(rows>256)
                //                a = floor(rows/2)-128;
                //            else
                //                a=0;
                //            if(cols<256)
                //                b = floor(cols/2)-128;
                //            else
                //                b=0;
                //            if(rows<256||cols<256)
                //                continue;
                //cv::Rect r(b,a,256,256);
                //            cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
                //            cv::imshow( "Display window", image );
                //            cv::waitKey();
                //estimator.setIm(image);
                //estimator.setIndices(indices);
                //estimator->compute(image,signature);
                //test = signature.transpose();
                cv::Mat image = cv::imread(fp);
                Eigen::MatrixXf signature;
                if(image.rows != 256 || image.cols != 256)
                    cv::resize( image, image, cv::Size(256, 256));
                //bool success = estimator->computeCNN(signature);
                estimator->compute(image,signature);
                //all_model_signatures_.conservativeResize(all_model_signatures_.rows()+1,4096);


                all_model_signatures_.row(old_rows+i) = signature.row(0);
                models.conservativeResize(models.rows()+1);
                models(models.rows()-1) = o;


                // std::cout<<all_model_signatures_ <<std::endl;
                std::cout<<"ok" <<std::endl;

            }





        }

        std::string f_file = path;
        v4r::io::writeDescrToFile(f_file.append("svm/Signatures.txt"),all_model_signatures_);
        f_file = path;
        v4r::io::writeLabelToFile(f_file.append("svm/Labels.txt"),models);
    }
    else{
        all_model_signatures_ = v4r::io::readDescrFromFile(path+"svm/Signatures.txt",0,4096);
        models = v4r::io::readLabelFromFile(path+"svm/Labels.txt",0);
        std::cout << "Loading Descriptors and Labels from " << path << "svm." <<std::endl;
    }



    fs = path;
    fs.append("svm/Class.model");
    classifier.setOutFilename(fs);

    //test.resize(all_model_signatures_.cols(),all_model_signatures_.rows());
    //test = all_model_signatures_.transpose();
    //classifier.shuffleTrainingData(all_model_signatures_, models);
    classifier.param_.do_cross_validation_=0;
    int testC = classifier.param_.svm_.C;
    classifier.param_.svm_.probability=1;
    classifier.setNumClasses(objects.size());
    classifier.train(all_model_signatures_, models);

    classifier.saveModel(fs);








}
