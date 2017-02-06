//Header
#include <sstream>
#include <iostream>
#include <string>

#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <v4r/segmentation/pcl_segmentation_methods.h>
#include <v4r/common/pcl_opencv.h>
#include <v4r/ml/caffe_utils.h>
#include <v4r/ml/cnn.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/features/esf.h>
#include <pcl/common/centroid.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <sys/stat.h>

namespace po = boost::program_options;

struct stat sb;

Eigen::MatrixXf all_model_signatures_;
boost::shared_ptr<flann::Matrix<float> > flann_data_;
std::vector<int> models;
v4r::PCLSegmenter<pcl::PointXYZRGB>::Parameter seg_param;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
std::vector<pcl::PointIndices> found_clusters;

typedef pcl::PointXYZ PointT;

std::vector<std::string>::iterator it;

//Eigen::MatrixXf Checkmatrix_80u, Checkmatrix_80d, Checkmatrix_90u, Checkmatrix_90d;

int main(int argc, char** argv)
{
    //variables

    std::string main_path, train_path, class_path, solver_path, init_weight_path, ft;
    std::vector<std::string> model_names;
    v4r::CNN classifier;
    bool visualize = false;
    bool init_train = false;
    size_t trainIter = 100;
    seg_param.seg_type_ = 1;
    //parameter einlesen

    po::options_description desc("Classifies objects with ESF, AlexNet and Combination\n======================================\n**Allowed options");
    desc.add_options()
            ("help,h", "produce help message")
            ("path_main,m", po::value<std::string>(&main_path)->required(), "Path to folder with init folder for net")
            ("path_train,t", po::value<std::string>(&train_path), "Path to folder with training-images")
            ("path_class,c", po::value<std::string>(&class_path)->required(), "Path to folder with pcd to classify")
            ("initial_training,n", po::bool_switch(&init_train), "do initial Training with pics in train path")
            ("train_iter,r", po::value<size_t>(&trainIter)->required(), "Number of iterations for training")
            //seg param
            ("chop_z,z", po::value<double>(&seg_param.chop_at_z_ )->default_value(seg_param.chop_at_z_, boost::str(boost::format("%.2e") % seg_param.chop_at_z_)), "")
            ("seg_type", po::value<int>(&seg_param.seg_type_ )->default_value(seg_param.seg_type_), "")
            ("min_cluster_size", po::value<int>(&seg_param.min_cluster_size_ )->default_value(seg_param.min_cluster_size_), "")
            ("max_vertical_plane_size", po::value<int>(&seg_param.max_vertical_plane_size_ )->default_value(seg_param.max_vertical_plane_size_), "")
            ("num_plane_inliers,i", po::value<int>(&seg_param.num_plane_inliers_ )->default_value(seg_param.num_plane_inliers_), "")
            ("max_angle_plane_to_ground", po::value<double>(&seg_param.max_angle_plane_to_ground_ )->default_value(seg_param.max_angle_plane_to_ground_), "")
            ("sensor_noise_max", po::value<double>(&seg_param.sensor_noise_max_ )->default_value(seg_param.sensor_noise_max_), "")
            ("table_range_min", po::value<double>(&seg_param.table_range_min_ )->default_value(seg_param.table_range_min_), "")
            ("table_range_max", po::value<double>(&seg_param.table_range_max_ )->default_value(seg_param.table_range_max_), "")
            ("angular_threshold_deg", po::value<double>(&seg_param.angular_threshold_deg_ )->default_value(seg_param.angular_threshold_deg_), "")
            //
            ("visualize,v", po::bool_switch(&visualize), "visualize classified cluster")
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

    //check path ends
    {
    char end = main_path.back();
    if(end!='/')
        main_path.append("/");

    end = train_path.back();
    if(end!='/')
        train_path.append("/");

    end = class_path.back();
    if(end!='/')
        class_path.append("/");

    }
    //initial training
    if(init_train)
    {
        //check model files

        if(!(v4r::io::existsFile(main_path+"init/model/deploy.prototxt") && v4r::io::existsFile(main_path+"init/model/bvlc_alexnet.caffemodel")))
        {

            std::cerr << "Necessery files do not exist. Please check model folder and rerun with initial training." << std::endl;
            return 0;

        }

        //create list

        v4r::caffe_utils::createTrainList(train_path,main_path+"init/train_list.txt",main_path+"init/model_names.txt");

        //load model names

        model_names = v4r::caffe_utils::loadModelNames(main_path+"init/model_names.txt");

        //create db

        v4r::caffe_utils::createDatabase(main_path+"init/train_list.txt",main_path+"init/db/");

        //create Dir for mean file

        if (!(boost::filesystem::is_directory(main_path+"init/db/meanFile")))
            boost::filesystem::create_directory(main_path+"init/db/meanFile");

        //create mean file

        v4r::caffe_utils::computeMean(main_path+"init/db/",main_path+"init/db/meanFile/init_mean.binaryproto");

        //create Dir for snapshots

        if (!(boost::filesystem::is_directory(main_path+"init/model/caffe_train")))
            boost::filesystem::create_directory(main_path+"init/model/caffe_train");

        //set Number of training iterations

        v4r::caffe_utils::editTrainIter(main_path+"init/model/solver.prototxt",trainIter);

        //train net

        classifier.Train(main_path+"init/model/solver.prototxt",main_path+"init/model/bvlc_alexnet.caffemodel");

        //create std list

        v4r::caffe_utils::createStdList(main_path+"init/train_list.txt", main_path+"init/std_list.txt", model_names.size());

        //create std db

        v4r::caffe_utils::createDatabase(main_path+"init/std_list.txt",main_path+"init/StdDb/");



        //if (!(boost::filesystem::is_directory(main_path+"work")))
        //    boost::filesystem::create_directory(main_path+"work");


    }

    //initial load

    {
        //check if other files exist

        if(!(v4r::io::existsFile(main_path+"init/"+"std_list.txt") &&  v4r::io::existsFile(main_path+"init/model/caffe_train/caffe_train_iter_" + std::to_string(trainIter) + ".caffemodel") && v4r::io::existsFile(main_path+"init/db/meanFile/init_mean.binaryproto") && v4r::io::existsFile(main_path+"init/model/deploy.prototxt") && v4r::io::existsFile(main_path+"init/StdDb/data.mdb") && v4r::io::existsFile(main_path+"init/StdDb/lock.mdb")))
        {
            std::cerr << "Necessery files do not exist. Please check model folder and rerun with initial training." << std::endl;
            return 0;

        }

        //load weigths from model

        classifier.LoadNetwork(main_path+"init/model/deploy.prototxt",main_path+"init/model/caffe_train/caffe_train_iter_" + std::to_string(trainIter) + ".caffemodel",main_path+"init/db/meanFile/init_mean.binaryproto");







        //create work dir

        v4r::io::copyDir(main_path+"init/",main_path+"work/", true);

    }


    Eigen::MatrixXf Checkmatrix;
    Checkmatrix.resize(model_names.size(),model_names.size());
    Checkmatrix.setZero();

    Eigen::MatrixXf Checkmatrix_80u = Checkmatrix, Checkmatrix_80d = Checkmatrix, Checkmatrix_90u = Checkmatrix, Checkmatrix_90d = Checkmatrix;

    //classify data
    std::vector<std::string> files = v4r::io::getFilesInDirectory(class_path,".*.pcd",false);
    for(size_t k=0; k<files.size();k++)
    {
        //load cloud
        std::cout << "File classified: " << files[k] << std::endl;
        pcl::io::loadPCDFile(class_path+files[k], *cloud);
        //        if(found_clusters.size()==0)
        //            continue;
        //        else
        //            num_examples++;



        //segment cloud
        v4r::PCLSegmenter<pcl::PointXYZRGB> seg(seg_param);
        seg.set_input_cloud(*cloud);
        seg.do_segmentation(found_clusters);

        int min_id=-1;
        double min_centroid = std::numeric_limits<double>::max();

        for(size_t i=0; i < found_clusters.size(); i++)
        {
            typename pcl::PointCloud<PointT>::Ptr clusterXYZ (new pcl::PointCloud<PointT>());

            pcl::copyPointCloud(*cloud, found_clusters[i], *clusterXYZ);
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid (*clusterXYZ, centroid);

            //            double dist = centroid[0]*centroid[0] + centroid[1]*centroid[1] + centroid[2]*centroid[2];

            if (centroid[2] < min_centroid) {
                min_centroid = centroid[2];
                min_id = i;
            }
        }

        if (min_id >= 0) {
            std::vector<pcl::PointIndices> closest_cluster;
            closest_cluster.push_back( found_clusters[min_id] );
            found_clusters = closest_cluster;
        }

        //create pic

        cv::Mat img = v4r::ConvertPCLCloud2FixedSizeImage(*cloud, found_clusters[0].indices , 256, 256, 10, cv::Scalar(255,255,255), true);

        //classify pic

        std::vector<int> result = classifier.Classify(img,3);

        std::vector<float> prob;

        classifier.getOutput("prob", prob);



        std::string matched_model_caffe = model_names[result[0]];
        it = std::find(model_names.begin(),model_names.end(),matched_model_caffe);
        int matched_index_caffe = std::distance(model_names.begin(), it);
        std::vector<std::string> strs_2;
        boost::split (strs_2, files[k], boost::is_any_of ("-"));
        std::string correct_model_name = strs_2[0];
        it = std::find(model_names.begin(),model_names.end(),correct_model_name);
        int correct_model_index = std::distance(model_names.begin(), it);
        Checkmatrix_80d(correct_model_index,matched_index_caffe)++;

        //check probability

        if(prob[result[0]]<0.80)
        {
            Checkmatrix_80d(correct_model_index,matched_index_caffe)++;
            //std::cout <<"File: " << files[k] <<" Model clasified: " << model_names[result[0]] << std::endl;
            //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
            //cv::imshow( "Display window", img );
            //cv::waitKey(0);
        }
        else
            Checkmatrix_80u(correct_model_index,matched_index_caffe)++;

        if(prob[result[0]]<0.90)
        {
            Checkmatrix_90d(correct_model_index,matched_index_caffe)++;
            //std::cout <<"File: " << files[k] <<" Model clasified: " << model_names[result[0]] << std::endl;
            //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
            //cv::imshow( "Display window", img );
            //cv::waitKey(0);
        }
        else
            Checkmatrix_90u(correct_model_index,matched_index_caffe)++;

        {
            //train net on pic

            //check for bias

            {
                //train net on std db

            }

        }
    }

    //if (!(boost::filesystem::is_directory(main_path+"final")))
    //    boost::filesystem::create_directory(main_path+"final");

    v4r::io::copyDir(main_path+"init/",main_path+"final/",true);


    v4r::io::writeDescrToFile(main_path + "final/80u.matrix",Checkmatrix_80u);
    v4r::io::writeDescrToFile(main_path + "final/80d.matrix",Checkmatrix_80d);
    v4r::io::writeDescrToFile(main_path + "final/90u.matrix",Checkmatrix_90u);
    v4r::io::writeDescrToFile(main_path + "final/90d.matrix",Checkmatrix_90d);


}
