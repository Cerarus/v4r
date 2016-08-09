#include <sstream>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>

#include <v4r/segmentation/pcl_segmentation_methods.h>
#include <v4r/ml/svmWrapper.h>

#include <pcl/io/pcd_io.h>
#include <pcl/features/esf.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/cloud_viewer.h>




namespace po = boost::program_options;
typedef pcl::PointXYZ PointT;

int main(int argc, char** argv){



    //    v4r::svmClassifier::Parameter paramSVM;
    po::options_description desc("Calculates one point cloud for classification\n======================================\n**Allowed options");
    std::string path, input;
    v4r::PCLSegmenter<pcl::PointXYZRGB>::Parameter seg_param;
    bool visualize = false;
    desc.add_options()
            ("help,h", "produce help message")
            ("path,p", po::value<std::string>(&path)->required(), "Path to folders with training files")
            ("file,f", po::value<std::string>(&input)->required(),"file to be classified")
            //
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
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
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




    std::vector<int> matched_models;
    size_t num_total, savecount, num_correct = 0;
    std::string path_check = input;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>());

    std::vector<pcl::PointIndices> found_clusters;
    std::vector<std::string> paths, Files;
    v4r::svmClassifier::Parameter paramSVM;
    paramSVM.knn_ = 1;
    v4r::svmClassifier classifier(paramSVM);

    std::string classinput, fo, fp;
    Eigen::MatrixXf emptym;
    Eigen::VectorXi emptyv;
    Eigen::MatrixXi Checkmatrix;
    Checkmatrix.resize(1,1);
    Checkmatrix(0,0)=0;
    classinput = path;
    classinput.append("/svm/Class.model");
    classifier.setInFilename(classinput);
    classifier.train(emptym,emptyv);
    char end = path_check.back();
    if(end!='/')
        path_check.append("/");
    std::vector<std::string> objects(v4r::io::getFoldersInDirectory(path_check));
    objects.erase(find(objects.begin(),objects.end(),"svm"));
    Checkmatrix.resize(objects.size(),objects.size());
    Checkmatrix.setZero();
    for(size_t o=0;o<objects.size();o++){

        fo = path_check;
        fo.append(objects[o]);
        //fo.append(objects[4]);
        fo.append("/");
        paths.clear();
        paths = v4r::io::getFoldersInDirectory(fo);
        savecount = 0;

        for(size_t i=0;i<paths.size();i++){
            fp = fo;
            fp.append(paths[i]);
            fp.append("/");
            Files.clear();
            std::vector<std::string> files = v4r::io::getFilesInDirectory(fp,".*.pcd",false);

            for(size_t k=0; k<files.size();k++){

                num_total = files.size();
                std::string ft = fp;
                ft.append(files[k]);
                //ft.append("tt_hmrr89.pcd");
                std::cout<<"File classified: "<<files[k]<<std::endl;



                pcl::io::loadPCDFile(ft, *cloud);
                pcl::PointCloud<pcl::PointXYZ>::Ptr object_tmp(new pcl::PointCloud<pcl::PointXYZ>);

                object_tmp->points.resize(cloud->points.size());
                size_t kept=0;
                for(size_t pt=0; pt<cloud->points.size(); pt++)
                {
                    if( pcl::isFinite(cloud->points[pt] ) )
                    {
                        object_tmp->points[kept] = cloud->points[pt];
                        kept++;
                    }
                }
                object_tmp->points.resize(kept);
                object_tmp->width = kept;
                object_tmp->height = 1;

                pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
                esf.setInputCloud(object_tmp);
                pcl::PointCloud<pcl::ESFSignature640>::Ptr query_sig(new pcl::PointCloud<pcl::ESFSignature640>);
                esf.compute(*query_sig);

                Eigen::MatrixXf query = query_sig.get()->getMatrixXfMap();
                Eigen::MatrixXf predicted_label;

                int temp;
                predicted_label.resize(640,3);
                predicted_label(0,0) = 1;
                classifier.predict(query.transpose(), predicted_label);
                //std::cout<<predicted_label<<std::endl;
                matched_models.clear();
                for (size_t c=0; c<predicted_label.cols();c++)
                {
                    temp = 0;
                    for (size_t n=0; n < predicted_label.rows();n++)
                        temp += predicted_label(n,c);
                    temp/=predicted_label.rows();
                    matched_models.push_back(temp);

                }

                std::vector<std::string> model_names = objects;
                std::vector<std::string> models_name;
                std::string model,matched_model;
                std::vector<int> numbers;
                std::vector<std::string>::iterator it;
                for(size_t t=0; t<matched_models.size(); t++){
                    model = model_names[matched_models[t]];
                    if(std::find(models_name.begin(),models_name.end(),model)==models_name.end()){
                        models_name.push_back(model);
                        numbers.push_back(0);
                    }
                    it = std::find(models_name.begin(),models_name.end(),model);
                    numbers[std::distance(models_name.begin(),it)]++;

                }
                //int index = std::distance(numbers.begin(),std::max_element(numbers.begin(),numbers.end()));
                matched_model = models_name[std::distance(numbers.begin(),std::max_element(numbers.begin(),numbers.end()))];
                it = std::find(model_names.begin(),model_names.end(),matched_model);
                int index = std::distance(model_names.begin(), it);
                std::cout<< "Objects Should: " << matched_model << ". Index: " << index <<  std::endl;

                Checkmatrix(o,index)++;
                std::cout<< Checkmatrix << std::endl;
            }
        }
    }


}
