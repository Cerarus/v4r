
#include <sstream>
#include <iostream>
#include <string>

#include <v4r/io/filesystem.h>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <flann/flann.hpp>

#include <v4r/features/global_alexnet_cnn_estimator.h>
#include <pcl/io/pcd_io.h>

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




    Eigen::MatrixXf signatures;

    v4r::CNN_Feat_Extractor<pcl::PointXYZRGB,float>::Parameter estimator_param;
    estimator_param.init(argc, argv);
    v4r::CNN_Feat_Extractor<pcl::PointXYZRGB,float> estimator(estimator_param);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::vector<std::string> objects(v4r::io::getFoldersInDirectory(path));
    char end = path.back();
    if(end!='/')
        path.append("/");
    objects.erase(find(objects.begin(),objects.end(),"svm"));
    std::vector<std::string> paths,Files;
    std::vector<std::string> Temp;
    std::string fn, fp, fo;
    std::vector<int> models;
    std::vector<std::string> modelnames;
    std::vector<int> indices;

    for(size_t o=0;o<objects.size();o++){

        fo = path;
        fo.append(objects[o]);
        fo.append("/");
        paths.clear();
        paths = v4r::io::getFoldersInDirectory(fo);
        modelnames.push_back(objects[o]);

        for(size_t i=0;i<paths.size();i++){
            fp = fo;
            fp.append(paths[i]);
            std::cout << "Teaching File: " << fp << std::endl;



        }





    }



}

