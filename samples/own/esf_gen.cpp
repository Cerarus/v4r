#include <sstream>
#include <iostream>
#include <string>

#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <v4r/segmentation/pcl_segmentation_methods.h>

#include <pcl/features/esf.h>
#include <pcl/io/pcd_io.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

namespace po = boost::program_options;


int main(int argc, char** argv)
{



    std::vector<std::string> Files;
    std::string fn, fp, fo;
    std::string path;
    size_t savecount;
    po::options_description desc("Calculates one point cloud for classification\n======================================\n**Allowed options");

    desc.add_options()
            ("help,h", "produce help message")
            ("path,p", po::value<std::string>(&path)->required(), "Path to folders with pcd files")
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



    // Cloud for storing the object.
    pcl::PointCloud<pcl::PointXYZ>::Ptr object(new pcl::PointCloud<pcl::PointXYZ>);
    //pcl::io::loadPCDFile(fn, *object);
    // Object for storing the ESF descriptor.
    pcl::PointCloud<pcl::ESFSignature640>::Ptr temp(new pcl::PointCloud<pcl::ESFSignature640>);
    pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptor(new pcl::PointCloud<pcl::ESFSignature640>);
    std::vector<std::string> objects(v4r::io::getFoldersInDirectory(path));
    std::vector<std::string> paths;
    std::vector<std::string> Temp;
    Eigen::MatrixXf save;
    std::string savepath;
    for(size_t o=0;o<objects.size();o++){

        fo = path;
        fo.append(objects[o]);
        fo.append("/");
        paths.clear();
        paths = v4r::io::getFoldersInDirectory(fo);
        savecount = 0;
        for(size_t i=0;i<paths.size();i++){
            fp = fo;
            fp.append(paths[i]);
            fp.append("/");
            Files.clear();
            Temp = v4r::io::getFilesInDirectory(fp,".*.pcd", false);
            for(size_t k=0;k<Temp.size();k++){
                Files.push_back(Temp[k]);
            }
            for(size_t j=0;j<Files.size();j++){
                fn = fp;
                fn.append(Files[j]);



                // Note: you should have performed preprocessing to cluster out the object
                // from the cloud, and save it to this individual file.

                // Read a PCD file from disk.
                if (pcl::io::loadPCDFile<pcl::PointXYZ>(fn, *object) != 0)
                {
                    return -1;
                }

                pcl::PointCloud<pcl::PointXYZ>::Ptr object_tmp(new pcl::PointCloud<pcl::PointXYZ>);

                object_tmp->points.resize(object->points.size());
                size_t kept=0;
                for(size_t pt=0; pt<object->points.size(); pt++)
                {
                    if( pcl::isFinite(object->points[pt] ) )
                    {
                        object_tmp->points[kept] = object->points[pt];
                        kept++;
                    }
                }
                object_tmp->points.resize(kept);
                object_tmp->width = kept;
                object_tmp->height = 1;

                std::cout << "Estimating esf on " << kept << " points. " << std::endl;

                // ESF estimation object.
                pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
                esf.setInputCloud(object_tmp);
                std::cout << "Calculating Descriptor of File: " << fn;
                if (j==0)
                    std::cout << " " << paths.size()-i << " Folders remaining";
                std::cout<< std::endl;
                esf.compute(*temp);

                save = temp.get()->getMatrixXfMap();
                savepath = fo;
                v4r::io::createDirIfNotExist(savepath.append("esf/"));
                savepath.append(boost::lexical_cast<std::string>(savecount));
                savepath.append(".desc");

                if(v4r::io::writeDescrToFile(savepath,save))
                    std::cout<<"Saved"<<std::endl;

                savecount++;


            }
        }
    }
}
