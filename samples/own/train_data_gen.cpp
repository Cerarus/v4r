#include <sstream>
#include <iostream>
#include <string>

#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <v4r/segmentation/pcl_segmentation_methods.h>
#include <v4r/common/pcl_opencv.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <opencv/cv.h>

#include <pcl/io/pcd_io.h>
#include <pcl/features/esf.h>
#include <pcl/common/centroid.h>



namespace po = boost::program_options;
typedef pcl::PointXYZ PointT;

int main(int argc, char** argv){

    v4r::PCLSegmenter<pcl::PointXYZRGB>::Parameter seg_param;
    seg_param.seg_type_=1;
    po::options_description desc("Analyses the files in input directory and creates jpegs from largest class\n======================================\n**Allowed options");
    std::string output, input;

    desc.add_options()
            ("help,h", "produce help message")
            ("path_input,p", po::value<std::string>(&input)->required(), "Path to folders with desc files")
            ("path_output,o",po::value<std::string>(&output)->required(), "Path to folders to caffe files")
            //
            ("chop_z,z", po::value<double>(&seg_param.chop_at_z_ )->default_value(seg_param.chop_at_z_, boost::str(boost::format("%.2e") % seg_param.chop_at_z_)), "")
            ("seg_type,t", po::value<int>(&seg_param.seg_type_ )->default_value(seg_param.seg_type_), "")
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
    {
        char end = input.back();
        if(end!='/')
            input.append("/");
        end = output.back();
        if(end!='/')
            output.append("/");
    }


    std::vector<std::string> files = v4r::io::getFilesInDirectory(input,".*.pcd",false);
    std::vector<std::string> model_names;
    std::vector<int> model_numbers;
    v4r::PCLSegmenter<pcl::PointXYZRGB> seg(seg_param);
    for(uint i=0; i<files.size();i++){
        std::string filename = files[i];

        std::string object = filename.substr(0,filename.find_first_of("-",0));

        std::vector<std::string>::iterator it = std::find(model_names.begin(),model_names.end(),object);

        if(it==model_names.end()){
            model_names.push_back(object);
            model_numbers.push_back(1);
        }
        else{
            int index = std::distance(model_names.begin(),it);
            model_numbers[index]++;
        }
    }
    int largest_number = *std::max_element(model_numbers.begin(),model_numbers.end());
    int selected_number = largest_number/2;
    std::string selected_class = model_names[distance(model_numbers.begin(),std::find(model_numbers.begin(),model_numbers.end(),largest_number))];

    for(uint i=0;i<model_names.size();i++){
        std::cout << "Class: " << model_names[i] << " Number: " << model_numbers[i] << std::endl;
    }
    std::cout << std::endl << "Selected Class: " << selected_class << " Number of files to copied: " << selected_number <<  std::endl;
    std::cout << "Do you want to contiue[y/n]:";
    char c=std::getchar();
//    char c='y';
    if(std::tolower(c)=='y'){
        std::vector<pcl::PointIndices> found_clusters;
        for(int j = selected_number; j<=largest_number; j++){
            std::string file = selected_class;
            file.append("-"+std::to_string(j));
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
            pcl::io::loadPCDFile(input+file+".pcd",*cloud);

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

            cv::Mat img = v4r::ConvertPCLCloud2FixedSizeImage(*cloud, found_clusters[0].indices, 256,256, 10, cv::Scalar(255,255,255), true);
            cv::imwrite(output + file + ".JPEG",img);
            std::cout << "File processed:" << file << std::endl;

        }
    }



}
