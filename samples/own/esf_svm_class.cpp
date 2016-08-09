#include <sstream>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
//#include <v4r/ml/svmWrapper.h>
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
            ("visualize,v", po::bool_switch(&visualize), "visualize classified cluster")
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
    size_t num_total,num_examples = 0, num_correct = 0;
    std::string path_check = input;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
    v4r::PCLSegmenter<pcl::PointXYZRGB> seg(seg_param);
    std::vector<pcl::PointIndices> found_clusters;
    std::vector<std::string> model_names;
    v4r::svmClassifier::Parameter paramSVM;
    paramSVM.knn_ = 1;
    v4r::svmClassifier classifier(paramSVM);
    std::vector<std::string> objects;
    std::string classinput;
    Eigen::MatrixXf emptym;
    Eigen::VectorXi emptyv;
    int filesToSkip = 0;
    char end = path.back();
    if(end!='/')
        path.append("/");
    objects = v4r::io::getFoldersInDirectory(path);
    objects.erase(find(objects.begin(),objects.end(),"svm"));
    classinput = path;
    classinput.append("svm/Class.model");
    classifier.setInFilename(classinput);
    classifier.train(emptym,emptyv);
    end = path_check.back();
    if(end!='/')
        path_check.append("/");
    Eigen::MatrixXi Checkmatrix;
    Checkmatrix.resize(objects.size(),objects.size());
    Checkmatrix.setZero();
    std::vector<std::string> files = v4r::io::getFilesInDirectory(path_check.append("pcd_binary/"),".*.pcd",false);
    std::cout << filesToSkip << " files skipped!" << std::endl;
    for(size_t k=filesToSkip; k<files.size();k++){
        num_examples++;
        num_total = files.size();
        std::string ft = path_check;
        ft.append(files[k]);
        //ft.append("tt_hmrr89.pcd");
        std::cout<<"File classified: "<<files[k]<<std::endl;



        pcl::io::loadPCDFile(ft, *cloud);
        //        if(found_clusters.size()==0)
        //            continue;
        //        else
        //            num_examples++;

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




        pcl::PointCloud<pcl::PointXYZ>::Ptr clusterXYZ(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*cloud, found_clusters[0], *clusterXYZ);


        pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
        esf.setInputCloud(clusterXYZ);
        pcl::PointCloud<pcl::ESFSignature640>::Ptr query_sig(new pcl::PointCloud<pcl::ESFSignature640>);
        esf.compute(*query_sig);

        Eigen::MatrixXf query = query_sig.get()->getMatrixXfMap();
        Eigen::MatrixXf predicted_label;

        int temp;
        predicted_label.resize(640,3);
        predicted_label(0,0) = 1;
        classifier.predict(query.transpose(), predicted_label);
        //std::cout << predicted_label << std::endl;
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

        int test = std::distance(numbers.begin(),std::max_element(numbers.begin(),numbers.end()));
        matched_model = models_name[std::distance(numbers.begin(),std::max_element(numbers.begin(),numbers.end()))];

        it = std::find(model_names.begin(),model_names.end(),matched_model);
        int matched_index = std::distance(model_names.begin(), it);
        matched_models.clear();
        std::string instream;
        std::string inputfile,item;
        inputfile = input;
        inputfile.append("/annotation/");
        inputfile.append(files[k]);
        inputfile.erase(inputfile.end()-3,inputfile.end());
        inputfile.append("anno");
        std::ifstream myfile (inputfile);

        if (myfile.is_open())
        {
            getline (myfile,instream);
        }

        std::istringstream iss(instream);


        std::vector<std::string> elements{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
        std::string correct_model;
        if(elements.size()>5){
        correct_model = elements[elements.size()-2];
        correct_model.append("_");
        correct_model.append(elements[elements.size()-1]);
        }
        else
            correct_model = elements[elements.size()-1];


        it = std::find(model_names.begin(),model_names.end(),correct_model);
        int correct_index = std::distance(model_names.begin(), it);

        if(!correct_model.compare(matched_model)){
            num_correct++;
            std::cout<< "Objects Should: " << correct_model << ". Classified: " << matched_model <<". CORRECT" <<  std::endl;
        }
        else
            std::cout<< "Objects Should: " << correct_model << ". Classified: " << matched_model <<  std::endl;
        Checkmatrix(correct_index,matched_index)++;

        std::cout<<std::endl;

        //        for (size_t k=0; k<matched_models.size();k++)
        //            std::cout<<model_names[matched_models[k]]<<std::endl;

        /*std::cout<<"Total number of examples classified: "<< num_examples<<std::endl;

        std::cout<<"Total number of correct examples: "<<num_correct<<std::endl;*/
        std::cout<<std::endl;
        if(visualize){
            pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
            viewer.showCloud (clusterXYZ);

            while (!viewer.wasStopped ())
            {
            }

        }
    }

    std::cout << "Total number of examples classified: "<< num_examples<<std::endl;

    std::cout << "Total number of correct examples: " << num_correct<<std::endl;
    std::cout << std::endl;
    std::cout << Checkmatrix << std::endl;
    for (int k=0;k<objects.size();k++)
        std::cout << objects[k] << std::endl;

}
