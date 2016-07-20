#include <sstream>
#include <iostream>
#include <string>

#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <v4r/segmentation/pcl_segmentation_methods.h>

#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/features/esf.h>
#include <pcl/common/centroid.h>
#include <pcl/visualization/cloud_viewer.h>


#include <flann/flann.hpp>



namespace po = boost::program_options;


Eigen::MatrixXf all_model_signatures_;
boost::shared_ptr<flann::Matrix<float> > flann_data_;
std::vector<int> models;
v4r::PCLSegmenter<pcl::PointXYZRGB>::Parameter seg_param;

boost::shared_ptr<flann::Index<flann::L1<float> > > flann_index_l1_;
boost::shared_ptr<flann::Index<flann::L2<float> > > flann_index_l2_;

typedef pcl::PointXYZ PointT;

void createFLANN();
void featureMatching(const Eigen::VectorXf &, std::vector<int> &);


class V4R_EXPORTS Parameter
{
public:
    int kdtree_splits_;
    int distance_metric_; /// @brief defines the norm used for feature matching (1... L1 norm, 2... L2 norm)
    int knn_;

    Parameter(
            int kdtree_splits = 128,
            int distance_metric = 2,
            int knn = 11
            )
        : kdtree_splits_ (kdtree_splits),
          distance_metric_ (distance_metric),
          knn_ (knn)
    {}
}param_;


int main(int argc, char** argv){


    po::options_description desc("Calculates one point cloud for classification\n======================================\n**Allowed options");
    std::string path, input;
    bool visualize = false;
    desc.add_options()
            ("help,h", "produce help message")
            ("path,p", po::value<std::string>(&path)->required(), "Path to folders with desc files")
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

    std::vector<std::string> Temp;
    std::vector<std::string> objects(v4r::io::getFoldersInDirectory(path));
    std::vector<std::string> paths;
    std::vector<std::string> Files;
    std::vector<int> matched_models;
    std::vector<std::string> model_names;
    std::string fn, fp, fo;
    Eigen::MatrixXf test;
    size_t num_total, num_examples = 1, num_correct = 0;
    Eigen::MatrixXi Checkmatrix;
    Checkmatrix.resize(objects.size(),objects.size());
    Checkmatrix.setZero();
    objects.erase(find(objects.begin(),objects.end(),"svm"));
    char end = path.back();
    if(end!='/')
        path.append("/");

    for(size_t o=0;o<objects.size();o++){

        fo = path;
        fo.append(objects[o]);
        fo.append("/");
        fo.append("esf/");
        model_names.push_back(objects[o]);
        fp = fo;
        Files.clear();
        Temp = v4r::io::getFilesInDirectory(fp,".*.desc", false);
        for(size_t k=0;k<Temp.size();k++){
            Files.push_back(Temp[k]);
        }
        for(size_t j=0;j<Files.size();j++){
            fn = fp;
            fn.append(Files[j]);

            test = v4r::io::readDescrFromFile(fn,4);

            all_model_signatures_.conservativeResize(640,all_model_signatures_.cols()+1);

            all_model_signatures_.col(all_model_signatures_.cols()-1) = test.col(0);
            models.push_back(o);
        }
    }

    createFLANN();
    std::string path_check = input;
    std::vector<std::string> files = v4r::io::getFilesInDirectory(path_check.append("/pcd_binary/"),".*.pcd",false);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
    v4r::PCLSegmenter<pcl::PointXYZRGB> seg(seg_param);
    std::vector<pcl::PointIndices> found_clusters;
    for(size_t k=0; k<files.size();k++){

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

        featureMatching(query_sig.get()->getMatrixXfMap(640,640,0),matched_models);
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
        matched_model = models_name[std::distance(numbers.begin(),std::max_element(numbers.begin(),numbers.end()))];
        it = std::find(model_names.begin(),model_names.end(),matched_model);
        int matched_index = std::distance(model_names.begin(), it);
        matched_models.clear();

        std::string instream;
        std::string inputfile,item, correct_model;
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
        std::cout<<std::endl;
        Checkmatrix(correct_index,matched_index)++;
        //        for (size_t k=0; k<matched_models.size();k++)
        //            std::cout<<model_names[matched_models[k]]<<std::endl;


        std::cout<<std::endl;
        if(visualize){
            pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
            viewer.showCloud (clusterXYZ);

            while (!viewer.wasStopped ())
            {
            }

        }
    }


    std::cout<< Checkmatrix <<std::endl;

    std::cout<<"Total number of examples classified: "<< num_examples<<std::endl;

    std::cout<<"Total number of correct examples: "<<num_correct<<std::endl;

}



void createFLANN()
{

    size_t size_feat = all_model_signatures_.rows();
    flann_data_.reset( new flann::Matrix<float>(new float[all_model_signatures_.cols() * size_feat], all_model_signatures_.cols (), size_feat) );
    for (size_t i = 0; i < flann_data_->rows; i++) {
        for (size_t j = 0; j < flann_data_->cols; j++) {
            flann_data_->ptr()[i * flann_data_->cols + j] = all_model_signatures_(j,i);
        }
    }

    if(param_.distance_metric_==2)
    {
        flann_index_l2_.reset( new flann::Index<flann::L2<float> > (*flann_data_, flann::KDTreeIndexParams (4)));
        flann_index_l2_->buildIndex();
    }
    else
    {
        flann_index_l1_.reset( new flann::Index<flann::L1<float> > (*flann_data_, flann::KDTreeIndexParams (4)));
        flann_index_l1_->buildIndex();
    }
}

void featureMatching(const Eigen::VectorXf &query_sig, std::vector<int> &matched_models)
{
    if(query_sig.cols() != 1 )
        return;

    //CHECK(query_sig.cols() == 1);

    int size_feat = query_sig.rows();

    float query_data[size_feat];
    for (int f = 0; f < size_feat; f++)
        query_data[f] = query_sig(f,0);

    flann::Matrix<float> query_desc (query_data, 1, size_feat);
    flann::Matrix<float> distances (new float[param_.knn_], 1, param_.knn_);
    flann::Matrix<int> indices (new int[param_.knn_], 1, param_.knn_);

    if(param_.distance_metric_==2)
        flann_index_l2_->knnSearch (query_desc, indices, distances, param_.knn_, flann::SearchParams (param_.kdtree_splits_));
    else
        flann_index_l1_->knnSearch (query_desc, indices, distances, param_.knn_, flann::SearchParams (param_.kdtree_splits_));

    matched_models.resize(param_.knn_);
    for (int i = 0; i < param_.knn_; i++)
    {
        //if (indices[0][i]==1)// indices[0][i] ... entspricht der ID von den Trainingsdaten
        matched_models[i]=models[indices[0][i]];
        // distances[0][i] ... die Distanz von query sig zur signature von ID
    }
    delete[] indices.ptr ();
    delete[] distances.ptr ();
}
