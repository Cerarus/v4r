#include <pcl/io/pcd_io.h>
#include <pcl/features/esf.h>
#include <pcl/common/centroid.h>
#include <pcl/point_types_conversion.h>

#include <sstream>
#include <iostream>
#include <string>

#include <v4r/io/filesystem.h>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <flann/flann.hpp>
//#include <flann/io/hdf5.h>

#include <v4r/segmentation/pcl_segmentation_methods.h>

boost::shared_ptr<flann::Index<flann::L1<float> > > flann_index_l1_;
boost::shared_ptr<flann::Index<flann::L2<float> > > flann_index_l2_;

boost::shared_ptr<flann::Matrix<float> > flann_data_;
Eigen::MatrixXf all_model_signatures_;
std::vector<int> models;
std::vector<std::string> modelnames;
v4r::PCLSegmenter<pcl::PointXYZRGB>::Parameter seg_param;

typedef pcl::PointXYZ PointT;


class V4R_EXPORTS Parameter
{
public:
    int kdtree_splits_;
    int distance_metric_; /// @brief defines the norm used for feature matching (1... L1 norm, 2... L2 norm)
    int knn_;

    Parameter(
            int kdtree_splits = 128,
            int distance_metric = 2,
            int knn = 3
            )
        : kdtree_splits_ (kdtree_splits),
          distance_metric_ (distance_metric),
          knn_ (knn)
    {}
}param_;



//std::vector<flann_model> flann_models_;
namespace po = boost::program_options;


void createFLANN();
void featureMatching(const Eigen::VectorXf &, std::vector<int> &);






int main(int argc, char** argv)
{


    Eigen::MatrixXf out;
    std::vector<std::string> Files;
    std::string fn, fp, fo;
    std::string ft = "/media/martin/Diplomarbeit_Dat/TestData/pcd_binary/tt_applegreen_veg0.pcd";
    std::string path;
    std::vector<int> matched_models;
    po::options_description desc("Calculates one point cloud for classification\n======================================\n**Allowed options");

    desc.add_options()
            ("help,h", "produce help message")
            ("path,p", po::value<std::string>(&path)->required(), "Path to folders with pcd files")
            //
            ("chop_z,z", po::value<double>(&seg_param.chop_at_z_ )->default_value(seg_param.chop_at_z_, boost::str(boost::format("%.2e") % seg_param.chop_at_z_)), "")
            ("seg_type", po::value<int>(&seg_param.seg_type_ )->default_value(seg_param.seg_type_), "")
            ("min_cluster_size", po::value<int>(&seg_param.min_cluster_size_ )->default_value(seg_param.min_cluster_size_), "")
            ("max_vertical_plane_size", po::value<int>(&seg_param.max_vertical_plane_size_ )->default_value(seg_param.max_vertical_plane_size_), "")
            ("num_plane_inliers", po::value<int>(&seg_param.num_plane_inliers_ )->default_value(seg_param.num_plane_inliers_), "")
            ("max_angle_plane_to_ground", po::value<double>(&seg_param.max_angle_plane_to_ground_ )->default_value(seg_param.max_angle_plane_to_ground_), "")
            ("sensor_noise_max", po::value<double>(&seg_param.sensor_noise_max_ )->default_value(seg_param.sensor_noise_max_), "")
            ("table_range_min", po::value<double>(&seg_param.table_range_min_ )->default_value(seg_param.table_range_min_), "")
            ("table_range_max", po::value<double>(&seg_param.table_range_max_ )->default_value(seg_param.table_range_max_), "")
            ("angular_threshold_deg", po::value<double>(&seg_param.angular_threshold_deg_ )->default_value(seg_param.angular_threshold_deg_), "")

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
                std::cout << "Adding File: " << fn;
                if (j==0)
                    std::cout << " " << paths.size()-i << " Folders remaining";
                std::cout<< std::endl;
                esf.compute(*temp);
                descriptor.get()->push_back((temp.get()->points)[0]);


                models.push_back(o+1);

                all_model_signatures_.conservativeResize(640,all_model_signatures_.cols()+1);

                all_model_signatures_.col(all_model_signatures_.cols()-1) = temp.get()->getMatrixXfMap(640,640,0).col(0);
            }
        }




    }

    //std::stringstream ss; ss << path << "esf.txt";
    //std::string file = ss.str();
    //std::vector<flann_model> flann_models_;
    //Eigen::MatrixXf all_model_signatures_;

    createFLANN();
    std::cout<<"Flann created"<<std::endl;

    //flann::save_to_file(flann_data_.get()->Matrix,path.append("Flann.dat"),"Flann");


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
    v4r::PCLSegmenter<pcl::PointXYZRGB> seg(seg_param);
    pcl::io::loadPCDFile(ft, *cloud);
    seg.set_input_cloud(*cloud);

    std::vector<pcl::PointIndices> found_clusters;
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




    pcl::PointCloud<pcl::PointXYZ>::Ptr clusterXYZ1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, found_clusters[0], *clusterXYZ1);
    //pcl::PointCloudXYZ

    //clusterXYZ.get()->ge
    //pcl::PointCloud<pcl::PointXYZ> cloud = renderer.renderPointcloud(visible);
    //pcl::io::savePCDFile(file,*descriptor);


    pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
    esf.setInputCloud(clusterXYZ1);
    esf.compute(*temp);

    featureMatching(temp.get()->getMatrixXfMap(640,640,0).col(0),matched_models);

    std::cout<<"Model(s) matched:"<<std::endl;
    for (size_t i=0; i<matched_models.size();i++)
        std::cout<<modelnames[matched_models[i]]<<std::endl;
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

