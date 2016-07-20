
#include <sstream>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include <v4r/io/filesystem.h>
#include <v4r/io/eigen.h>
#include <v4r/ml/svmWrapper.h>

//#include <v4r/own/svmWrapper.h>

#include <pcl/io/pcd_io.h>
#include <pcl/features/esf.h>



namespace po = boost::program_options;


int main(int argc, char** argv){



//    v4r::svmClassifier::Parameter paramSVM;
    po::options_description desc("Calculates one point cloud for classification\n======================================\n**Allowed options");
    std::string path;

    desc.add_options()
            ("help,h", "produce help message")
            ("path,p", po::value<std::string>(&path)->required(), "Path to folders with desc files")
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

    std::vector<std::string> Temp;
    std::vector<std::string> objects(v4r::io::getFoldersInDirectory(path));
    std::vector<std::string> paths;
    std::vector<std::string> Files;
    std::vector<int> matched_models;
    std::vector<std::string> model_names;
    std::string fn, fp, fo, fs;
    Eigen::MatrixXf test;
    //size_t num_total, num_examples = 1, num_correct = 0;
    Eigen::MatrixXf all_model_signatures_;
    //Eigen::vector<std::string> models;
    Eigen::VectorXi models;
    v4r::svmClassifier::Parameter paramSVM;
    paramSVM.knn_=1;
    paramSVM.do_cross_validation_=1;
    v4r::svmClassifier classifier(paramSVM);
    char end = path.back();
    if(end!='/')
        path.append("/");
    objects.erase(find(objects.begin(),objects.end(),"svm"));
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
            models.conservativeResize(models.rows()+1);
            models(models.rows()-1) = o;

        }

    }
    fs = path;
    fs.append("/svm/Class.model");
    //classifier.setOutFilename(fs);

    //test.resize(all_model_signatures_.cols(),all_model_signatures_.rows());
    test = all_model_signatures_.transpose();
    classifier.trainSVM(test, models);

    classifier.saveModel(fs);
    for(size_t k=0; k<objects.size();k++)
        std::cout << objects[k] << std::endl;

}
