#include <v4r/io/filesystem.h>
#include <boost/program_options.hpp>
#include <pcl/io/pcd_io.h>
#include <string>
#include <algorithm>
#include <iostream>

namespace po = boost::program_options;

int main(int argc, char** argv){


    po::options_description desc("Copies and renames files\n======================================\n**Allowed options");
    std::string path, input, pc;
    bool visualize = false;
    desc.add_options()
            ("help,h", "produce help message")
            ("input path,p", po::value<std::string>(&input)->required(), "Path to folders with files to copy")
            ("output path,o", po::value<std::string>(&path)->required(), "Path to target folder")
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


    std::vector<int> model_numbers;
    std::vector<std::string> model_names;

    char end = path.back();
    if(end!='/')
        path.append("/");

    end = input.back();
    if(end!='/')
        input.append("/");

    std::string load_path = input, inputfile, file_name, save_number_string;
    load_path.append("pcd_binary/");
    std::vector<std::string> files = v4r::io::getFilesInDirectory(load_path,".*.pcd",false);

    for(int k=0; k<files.size(); k++){



        std::string instream;
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
        std::vector<std::string>::iterator it;

        std::vector<std::string> elements{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

        if(elements.size()>5){
            file_name = elements[elements.size()-2];
            file_name.append("_");
            file_name.append(elements[elements.size()-1]);
        }
        else
            file_name = elements[elements.size()-1];


        it = std::find(model_names.begin(),model_names.end(),file_name);
        if(it != model_names.end()){
            size_t index = std::distance(model_names.begin(),it);
            model_numbers[index]++;
            save_number_string = std::to_string( model_numbers[index]);
        }
        else {
            model_names.push_back(file_name);
            model_numbers.push_back(1);
            save_number_string = "1";
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>());



        pcl::io::loadPCDFile(load_path+files[k],*cloud);

        pcl::io::savePCDFileBinary(path+file_name+"-"+save_number_string+".pcd",*cloud);

        if(k%101 ==0 ){
            for(int i=0;i<model_names.size();i++){
                std::cout << "Model: " << model_names[i] << " Number: " <<std::to_string(model_numbers[i])<< std::endl;
            }
            std::cout<<std::endl;
        }
    }

    for(int i=0;i<model_names.size();i++){
        std::cout << "Model: " << model_names[i] << " Number: " <<std::to_string(model_numbers[i])<< std::endl;

    }

}
