#include <boost/program_options.hpp>
#include <v4r/io/filesystem.h>
#include <iostream>
#include <fstream>
namespace po = boost::program_options;



int main(int argc, char** argv){


    po::options_description desc("Copies and renames files\n======================================\n**Allowed options");
    std::string path, input;

    desc.add_options()
            ("help,h", "produce help message")
            ("input path,p", po::value<std::string>(&input)->required(), "Path to folders with files to train on")
            ("output path,o", po::value<std::string>(&path)->required(), "Path to .txt file")
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

//    char end = path.back();
//    if(end!='/')
//        path.append("/");

    char end = input.back();
    if(end!='/')
        input.append("/");

    std::string load_path = input, inputfile, file_name;
    std::ofstream outSt;
    outSt.open(path,std::ios::out);
    std::vector<std::string> objects = v4r::io::getFoldersInDirectory(load_path);
    objects.erase(find(objects.begin(),objects.end(),"svm"));

    for(int i=0; i<objects.size(); i++){
        load_path = input;
        load_path.append(objects[i]);
        load_path.append("/");
        std::vector<std::string> files = v4r::io::getFilesInDirectory(load_path,".*.JPEG",false);
        std::cout<<"Converting Folder: " << objects[i] << std::endl;
        for(int j=0; j<files.size();j++){
//        for(int j=0; j<4;j++){
            load_path.append(files[j]);
            file_name = objects[i];
            file_name.append("/");
            file_name.append(files[j]);
            file_name.append(" ");
            file_name.append(std::to_string(i));

            outSt<<file_name;
            outSt<<std::endl;

        }




    }

outSt.close();







}
