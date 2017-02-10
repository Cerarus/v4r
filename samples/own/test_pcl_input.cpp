#include <sstream>
#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>


int main(){


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::io::loadPCDFile("/media/martin/Diplomarbeit_Dat/TestData/pcd_binary/0.pcd", *cloud);
    std::string name;

    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (cloud);

    while (!viewer.wasStopped ())
    {

        if(name.empty()){
        std::cout << "Name the Object:";
        std::getline(std::cin,name);
        }
    }


    std::cout << "Object selected: " << name << std::endl;
}
