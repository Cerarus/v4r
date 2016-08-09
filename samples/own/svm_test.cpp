#include <sstream>
#include <iostream>
#include <string>

#include <v4r/ml/svmWrapper.h>

#include <v4r/io/eigen.h>




int main(){
    v4r::svmClassifier::Parameter paramSVM;
    paramSVM.knn_=1;
    v4r::svmClassifier SVMC(paramSVM);
    Eigen::MatrixXf Data, query;
    Eigen::MatrixXi predicted_label;
    Eigen::VectorXi Label;

    query.resize(1,2);
    Data.resize(11,2);
    Label.resize(11,1);
    //predicted_label.resize(2,1);


    Data(0,0) = 0;
    Data(0,1) = 0;

    Data(1,0) = 1;
    Data(1,1) = 0;

    Data(2,0) = -1;
    Data(2,1) = 0;

    Data(3,0) = 0;
    Data(3,1) = 1;

    Data(4,0) = 0;
    Data(4,1) = 2;

    Data(5,0) = 1;
    Data(5,1) = 1;

    Data(6,0) = -1;
    Data(6,1) = 1;


    Data(7,0) = 0;
    Data(7,1) = -1;

    Data(8,0) = 0;
    Data(8,1) = -2;

    Data(9,0) = 1;
    Data(9,1) = -1;

    Data(10,0) = -1;
    Data(10,1) = -1;

    query(0,0) = 100;
    query(0,1) = 0.5;


    Label(0)=0;
    Label(1)=0;
    Label(2)=0;
    Label(3)=1;
    Label(4)=1;
    Label(5)=1;
    Label(6)=1;
    Label(7)=2;
    Label(8)=2;
    Label(9)=2;
    Label(10)=2;

    SVMC.trainSVM(Data,Label);

    SVMC.predict(query, predicted_label);

    std::cout << Data<< std::endl << std::endl << Label << std::endl << std::endl << query << std::endl << std::endl << predicted_label << std::endl;



}
