#include <v4r/io/eigen.h>
#include <vector>
#include <iostream>

#include <ctime>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

namespace v4r
{
namespace io
{

bool
is_number(const std::string& s)
{
    std::string::const_iterator it = s.begin();
    while (it != s.end() && std::isdigit(*it)) ++it;
    return !s.empty() && it == s.end();
}


bool
writeMatrixToFile (const std::string &file, const Eigen::Matrix4f & matrix)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            out << matrix (i, j);
            if (!(i == 3 && j == 3))
                out << " ";
        }
    }
    out.close ();

    return true;
}

bool
readMatrixFromFile(const std::string &file, Eigen::Matrix4f & matrix, int padding)
{
    // check if file exists
    boost::filesystem::path path = file;
    if ( ! (boost::filesystem::exists ( path ) && boost::filesystem::is_regular_file(path)) ) {
        const std::string error_msg = "Given file path " + file + " to read matrix does not exist!";
        throw std::runtime_error (error_msg);
    }


    std::ifstream in;
    in.open (file.c_str (), std::ifstream::in);

    char linebuf[1024];
    in.getline (linebuf, 1024);
    std::string line (linebuf);
    std::vector < std::string > strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));

    for (int i = 0; i < 16; i++)
    {
        matrix (i / 4, i % 4) = static_cast<float> (atof (strs_2[padding+i].c_str ()));
    }

    return true;
}

Eigen::Matrix4f
readMatrixFromFile(const std::string &file, int padding)
{

    // check if file exists
    boost::filesystem::path path = file;
    if ( ! (boost::filesystem::exists ( path ) && boost::filesystem::is_regular_file(path)) )
        throw std::runtime_error ("Given file path to read Matrix does not exist!");

    std::ifstream in (file.c_str (), std::ifstream::in);

    char linebuf[1024];
    in.getline (linebuf, 1024);
    std::string line (linebuf);
    std::vector < std::string > strs_2;
    boost::split (strs_2, line, boost::is_any_of (" "));

    Eigen::Matrix4f matrix;
    for (int i = 0; i < 16; i++)
        matrix (i / 4, i % 4) = static_cast<float> (atof (strs_2[padding+i].c_str ()));

    return matrix;
}

bool
writeDescrToFile (const std::string &file, const Eigen::MatrixXf &matrix)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }
    size_t i ,j;

    for (i = 0; i < matrix.rows(); i++)
    {
        for (j = 0; j < matrix.cols(); j++)
        {
            out << matrix (i, j);
                out << " ";
        }
        out<<'\n';
    }
    out.close ();

    return true;
}

Eigen::MatrixXf
readDescrFromFile(const std::string &file, int padding, int rowSize)
{

    // check if file exists
    boost::filesystem::path path = file;
    if ( ! (boost::filesystem::exists ( path ) && boost::filesystem::is_regular_file(path)) )
        throw std::runtime_error ("Given file path to read Matrix does not exist!");

    std::ifstream in (file.c_str (), std::ifstream::in);

    int bufferSize = 819200;
    //int bufferSize = rowSize * 10;

    char linebuf[bufferSize];



    Eigen::MatrixXf matrix;
    matrix.resize(0,rowSize);
    int j=0;
    while(in.getline (linebuf, bufferSize)){
        int start_s=clock();
        std::string line (linebuf);
        std::vector < std::string > strs_2;
        boost::split (strs_2, line, boost::is_any_of (" "));
        matrix.conservativeResize(matrix.rows()+1,rowSize);
        for (int i = 0; i < strs_2.size()-1; i++)
            matrix (j, i) = static_cast<float> (atof (strs_2[i].c_str ()));
        j++;
        int stop_s=clock();
        std::cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << std::endl;
    }
    return matrix;
}


bool
writeLabelToFile (const std::string &file, const Eigen::VectorXi &vector)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }
    size_t i ,j;

    for (i = 0; i < vector.rows(); i++)
    {

            out << vector (i);


        out<<" ";
    }
    out.close ();

    return true;
}

Eigen::VectorXi
readLabelFromFile(const std::string &file, int padding)
{

    // check if file exists
    boost::filesystem::path path = file;
    if ( ! (boost::filesystem::exists ( path ) && boost::filesystem::is_regular_file(path)) )
        throw std::runtime_error ("Given file path to read Matrix does not exist!");

    std::ifstream in (file.c_str (), std::ifstream::in);

    char linebuf[819200];



    Eigen::VectorXi Vector;

    in.getline (linebuf, 819200);
        std::string line (linebuf);
        std::vector < std::string > strs_2;
        boost::split (strs_2, line, boost::is_any_of (" "));
        Vector.resize(strs_2.size()-1);
        for (int i = 0; i < strs_2.size()-1; i++)
            Vector ( i) = static_cast<int> (atof (strs_2[i].c_str ()));

    return Vector;
}


template<typename T>
bool
writeVectorToFile (const std::string &file, const typename std::vector<T>& val)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }

    for(size_t i=0; i<val.size(); i++)
        out << val[i] << " ";

    out.close ();

    return true;
}

bool
getCentroidFromFile (const std::string &file, Eigen::Vector3f & centroid)
{
    std::ifstream in;
    in.open (file.c_str (), std::ifstream::in);

    if (!in)
    {
        std::cout << "Cannot open file " << file.c_str () << ".\n";
        return false;
    }

    char linebuf[256];
    in.getline (linebuf, 256);
    std::string line (linebuf);
    std::vector < std::string > strs;
    boost::split (strs, line, boost::is_any_of (" "));
    centroid[0] = static_cast<float> (atof (strs[0].c_str ()));
    centroid[1] = static_cast<float> (atof (strs[1].c_str ()));
    centroid[2] = static_cast<float> (atof (strs[2].c_str ()));

    return true;
}

bool
writeFloatToFile (const std::string &file, const float value)
{
    std::ofstream out (file.c_str ());
    if (!out)
    {
        std::cout << "Cannot open file.\n";
        return false;
    }

    out << value;
    out.close ();

    return true;
}

bool
readFloatFromFile (const std::string &file, float& value)
{

    std::ifstream in;
    in.open (file.c_str (), std::ifstream::in);

    if (!in)
    {
        std::cout << "Cannot open file " << file.c_str () << ".\n";
        return false;
    }

    char linebuf[1024];
    in.getline (linebuf, 1024);
    value = static_cast<float> (atof (linebuf));

    return true;
}

template V4R_EXPORTS bool writeVectorToFile<float> (const std::string &, const typename std::vector<float>&);
template V4R_EXPORTS bool writeVectorToFile<double> (const std::string &, const typename std::vector<double>&);
template V4R_EXPORTS bool writeVectorToFile<int> (const std::string &, const typename std::vector<int>&);
template V4R_EXPORTS bool writeVectorToFile<size_t> (const std::string &, const typename std::vector<size_t>&);
}

}


