#include <v4r/ml/caffe_utils.h>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include </home/martin/github/caffe/include/caffe/util/db.hpp>
#include </home/martin/github/caffe/include/caffe/util/io.hpp>
#include </home/martin/github/caffe/include/caffe/util/rng.hpp>

#include <v4r/common/pcl_opencv.h>
#include <v4r/io/filesystem.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <boost/scoped_ptr.hpp>
#include <algorithm>
#include <random>
#include <chrono>       // std::chrono::system_clock

#include <sstream>
#include <iostream>
#include <string>


DEFINE_bool(gray, false,
            "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, true,
            "Randomly shuffle the order of images and their labels");
//DEFINE_string(backend, "lmdb", "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
            "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
            "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
              "Optional: What type should we encode the image as ('png','jpg',...).");

DEFINE_string(backend, "lmdb", "The backend {lmdb, leveldb} for storing the result");


namespace bf = boost::filesystem;

namespace v4r
{
namespace caffe_utils
{

V4R_EXPORTS void editPrototxtOutput(const std::string &protoxt_filename, const std::string &output_layer_name, size_t num_outputs)
{
    std::ifstream infile(protoxt_filename.c_str());
    const std::string tmp_file_name = protoxt_filename + "~~";
    std::ofstream outfile(tmp_file_name);

    std::string line;
    while (std::getline(infile, line)) {
        outfile << line << std::endl;
        if ( line.find( output_layer_name ) != std::string::npos)
        {
            while (std::getline(infile, line))
            {
                size_t pos = line.find( "num_output:" );
                if (  pos != std::string::npos)
                {
                    for(size_t i=0; i<pos; i++)
                        outfile << " " ;

                    outfile << "num_output: " << num_outputs << std::endl;
                }
                else
                    outfile << line << std::endl;
            }
        }
    }
    infile.close();
    outfile.close();

    bf::copy_file(tmp_file_name, protoxt_filename, bf::copy_option::overwrite_if_exists);
    bf::remove(tmp_file_name);
}


V4R_EXPORTS void createDatabase(const std::string &img2label_list, const std::string &db_path)
{
    // Print output to stderr (while still logging)
    FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
    namespace gflags = google;
#endif

    const bool is_color = !FLAGS_gray;
    const bool check_size = FLAGS_check_size;
    const bool encoded = FLAGS_encoded;
    const std::string encode_type = FLAGS_encode_type;

    std::ifstream infile(img2label_list.c_str());
    std::vector<std::pair<std::string, int> > lines;
    std::string line;
    size_t pos;
    int label;
    while (std::getline(infile, line)) {
        pos = line.find_last_of(' ');
        label = atoi(line.substr(pos + 1).c_str());
        lines.push_back(std::make_pair(line.substr(0, pos), label));
    }
    if (FLAGS_shuffle) {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(lines.begin(), lines.end(), std::default_random_engine(seed));
    }
    LOG(INFO) << "A total of " << lines.size() << " images.";

    if (encode_type.size() && !encoded)
        LOG(INFO) << "encode_type specified, assuming encoded=true.";

    int resize_height = std::max<int>(0, FLAGS_resize_height);
    int resize_width = std::max<int>(0, FLAGS_resize_width);


    if(v4r::io::existsFolder(db_path))
    {
        std::cerr << "File " << db_path << " already exists. Overwriting..." << std::endl;
        boost::filesystem::remove_all(db_path);
    }

    // Create new DB
    boost::scoped_ptr<caffe::db::DB> db( caffe::db::GetDB("lmdb") );
    db->Open( db_path, caffe::db::NEW );
    boost::scoped_ptr<caffe::db::Transaction> txn( db->NewTransaction() );

    // Storing to db
    caffe::Datum datum;
    int count = 0;
    int data_size = 0;
    bool data_size_initialized = false;

    for (int line_id = 0; line_id < static_cast<int>(lines.size()); ++line_id) {
        bool status;
        std::string enc = encode_type;
        if (encoded && !enc.size()) {
            // Guess the encoding type from the file name
            std::string fn = lines[line_id].first;
            size_t p = fn.rfind('.');
            if ( p == fn.npos )
                LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
            enc = fn.substr(p);
            std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
        }
        status = ReadImageToDatum(lines[line_id].first,
                                  lines[line_id].second, resize_height, resize_width, is_color,
                                  enc, &datum);
        if (status == false) continue;
        if (check_size) {
            if (!data_size_initialized) {
                data_size = datum.channels() * datum.height() * datum.width();
                data_size_initialized = true;
            } else {
                const std::string& data = datum.data();
                CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
                                                 << data.size();
            }
        }
        // sequential
        std::string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first;

        // Put in db
        std::string out;
        CHECK(datum.SerializeToString(&out));
        txn->Put(key_str, out);

        if (++count % 1000 == 0) {
            // Commit db
            txn->Commit();
            txn.reset(db->NewTransaction());
            LOG(INFO) << "Processed " << count << " files.";
        }
    }
    // write the last batch
    if (count % 1000 != 0) {
        txn->Commit();
        LOG(INFO) << "Processed " << count << " files.";
    }
    db->Close();
    infile.close();
}

V4R_EXPORTS
void
computeMean(const std::string &db_filename, const std::string &mean_file)
{
    boost::scoped_ptr<caffe::db::DB> db(caffe::db::GetDB(FLAGS_backend));
    db->Open(db_filename, caffe::db::READ);
    boost::scoped_ptr<caffe::db::Cursor> cursor(db->NewCursor());

    caffe::BlobProto sum_blob;
    int count = 0;
    // load first datum
    caffe::Datum datum;
    datum.ParseFromString(cursor->value());

    if (caffe::DecodeDatumNative(&datum)) {
      LOG(INFO) << "Decoding Datum";
    }

    sum_blob.set_num(1);
    sum_blob.set_channels(datum.channels());
    sum_blob.set_height(datum.height());
    sum_blob.set_width(datum.width());
    const int data_size = datum.channels() * datum.height() * datum.width();
    int size_in_datum = std::max<int>(datum.data().size(),
                                      datum.float_data_size());
    for (int i = 0; i < size_in_datum; ++i) {
      sum_blob.add_data(0.);
    }
    LOG(INFO) << "Starting Iteration";
    while (cursor->valid()) {
      caffe::Datum datum_tmp;
      datum_tmp.ParseFromString(cursor->value());
      caffe::DecodeDatumNative(&datum_tmp);

      const std::string& data = datum_tmp.data();
      size_in_datum = std::max<int>(datum_tmp.data().size(),
          datum_tmp.float_data_size());
      CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
          size_in_datum;
      if (data.size() != 0) {
        CHECK_EQ(data.size(), size_in_datum);
        for (int i = 0; i < size_in_datum; ++i) {
          sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
        }
      } else {
        CHECK_EQ(datum_tmp.float_data_size(), size_in_datum);
        for (int i = 0; i < size_in_datum; ++i) {
          sum_blob.set_data(i, sum_blob.data(i) +
              static_cast<float>(datum_tmp.float_data(i)));
        }
      }
      ++count;
      if (count % 10000 == 0) {
        LOG(INFO) << "Processed " << count << " files.";
      }
      cursor->Next();
    }

    if (count % 10000 != 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    for (int i = 0; i < sum_blob.data_size(); ++i) {
      sum_blob.set_data(i, sum_blob.data(i) / count);
    }
    // Write to disk
    LOG(INFO) << "Write to " << mean_file;
    caffe::WriteProtoToBinaryFile(sum_blob, mean_file);
    const int channels = sum_blob.channels();
    const int dim = sum_blob.height() * sum_blob.width();
    std::vector<float> mean_values(channels, 0.0);
    LOG(INFO) << "Number of channels: " << channels;
    for (int c = 0; c < channels; ++c) {
      for (int i = 0; i < dim; ++i) {
        mean_values[c] += sum_blob.data(dim * c + i);
      }
      LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
    }
}

V4R_EXPORTS void createTrainList(const std::string &root_path, const std::string &list_file, const std::string &class_name_file)
{

std::vector<int> model_numbers;
std::vector<std::string> file_names , model_names;


std::string load_path, inputfile, file_name;
std::vector<std::string> objects = v4r::io::getFoldersInDirectory(root_path);
objects.erase(find(objects.begin(),objects.end(),"svm"));

for(int i=0; i<objects.size(); i++){
    load_path = root_path;
    load_path.append(objects[i]);
    load_path.append("/");
    std::vector<std::string> files = v4r::io::getFilesInDirectory(load_path,".*.JPEG",false);
    std::cout<<"Converting Folder: " << objects[i] << std::endl;
    model_names.push_back(objects[i]);
    for(int j=0; j<files.size();j++){
//        for(int j=0; j<4;j++){
        load_path.append(files[j]);
        file_name = root_path;
        file_name.append(objects[i]);
        file_name.append("/");
        file_name.append(files[j]);
        file_name.append(" ");
        file_name.append(std::to_string(i));

        file_names.push_back(file_name);

    }


}

std::random_shuffle(file_names.begin(),file_names.end());
std::ofstream outSt;
outSt.open(list_file,std::fstream::out | std::fstream::trunc);
for(int i=0; i<file_names.size(); i++){
    outSt<<file_names[i];
    outSt<<std::endl;
}

outSt.close();

outSt.open(class_name_file,std::fstream::out | std::fstream::trunc);

for(int i=0; i<model_names.size(); i++){
    outSt<<model_names[i];
    outSt<<std::endl;
}

outSt.close();
}

V4R_EXPORTS std::vector<std::string> loadModelNames(const std::string &class_name_file)
{

    int bufferSize = 1024;

    char linebuf[bufferSize];

    std::ifstream in (class_name_file.c_str (), std::ifstream::in);

    std::vector<std::string> names;

    while(in.getline (linebuf, bufferSize)){

        std::string line (linebuf);
        names.push_back(line);

    }

    return names;

}

V4R_EXPORTS void createStdList(const std::string &train_list, const std::string &std_list, const int &class_num)
{
    int bufferSize = 1024;

    char linebuf[bufferSize];



    std::ofstream outSt;

    outSt.open(std_list.c_str(),std::ios::out);
    std::vector < std::string > strs_2;

    for(int j=0;j<class_num;j++)
    {

        int i=1;
        std::ifstream in (train_list.c_str (), std::ifstream::in);
        while(i<=20)
        {
            in.getline (linebuf, bufferSize);
            std::string line (linebuf);
            boost::split (strs_2, line, boost::is_any_of (" "));
            if(static_cast<int> (atof (strs_2[1].c_str ()))==j)
            {
                outSt << line;
                outSt << std::endl;
                i++;
            }

        }
        in.close();


    }


    outSt.close();
}

V4R_EXPORTS void editTrainIter(const std::string &protoxt_filename, size_t num_iter)
{
    std::ifstream infile(protoxt_filename.c_str());
    const std::string tmp_file_name = protoxt_filename + "~~";
    std::ofstream outfile(tmp_file_name);

    std::string line;
    while (std::getline(infile, line)) {
        outfile << line << std::endl;
        while (std::getline(infile, line))
        {
            size_t pos = line.find( "max_iter:" );
            if (  pos != std::string::npos)
            {
                for(size_t i=0; i<pos; i++)
                    outfile << " " ;

                outfile << "max_iter: " << num_iter << std::endl;
            }
            else
                outfile << line << std::endl;
        }

    }
    infile.close();
    outfile.close();

    bf::copy_file(tmp_file_name, protoxt_filename, bf::copy_option::overwrite_if_exists);
    bf::remove(tmp_file_name);
}

V4R_EXPORTS size_t getTrainIter(const std::string &protoxt_filename)
{
    std::ifstream infile(protoxt_filename.c_str());

    size_t number;
    std::string line;
    while (std::getline(infile, line))
    {

        while (std::getline(infile, line))
        {
            size_t pos = line.find( "max_iter:" );
            if (  pos != std::string::npos)
            {
                std::vector<std::string> strs_2;
                boost::split (strs_2, line, boost::is_any_of (" "));
                number = static_cast<size_t> (atof (strs_2[1].c_str ()));


            }

        }

    }
    infile.close();

    return number;

}



}
}

