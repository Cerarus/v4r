/*
 * global_nn_classifier.cpp
 *
 *  Created on: Mar 9, 2012
 *      Author: aitor
 */

#include <v4r/recognition/impl/global_nn_recognizer_cvfh.hpp>
#include <v4r/recognition/metrics.h>

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<367>,
    (float[367], histogram, histogram367)
)

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<431>,
    (float[431], histogram, histogram431)
)

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<559>,
    (float[559], histogram, histogram559)
)

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<815>,
    (float[815], histogram, histogram815)
)

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<879>,
    (float[879], histogram, histogram879)
)

POINT_CLOUD_REGISTER_POINT_STRUCT (pcl::Histogram<1327>,
    (float[1327], histogram, histogram1327)
)


//Instantiation
//template class pcl::GlobalNNCVFHRecognizer<flann::L2, pcl::PointXYZ, pcl::VFHSignature308>;
//template class pcl::GlobalNNCVFHRecognizer<flann::L1, pcl::PointXYZ, pcl::VFHSignature308>;
template class v4r::GlobalNNCVFHRecognizer<v4r::Metrics::HistIntersectionUnionDistance, pcl::PointXYZ, pcl::VFHSignature308>;
template class v4r::GlobalNNCVFHRecognizer<v4r::Metrics::HistIntersectionUnionDistance, pcl::PointXYZRGB, pcl::VFHSignature308>;
template class v4r::GlobalNNCVFHRecognizer<v4r::Metrics::HistIntersectionUnionDistance, pcl::PointXYZRGB, pcl::Histogram<431> >;
template class v4r::GlobalNNCVFHRecognizer<v4r::Metrics::HistIntersectionUnionDistance, pcl::PointXYZRGB, pcl::Histogram<559> >;
//template class pcl::GlobalNNCVFHRecognizer<Metrics::HistIntersectionUnionDistance, pcl::PointXYZRGB, pcl::Histogram<815> >;
//template class pcl::GlobalNNCVFHRecognizer<Metrics::HistIntersectionUnionDistance, pcl::PointXYZRGB, pcl::Histogram<879> >;
template class v4r::GlobalNNCVFHRecognizer<v4r::Metrics::HistIntersectionUnionDistance, pcl::PointXYZRGB, pcl::Histogram<1327> >;
//template class pcl::GlobalNNCVFHRecognizer<flann::L2, pcl::PointXYZRGB, pcl::Histogram<431> >;
//template class pcl::GlobalNNCVFHRecognizer<flann::L2, pcl::PointXYZRGB, pcl::Histogram<431> >;
//template class pcl::GlobalNNCVFHRecognizer<flann::L1, pcl::PointXYZRGB, pcl::Histogram<431> >;
