/*
 * multi_pipeline_recognizer.h
 *
 *  Created on: Feb 24, 2013
 *      Author: aitor
 */

#ifndef MULTI_PIPELINE_RECOGNIZER_HPP_
#define MULTI_PIPELINE_RECOGNIZER_HPP_

#include <v4r/recognition/multi_pipeline_recognizer.h>
#include <v4r/common/normal_estimator.h>
//#include "multi_object_graph_CG.h"
//#include <pcl/visualization/pcl_visualizer.h>

template<typename PointInT>
void
v4r::MultiRecognitionPipeline<PointInT>::initialize()
{
    if(ICP_iterations_ > 0 && icp_type_ == 1)
    {
        for(size_t i=0; i < recognizers_.size(); i++)
        {
            recognizers_[i]->getDataSource()->createVoxelGridAndDistanceTransform(VOXEL_SIZE_ICP_);
        }
    }
}

template<typename PointInT>
void
v4r::MultiRecognitionPipeline<PointInT>::reinitialize()
{
    for(size_t i=0; i < recognizers_.size(); i++)
    {
        recognizers_[i]->reinitialize();
    }

    initialize();
}

template<typename PointInT>
void
v4r::MultiRecognitionPipeline<PointInT>::reinitialize(const std::vector<std::string> & load_ids)
{
    for(size_t i=0; i < recognizers_.size(); i++)
    {
        std::cout << "Calling recognizer with ids size: " << load_ids.size() << std::endl;
        recognizers_[i]->reinitialize(load_ids);
    }

    initialize();
}

template<typename PointInT>
void
v4r::MultiRecognitionPipeline<PointInT>::getPoseRefinement(
        const std::vector<ModelTPtr> &models,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > &transforms)
{
    models_ = models;
    transforms_ = transforms;
    poseRefinement();
    transforms = transforms_; //is this neccessary?
}

template<typename PointInT>
void
v4r::MultiRecognitionPipeline<PointInT>::recognize()
{
    models_.clear();
    transforms_.clear();

    //first version... just call each recognizer independently...
    //more advanced version should compute normals and preprocess the input cloud so that
    //we avoid recomputing stuff shared among the different pipelines
    std::vector<int> input_icp_indices;
    if(cg_algorithm_)
        set_save_hypotheses_ = true;

    std::cout << "Number of recognizers:" << recognizers_.size() << std::endl;

    //typename std::map<std::string, ObjectHypothesis<PointInT> > object_hypotheses_;
    object_hypotheses_.clear();
    typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map_oh;
    keypoints_cloud_.reset(new pcl::PointCloud<PointInT>);
    keypoint_indices_.indices.clear();

    for(size_t i=0; i < recognizers_.size(); i++)
    {
        recognizers_[i]->setInputCloud(input_);

        if(recognizers_[i]->requiresSegmentation())
        {
            if(recognizers_[i]->acceptsNormals() && normals_set_)
            {
                recognizers_[i]->setSceneNormals(scene_normals_);
            }

            for(size_t c=0; c < segmentation_indices_.size(); c++)
            {
                recognizers_[i]->setIndices(segmentation_indices_[c].indices);
                recognizers_[i]->recognize();
                std::vector<ModelTPtr> models_tmp = recognizers_[i]->getModels ();
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_tmp = recognizers_[i]->getTransforms ();

                models_.insert(models_.end(), models_tmp.begin(), models_tmp.end());
                transforms_.insert(transforms_.end(), transforms_tmp.begin(), transforms_tmp.end());
                input_icp_indices.insert(input_icp_indices.end(), segmentation_indices_[c].indices.begin(), segmentation_indices_[c].indices.end());
            }
        }
        else
        {
            recognizers_[i]->setSaveHypotheses(set_save_hypotheses_);
            recognizers_[i]->setIndices(indices_);
            recognizers_[i]->recognize();

            if(!set_save_hypotheses_)
            {
                std::vector<ModelTPtr> models = recognizers_[i]->getModels ();
                std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms = recognizers_[i]->getTransforms ();

                models_.insert(models_.end(), models.begin(), models.end());
                transforms_.insert(transforms_.end(), transforms.begin(), transforms.end());
            }
            else
            {
                typename std::map<std::string, ObjectHypothesis<PointInT> > oh_tmp;
                recognizers_[i]->getSavedHypotheses(oh_tmp);

                pcl::PointIndices kp_idx_tmp;
                typename pcl::PointCloud<PointInT>::Ptr kp_tmp(new pcl::PointCloud<PointInT>);
                recognizers_[i]->getKeypointCloud(kp_tmp);
                recognizers_[i]->getKeypointIndices(kp_idx_tmp);

                typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it;
                for (it = oh_tmp.begin (); it != oh_tmp.end (); it++)
                {
                    const std::string id = it->second.model_->id_;

                    it_map_oh = object_hypotheses_.find(id);
                    if(it_map_oh == object_hypotheses_.end())   // no feature correspondences exist yet
                    {
                        if(keypoints_cloud_->points.size())
                        {
                            std::cerr << "???" << std::endl;
                            for(size_t kk=0; kk < it->second.model_scene_corresp->size(); kk++)
                            {
                                it->second.model_scene_corresp->at(kk).index_match += keypoints_cloud_->points.size();
                            }
                        }

                        object_hypotheses_.insert(std::pair<std::string, ObjectHypothesis<PointInT> >(id, it->second));
                    }
                    else
                    {
                        //update correspondences indices so they match each recognizer (keypoints_cloud to correspondences_pointcloud)
                        for(size_t kk=0; kk < it->second.model_scene_corresp->size(); kk++)
                        {
                            pcl::Correspondence c = it->second.model_scene_corresp->at(kk);
                            c.index_match += keypoints_cloud_->points.size();
                            c.index_query += it_map_oh->second.model_keypoints->points.size();
                            it_map_oh->second.model_scene_corresp->push_back(c);
                        }

                        *it_map_oh->second.model_keypoints += * it->second.model_keypoints;
                        *it_map_oh->second.model_kp_normals += * it->second.model_kp_normals;
                        it_map_oh->second.indices_to_flann_models_.insert(
                                    it_map_oh->second.indices_to_flann_models_.end(),
                                    it->second.indices_to_flann_models_.begin(),
                                    it->second.indices_to_flann_models_.end());
                    }
                }

                *keypoints_cloud_ += *kp_tmp;
                keypoint_indices_.indices.insert(keypoint_indices_.indices.end(), kp_idx_tmp.indices.begin(), kp_idx_tmp.indices.end());
                assert(keypoints_cloud_->points.size() == keypoint_indices_.indices.size());
            }

            input_icp_indices.insert(input_icp_indices.end(), indices_.begin(), indices_.end());
        }
    }

    if(cg_algorithm_ || multi_object_correspondence_grouping_)
        correspondenceGrouping();

    if ((ICP_iterations_ > 0 || hv_algorithm_)  && (cg_algorithm_ || multi_object_correspondence_grouping_)) {
        //Prepare scene and model clouds for the pose refinement step

        std::sort( input_icp_indices.begin(), input_icp_indices.end() );
        input_icp_indices.erase( std::unique( input_icp_indices.begin(), input_icp_indices.end() ), input_icp_indices.end() );

        pcl::PointIndices ind;
        ind.indices = input_icp_indices;
        icp_scene_indices_.reset(new pcl::PointIndices(ind));
        getDataSource()->voxelizeAllModels (VOXEL_SIZE_ICP_);
    }

    if (ICP_iterations_ > 0  && (cg_algorithm_ || multi_object_correspondence_grouping_))
    {
        std::cout << "Do pose refinement.... " << std::endl;
        poseRefinement();
    }

    if (hv_algorithm_ && (models_.size () > 0) && (cg_algorithm_ || multi_object_correspondence_grouping_))
    {
        std::cout << "Do hypothesis verification..." << models_.size () << std::endl;
        hypothesisVerification();
    }
}


template<typename PointInT>
void v4r::MultiRecognitionPipeline<PointInT>::correspondenceGrouping()
{
    if(set_save_hypotheses_ && object_hypotheses_.size() > 0)
    {
        pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
        if(cg_algorithm_->getRequiresNormals())
        {
            pcl::PointCloud<pcl::Normal>::Ptr all_scene_normals;

            //compute them...
            PCL_WARN("Need to compute normals due to the cg algorithm\n");
            all_scene_normals.reset(new pcl::PointCloud<pcl::Normal>);
            PointInTPtr processed (new pcl::PointCloud<PointInT>);

            if(!normals_set_)
            {
                pcl::ScopeTime t("compute normals\n");
                boost::shared_ptr<v4r::PreProcessorAndNormalEstimator<PointInT, pcl::Normal> > normal_estimator;
                normal_estimator.reset (new v4r::PreProcessorAndNormalEstimator<PointInT, pcl::Normal>);
                normal_estimator->setCMR (false);
                normal_estimator->setDoVoxelGrid (false);
                normal_estimator->setRemoveOutliers (false);
                normal_estimator->setValuesForCMRFalse (0.003f, 0.02f);
                normal_estimator->setForceUnorganized(true);
                normal_estimator->estimate (input_, processed, all_scene_normals);
            }
            else
            {
                PCL_WARN("Using scene normals given from user code\n");
                processed = input_;
                all_scene_normals = scene_normals_;
            }

            //      {
            //        pcl::ScopeTime t("finding correct indices...\n");
            //        std::vector<int> correct_indices;
            //        v4r::ORmiscellaneous::getIndicesFromCloud<PointInT>(processed, keypoints_cloud_, correct_indices);
            pcl::copyPointCloud(*all_scene_normals, keypoint_indices_.indices, *scene_normals);
            //      }
        }
        std::cout << "multi_object_CG:" << multi_object_correspondence_grouping_ << std::endl;
        if(multi_object_correspondence_grouping_)
        {
            PCL_ERROR("this option does not exist\n");

            /*PCL_ERROR("multi_object_correspondence_grouping\n");
            std::cout << "Number of model hypotheses:" << pObjectHypotheses_->size() << std::endl;

            //merge all correspondences from the different objects
            //gather all model and normals point clouds

            std::vector<pcl::PointCloud<pcl::Normal>::Ptr> models_normals;
            std::vector<PointInTPtr> models_clouds;
            std::vector<pcl::CorrespondencesPtr> models_to_scene_correspondences;

            typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map;
            std::vector<std::string> object_ids;
            for (it_map = pObjectHypotheses_->begin (); it_map != pObjectHypotheses_->end (); it_map++)
            {
                if(it_map->second.model_scene_corresp->size() < 3)
                    continue;

                models_normals.push_back((*it_map).second.normals_pointcloud);
                models_clouds.push_back((*it_map).second.correspondences_pointcloud);
                models_to_scene_correspondences.push_back((*it_map).second.model_scene_corresp);

                object_ids.push_back(it_map->second.model_->id_);
            }

            v4r::MultiObjectGraphGeometricConsistencyGrouping<PointInT, PointInT> mo_gcc;
            mo_gcc.setDotDistance(0.25f);
            mo_gcc.setGCSize(0.01);
            mo_gcc.setGCThreshold(3);
            mo_gcc.setSceneCloud(keypoints_cloud_);
            mo_gcc.setSceneNormals(scene_normals);
            mo_gcc.setModelClouds(models_clouds);
            mo_gcc.setInputNormals(models_normals);
            mo_gcc.setFullSceneCloud(input_);
            mo_gcc.setModelSceneCorrespondences(models_to_scene_correspondences);
            mo_gcc.setObjectIds(object_ids);

            std::vector < pcl::Correspondences > corresp_clusters;
            mo_gcc.cluster (corresp_clusters);

            std::cout << "Instances:" << corresp_clusters.size () << " Total correspondences:" << (*it_map).second.model_scene_corresp->size () << " " << it_map->first << std::endl;*/

        }
        else
        {
            PCL_ERROR("set_save_hypotheses, doing correspondence grouping at MULTIpipeline level\n");
            typename std::map<std::string, ObjectHypothesis<PointInT> >::iterator it_map;
            for (it_map = object_hypotheses_.begin (); it_map != object_hypotheses_.end (); it_map++)
            {
                if(it_map->second.model_scene_corresp->size() < 3)
                    continue;

                std::string id = it_map->second.model_->id_;
                //std::cout << id << " " << it_map->second.model_scene_corresp->size() << std::endl;
                std::vector < pcl::Correspondences > corresp_clusters;
                cg_algorithm_->setSceneCloud (keypoints_cloud_);
                cg_algorithm_->setInputCloud ((*it_map).second.model_keypoints);

                if(cg_algorithm_->getRequiresNormals())
                {
                    //std::cout << "CG alg requires normals..." << ((*it_map).second.normals_pointcloud)->points.size() << " " << (scene_normals)->points.size() << std::endl;
                    cg_algorithm_->setInputAndSceneNormals((*it_map).second.model_kp_normals, scene_normals);
                }
                //we need to pass the keypoints_pointcloud and the specific object hypothesis
                cg_algorithm_->setModelSceneCorrespondences ((*it_map).second.model_scene_corresp);
                cg_algorithm_->cluster (corresp_clusters);

                std::cout << "Instances:" << corresp_clusters.size () << " Total correspondences:" << (*it_map).second.model_scene_corresp->size () << " " << it_map->first << std::endl;

                /*if(corresp_clusters.size() > 0)
            {

                Eigen::Matrix4f best_trans;
                typename pcl::registration::TransformationEstimationSVD < PointInT, PointInT > t_est;
                t_est.estimateRigidTransformation (*(*it_map).second.correspondences_pointcloud, *keypoints_cloud, corresp_clusters[0], best_trans);

                Eigen::Matrix4f translate_y_camera;
                translate_y_camera.setIdentity();
                translate_y_camera(1,3) = -0.75;
                best_trans = translate_y_camera * best_trans;

                bool inverse=true;
                if(inverse)
                  best_trans = best_trans.inverse().eval();

                pcl::visualization::PCLVisualizer vis("correspondences");
                int v1,v2;
                vis.createViewPort(0,0,1,1,v1);
                //vis.createViewPort(0.5,0,1,1,v2);
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
                pcl::copyPointCloud(*input_, *scene_cloud);
                PointInTPtr keypoints_cloud_trans(new pcl::PointCloud<PointInT>);
                pcl::copyPointCloud(*keypoints_cloud, *keypoints_cloud_trans);

                if(inverse)
                {
                    pcl::transformPointCloud(*scene_cloud, *scene_cloud, best_trans);
                    pcl::transformPointCloud(*keypoints_cloud_trans, *keypoints_cloud_trans, best_trans);
                }

                vis.addPointCloud(scene_cloud, "scene cloud", v1);

                //{
                //    pcl::visualization::PointCloudColorHandlerCustom<PointInT> scene_handler(keypoints_cloud_trans, 255, 255, 0);
                //    vis.addPointCloud(keypoints_cloud_trans, scene_handler, "scene keypoints", v1);
                //    vis.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "scene keypoints");
                //}

                {
                    ConstPointInTPtr model_cloud = it_map->second.model_->getAssembled (-1);
                    PointInTPtr model_cloud_trans(new pcl::PointCloud<PointInT>(*model_cloud));

                    if(!inverse)
                      pcl::transformPointCloud(*model_cloud, *model_cloud_trans, best_trans);

                    vis.addPointCloud(model_cloud_trans, "model");
                }

                PointInTPtr model_keypoints_trans(new pcl::PointCloud<PointInT>(*(*it_map).second.correspondences_pointcloud));

                if(!inverse)
                  pcl::transformPointCloud(*(*it_map).second.correspondences_pointcloud, *model_keypoints_trans, best_trans);

                //pcl::visualization::PointCloudColorHandlerCustom<PointInT> scene_handler(model_keypoints_trans, 255, 0, 0);
                //vis.addPointCloud<PointInT>(model_keypoints_trans, scene_handler, "model keypoints", v1);

                float avg_distance = 0;
                for(size_t kk=0; kk < (*it_map).second.model_scene_corresp->size (); kk++)
                {
                    pcl::Correspondence c = (*it_map).second.model_scene_corresp->at(kk);
                    avg_distance += c.distance;
                }

                avg_distance /= static_cast<float>((*it_map).second.model_scene_corresp->size ());

                int step=1;
                for(size_t kk=0; kk < (*it_map).second.model_scene_corresp->size (); kk+=step)
                {
                    pcl::Correspondence c = (*it_map).second.model_scene_corresp->at(kk);

                    pcl::PointXYZ p1,p2;
                    p1.getVector3fMap() = keypoints_cloud_trans->points[c.index_match].getVector3fMap();
                    p2.getVector3fMap()  = model_keypoints_trans->points[c.index_query].getVector3fMap();

                    std::stringstream name;
                    name << "line_" << kk;
                    vis.addLine(p1,p2, 255, 0, 0, name.str(), v1);
                }

                for(size_t kk=0; kk < corresp_clusters[0].size (); kk+=step)
                {
                    pcl::Correspondence c = corresp_clusters[0][kk];

                    pcl::PointXYZ p1,p2;
                    p1.getVector3fMap() = keypoints_cloud_trans->points[c.index_match].getVector3fMap();
                    p2.getVector3fMap()  = model_keypoints_trans->points[c.index_query].getVector3fMap();

                    std::stringstream name;
                    name << "line_corresp_cluster" << kk;
                    vis.addLine(p1,p2, 0, 0, 255, name.str(), v1);
                }

                vis.setBackgroundColor(1,1,1);
                vis.addCoordinateSystem(0.2f);
                vis.spin();
            }*/

                for (size_t i = 0; i < corresp_clusters.size (); i++)
                {
                    //std::cout << "size cluster:" << corresp_clusters[i].size() << std::endl;
                    Eigen::Matrix4f best_trans;
                    typename pcl::registration::TransformationEstimationSVD < PointInT, PointInT > t_est;
                    t_est.estimateRigidTransformation (*(*it_map).second.model_keypoints, *keypoints_cloud_, corresp_clusters[i], best_trans);

                    models_.push_back ((*it_map).second.model_);
                    transforms_.push_back (best_trans);
                }
            }
        }
    }
}

template<typename PointInT>
bool
v4r::MultiRecognitionPipeline<PointInT>::isSegmentationRequired() const
{
    bool ret_value = false;
    for(size_t i=0; (i < recognizers_.size()) && !ret_value; i++)
    {
        ret_value = recognizers_[i]->requiresSegmentation();
    }

    return ret_value;
}

template<typename PointInT>
typename boost::shared_ptr<v4r::Source<PointInT> >
v4r::MultiRecognitionPipeline<PointInT>::getDataSource () const
{
    //NOTE: Assuming source is the same or contains the same models for all recognizers...
    //Otherwise, we should create a combined data source so that all models are present

    return recognizers_[0]->getDataSource();
}

#endif /* MULTI_PIPELINE_RECOGNIZER_H_ */
