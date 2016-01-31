#ifndef V4R_RECOGNITION_MODEL_HV_H__
#define V4R_RECOGNITION_MODEL_HV_H__

#include <v4r/core/macros.h>
#include <pcl/common/common.h>
#include <v4r/common/common_data_structures.h>

namespace v4r
{
  template<typename ModelT>
  class V4R_EXPORTS HVRecognitionModel
  {
    public:
      std::vector<int> explained_; /// @brief explained scene points by_RM_
      std::vector<float> explained_distances_; /// @brief closest distances to the scene for point i
      std::vector<int> unexplained_in_neighborhood; /// @brief indices vector referencing unexplained_by_RM_neighboorhods
      std::vector<float> unexplained_in_neighborhood_weights; /// @brief weights for the points not being explained in the neighborhood of a hypothesis
      std::vector<int> outlier_indices_; /// @brief outlier indices of this model (coming from all types)
      std::vector<int> color_outliers_indices_; /// @brief all model points that have a scene point nearby but whose color does not match
      std::vector<int> outliers_3d_indices_;    /// @brief all model points that do not have a scene point nearby
      std::vector<int> complete_cloud_occupancy_indices_;
      std::vector<bool> scene_point_explained_by_hypothesis_; /// @brief boolean vector indicating if a scene point is explained by this model or not

      typename pcl::PointCloud<ModelT>::Ptr visible_cloud_;
      typename pcl::PointCloud<ModelT>::Ptr complete_cloud_;
      pcl::PointCloud<pcl::Normal>::Ptr visible_cloud_normals_;
      pcl::PointCloud<pcl::Normal>::Ptr complete_cloud_normals_;
      std::vector<int> visible_indices_;

      float bad_information_;
      float outliers_weight_;
      size_t id_;
      float extra_weight_; /// @brief descriptor distance weight for instance
      float color_similarity_;
      float median_;
      float mean_;
      Eigen::MatrixXf color_mapping_;
      float hyp_penalty_;
      std::string id_s_;
      Eigen::MatrixXf cloud_color_channels_;  /// @brief color values for each point in the scene (row_id). Width is equal to the number of color channels
      std::vector<float> cloud_GS_; /// @brief Grayscale cloud
      float min_contribution_; /// @brief based on the amount of explained points and the amount of information in the hypotheses
      std::vector<float> normal_angle_histogram_;
      std::vector<float> color_diff_histogram_;
      float normal_entropy_;
      float color_entropy_;
      std::vector<int> cloud_indices_specified_;
      float color_diff_trhough_specification_;
      pcl::PointCloud<pcl::PointXYZL>::Ptr visible_labels_;
      bool is_planar_; /// @brief if true, this model is a planar model
      PlaneModel<ModelT> plane_model_;

      pcl::PointCloud<pcl::PointXYZL>::Ptr smooth_faces_;

      //inlier indices and distances for cloud_ (this avoids recomputing radius searches twice (one for specification and one for inlier/outlier detection)
      std::vector<std::vector<int> > inlier_indices_;
      std::vector<std::vector<float> > inlier_distances_;

      HVRecognitionModel() : extra_weight_(1.f)
      {
      }

      typedef boost::shared_ptr< HVRecognitionModel> Ptr;
      typedef boost::shared_ptr< HVRecognitionModel const> ConstPtr;
  };
}

#endif