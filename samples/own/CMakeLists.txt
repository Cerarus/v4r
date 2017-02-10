if(BUILD_EXAMPLES)

  if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_NOISY_WARNINGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
  endif()

  SET(sample_kind example)
  SET(sample_KIND EXAMPLE)

  SET(V4R_DEPS v4r_rendering v4r_segmentation v4r_own)
  V4R_DEFINE_CPP_EXAMPLE(cloud_generator)

  V4R_DEFINE_CPP_EXAMPLE(esf_class)


  V4R_DEFINE_CPP_EXAMPLE(esf_class_teach)


  V4R_DEFINE_CPP_EXAMPLE(esf_gen)

  SET(V4R_DEPS v4r_rendering v4r_segmentation v4r_ml)
  #V4R_DEFINE_CPP_EXAMPLE(esf_svm_teach)

  #V4R_DEFINE_CPP_EXAMPLE(esf_svm_class)

  #V4R_DEFINE_CPP_EXAMPLE(esf_svm_test)
  #V4R_DEFINE_CPP_EXAMPLE(svm_test)
  SET(V4R_DEPS v4r_io v4r_own v4r_ml v4r_rendering v4r_segmentation v4r_common caffe)
  V4R_DEFINE_CPP_EXAMPLE(caffee_teach)
  #V4R_DEFINE_CPP_EXAMPLE(caffe_class)
  V4R_DEFINE_CPP_EXAMPLE(caffe_esf_class)
V4R_DEFINE_CPP_EXAMPLE(test_data_gen)
V4R_DEFINE_CPP_EXAMPLE(alexnet_iterative_test)
V4R_DEFINE_CPP_EXAMPLE(create_caffe_list)
V4R_DEFINE_CPP_EXAMPLE(train_data_gen)
V4R_DEFINE_CPP_EXAMPLE(test_caffe_train)
V4R_DEFINE_CPP_EXAMPLE(test_pcl_input)

V4R_DEFINE_CPP_EXAMPLE(caffe_model_test)

V4R_DEFINE_CPP_EXAMPLE(create_caffe_list)
V4R_DEFINE_CPP_EXAMPLE(caffe_iterativ_v1)
endif()

if (INSTALL_C_EXAMPLES AND NOT WIN32)
  file(GLOB C_SAMPLES *.c *.cpp *.jpg *.png *.data makefile.* build_all.sh *.dsp *.cmd )
  install(FILES ${C_SAMPLES}
          DESTINATION ${V4R_SAMPLES_SRC_INSTALL_PATH}/cpp
          PERMISSIONS OWNER_READ GROUP_READ WORLD_READ COMPONENT samples)
endif()
