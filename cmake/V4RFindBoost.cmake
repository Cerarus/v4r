if(WITH_BOOST)
  find_package(Boost "${MIN_VER_BOOST}" COMPONENTS thread program_options serialization system filesystem regex)
  if(Boost_FOUND)
    set(BOOST_LIBRARIES "${Boost_LIBRARIES}")
    set(BOOST_INCLUDE_DIRS "${Boost_INCLUDE_DIRS}")
    set(HAVE_BOOST TRUE)
  endif()
endif()
