# -------------------------------------------------------------------
# MAdLib - Copyright (C) Cenaero
#
# See the Copyright.txt and License.txt files for license information.
# -------------------------------------------------------------------
list(APPEND SEQUENTIAL_TEST_DIRS
  countVertices mappingTriangles mappingQuad mappingTet mappingPyramid mappingPrism mappingHex mappingTriangle2O mappingQuad2O mappingTet2O cutTriangles cutTriangles2O cutTriangles3O cutTriangles4O cutTetra cutTetra2O cutTetra3O cutTetra4O SolutionBasedAdaptation qualityIndicator boundaryLayerUntangling CollapseTriangles2O CollapseTetra2O transformation3Points comparePerfs curved2DAdaptation curved3DAdaptation computeArea computeVolume metricWork solAnisoAdaptation2D solAnisoAdaptation3D solAnisoAdaptationHO2D solAnisoAdaptationHO3D logSimplexTest vertexSmoothing isoAdaptation anisoAdaptation metricSmoothing multipleSolAdaptation2D CustomAdaptation CustomAdaptation3D
)

list(APPEND PARALLEL_TEST_DIRS
)

if(MADLIB_USE_PARALLEL)
  set(TEST_DIRS ${PARALLEL_TEST_DIRS})
  set(NUMPROCS_EXEC 4)
else()
  set(TEST_DIRS ${SEQUENTIAL_TEST_DIRS})
endif()

foreach(test_dir ${TEST_DIRS})

  # -- Build tests --
  # -----------------
  # set testcase name (test_dir without extension)
  get_filename_component(TName ${test_dir} NAME_WE)
  set(TName "${TName}")
  if(MADLIB_USE_PARALLEL)
    set(TName "${TName}_para")
  endif()

  set(test_bin ${CMAKE_BINARY_DIR}/Testcases/${TName})
  set(full_test_dir ${CMAKE_SOURCE_DIR}/Testcases/${test_dir})

  if(MADLIB_BUILD_TESTCASES)
    add_executable(${TName} ${full_test_dir}/main.cc)
    if(MSVC)
      set_target_properties(${TName} PROPERTIES VS_DEBUGGER_COMMAND_ARGUMENTS "${full_test_dir}")
    endif()
  else()
    add_executable(${TName} EXCLUDE_FROM_ALL ${full_test_dir}/main.cc)
  endif()
  # Duplacated function in SUBLIBS and MAdLib to avoid libs 
  #   ordering when linking in static
  if(MADLIB_BUILD_STATIC_LIBRARY)
    target_link_libraries(${TName} MAdLibSTATIC ${MADLIB_EXTERNAL_LIBRARIES_STATIC})
  else()
    target_link_libraries(${TName} MAdLib ${MADLIB_EXTERNAL_LIBRARIES_SHARED})
  endif()

  include_directories(${MADLIB_INCLUDE_DIRS} ${MADLIB_EXTERNAL_INCLUDE_DIRS})
  if(MADLIB_BUILD_TESTCASES)
    install(TARGETS ${TName} RUNTIME DESTINATION bin)
  endif()

  set(submit_test_command "${test_bin}")
  set(submit_test_args "${full_test_dir}") # Path to mesh in source dir
  if(MADLIB_USE_PARALLEL)
    set(submit_test_args ${MPIEXEC_NUMPROC_FLAG} ${NUMPROCS_EXEC}
        ${MPIEXEC_PREFLAGS} ${mpiopt} ${submit_test_command})
    set(submit_test_command "${MPIEXEC}")
  endif()

  # -- Execution --
  # ---------------
  set(RUN_DIR "${test_bin}")
  
  add_test(${TName} ${submit_test_command} ${submit_test_args})

endforeach()
