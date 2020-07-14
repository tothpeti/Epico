# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.

set(HEADER_LIST 
  "${epico-app_SOURCE_DIR}/include/RandomDatasetGenerator.hpp"
  "${epico-app_SOURCE_DIR}/include/LogisticRegression.hpp"
)

# Make an automatic library - will be static or dynamic based on user setting
add_library(library RandomDatasetGenerator.cpp 
  ${HEADER_LIST}
)

# We need this directory, and users of our library will need it too
target_include_directories(library PUBLIC ../include)

# This depends on libtorch
target_link_libraries(library PRIVATE "${TORCH_LIBRARIES}")

# All users of this library will need at least C++11
target_compile_features(library PUBLIC cxx_std_11)

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})
