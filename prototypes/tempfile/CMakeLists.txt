add_executable(temp
  temp.cpp
)

target_link_libraries(temp
  PRIVATE
  LLVMSupport
)
