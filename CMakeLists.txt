cmake_minimum_required(VERSION 3.22)
project(SwellLab VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
include(FetchContent)
FetchContent_Declare(JUCE GIT_REPOSITORY https://github.com/juce-framework/JUCE.git GIT_TAG 8.0.4 GIT_SHALLOW ON)
FetchContent_MakeAvailable(JUCE)
juce_add_plugin(SwellLab
    PRODUCT_NAME "Swell Lab"
    COMPANY_NAME "Homemade"
    PLUGIN_MANUFACTURER_CODE Hmde
    PLUGIN_CODE Swlb
    FORMATS VST3
    VST3_CATEGORIES Fx Reverb
    NEEDS_WEB_BROWSER FALSE
    NEEDS_CURL FALSE)
target_sources(SwellLab PRIVATE Source/Plugin.cpp)
target_compile_definitions(SwellLab PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
juce_generate_juce_header(SwellLab)
target_link_libraries(SwellLab PRIVATE juce::juce_audio_utils juce::juce_dsp
    PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)
