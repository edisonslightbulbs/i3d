#ifndef INTACT_MACROS_HPP
#define INTACT_MACROS_HPP

#include "logger.h"
#include "ply.h"
#include "timer.h"

// colors handy for coloring clusters:
// see@ https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=9
//
__attribute__((unused)) constexpr uint8_t red[3] = { 215, 48, 39 };
__attribute__((unused)) constexpr uint8_t orange[3] = { 244, 109, 67 };
__attribute__((unused)) constexpr uint8_t gold[3] = { 253, 173, 97 };
__attribute__((unused)) constexpr uint8_t brown[3] = { 254, 224, 144 };
__attribute__((unused)) constexpr uint8_t yellow[3] = { 255, 255, 191 };
__attribute__((unused)) constexpr uint8_t skyblue[3] = { 224, 243, 248 };
__attribute__((unused)) constexpr uint8_t oceanblue[3] = { 171, 217, 233 };
__attribute__((unused)) constexpr uint8_t blue[3] = { 116, 173, 209 };
__attribute__((unused)) constexpr uint8_t deepblue[3] = { 69, 117, 180 };

// colors handy for coloring clusters:
// https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=9 */
//
__attribute__((unused)) constexpr uint8_t depbrown[3] = { 140, 81, 10 };
__attribute__((unused)) constexpr uint8_t darkbrown[3] = { 191, 129, 45 };
__attribute__((unused)) constexpr uint8_t goldenbrown[3] = { 223, 194, 125 };
__attribute__((unused)) constexpr uint8_t khaki[3] = { 223, 232, 195 };
__attribute__((unused)) constexpr uint8_t lightgrey[3] = { 245, 245, 245 };
__attribute__((unused)) constexpr uint8_t lightgreen[3] = { 199, 234, 229 };
__attribute__((unused)) constexpr uint8_t green[3] = { 128, 205, 193 };
__attribute__((unused)) constexpr uint8_t chromagreen[3] = { 120, 198, 121 };
__attribute__((unused)) constexpr uint8_t deepgreen[3] = { 53, 151, 143 };
__attribute__((unused)) constexpr uint8_t darkgreen[3] = { 1, 102, 94 };
__attribute__((unused)) constexpr uint8_t black[3] = { 0, 0, 0 };

#define SLEEP_UNTIL_RESOURCES_READY                                            \
    while (!sptr_i3d->isSensorReady()) {                                       \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }
#define RAISE_SENSOR_RESOURCES_READY_FLAG                                      \
    if (init) {                                                                \
        init = false;                                                          \
        LOG(INFO) << "-- k4a driver running";                                  \
        sptr_i3d->raiseSensorReadyFlag();                                      \
    }

#define SLEEP_UNTIL_POINTCLOUD_READY                                           \
    while (!sptr_i3d->isPCloudReady()) {                                       \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }
#define RAISE_POINTCLOUD_READY_FLAG                                            \
    if (init) {                                                                \
        init = false;                                                          \
        LOG(INFO) << "-- point cloud created";                                 \
        sptr_i3d->raisePCloudReadyFlag();                                      \
    }

#define SLEEP_UNTIL_FRAMES_READY                                               \
    while (!sptr_i3d->framesReady()) {                                         \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }
#define RAISE_FRAMES_READY_FLAG                                                \
    if (init) {                                                                \
        init = false;                                                          \
        LOG(INFO) << "-- openGL and openCV frames created ";                   \
        sptr_i3d->raiseFramesReadyFlag();                                      \
    }

#define SLEEP_UNTIL_BOUNDARY_SET                                               \
    while (!sptr_i3d->isBoundarySet()) {                                       \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }
#define RAISE_BOUNDARY_SET_FLAG                                                \
    if (init) {                                                                \
        init = false;                                                          \
        LOG(INFO) << "-- segment boundary set ";                               \
        sptr_i3d->raiseBoundarySetFlag();                                      \
    }

#define SLEEP_UNTIL_SEGMENTATION_DONE                                          \
    while (!sptr_i3d->isSegmented()) {                                         \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }
#define RAISE_SEGMENTATION_DONE_FLAG                                           \
    if (init) {                                                                \
        init = false;                                                          \
        LOG(INFO) << "-- segmenting point cloud";                              \
        sptr_i3d->raiseSegmentedFlag();                                        \
    }

#define CLUSTERS_READY                                                         \
    if (init) {                                                                \
        init = false;                                                          \
        sptr_i3d->raiseClusteredFlag();                                        \
    }

#define CHROMABACKGROUND_READY                                                 \
    if (init) {                                                                \
        init = false;                                                          \
        sptr_i3d->raiseBackgroundReadyFlag();                                  \
    }

#define POLL_EXIT_STATUS                                                       \
    if (sptr_i3d->isStop()) {                                                  \
        sptr_i3d->stop();                                                      \
    }

#define WHILE_SEGMENT_READY                                                    \
    while (!sptr_i3d->isSegmented()) {                                         \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define SLEEP_UNTIL_SEGMENT_READY                                              \
    while (!sptr_i3d->isSegmented()) {                                         \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define WHILE_CLUSTERS_READY                                                   \
    while (!sptr_i3d->isClustered()) {                                         \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define WHILE_CHROMABACKGROUND_READY                                           \
    while (!sptr_i3d->isBackgroundReady()) {                                   \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define STOP sptr_i3d->raiseStopFlag();

#define START bool init = true;

#define PRINT(pCloud) ply::write(pCloud);

#define BENCHMARK 0
#if BENCHMARK == 1
#define START_TIMER Timer timer;

#define STOP_TIMER(str) LOG(INFO) << str << timer.getDuration() << " ms";
#else
#define START_TIMER
#define STOP_TIMER(str)
#endif

#define BUILD_POINTCLOUD 1
#define FRAMES 1
#define PROPOSAL 1
#define SEGMENT 1
#define RENDER 1
#define OR 0        // testing not ready for commit
#define CHROMAKEY 0 // testing not ready for commit
#define CLUSTER 0   // testing not ready for commit

// todo: introduce macro configuration logic

#endif /*INTACT_MACROS_HPP*/
