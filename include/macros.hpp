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

#define KINECT_READY                                                           \
    if (init) {                                                                \
        init = false;                                                          \
        sptr_intact->raiseKinectReadyFlag();                                   \
    }

#define INTACT_READY                                                           \
    if (init) {                                                                \
        init = false;                                                          \
        sptr_intact->raiseIntactReadyFlag();                                   \
    }

#define SEGMENT_READY                                                          \
    if (init) {                                                                \
        init = false;                                                          \
        sptr_intact->raiseSegmentedFlag();                                     \
    }

#define CLUSTERS_READY                                                         \
    if (init) {                                                                \
        init = false;                                                          \
        sptr_intact->raiseClusteredFlag();                                     \
    }

#define CHROMABACKGROUND_READY                                                 \
    if (init) {                                                                \
        init = false;                                                          \
        sptr_intact->raiseBackgroundReadyFlag();                               \
    }

#define POLLING_EXIT_STATUS                                                    \
    if (sptr_intact->isStop()) {                                               \
        sptr_intact->stop();                                                   \
    }

#define WHILE_SEGMENT_READY                                                    \
    while (!sptr_intact->isSegmented()) {                                      \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define WHILE_KINECT_READY                                                     \
    while (!sptr_intact->isKinectReady()) {                                    \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define WHILE_INTACT_READY                                                     \
    while (!sptr_intact->isIntactReady()) {                                    \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define WHILE_SEGMENT_READY                                                    \
    while (!sptr_intact->isSegmented()) {                                      \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define WHILE_CLUSTERS_READY                                                   \
    while (!sptr_intact->isClustered()) {                                      \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define WHILE_CHROMABACKGROUND_READY                                           \
    while (!sptr_intact->isBackgroundReady()) {                                \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define STOP sptr_intact->raiseStopFlag();

#define START bool init = true;

#define PRINT(x) ply::write(x)

#define SEGMENT 1
#define SIFT 1
#define RENDER 1
#define DETECT 1
#define CHROMAKEY 1
#define CLUSTER 1

#endif /*INTACT_MACROS_HPP*/
