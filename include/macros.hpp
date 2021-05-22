#ifndef INTACT_MACROS_HPP
#define INTACT_MACROS_HPP

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

// #define WHILE_CLUSTERS_READY                                                   \
//     while (!sptr_intact->isSegmented()) {                                      \
//         std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
//     }

// #define WHILE_BACKGROUND_READY                                                 \
//     while (!sptr_intact->isChromakeyed()) {                                    \
//         std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
//     }

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

#define START bool init = true;

#define STOP sptr_intact->raiseStopFlag();

#define PRINT(x) ply::write(x)

#endif /*INTACT_MACROS_HPP*/
