#include <chrono>
#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>
#include <searcher.h>
#include <thread>
#include <utility>

#include "cast.h"
#include "dbscan.h"
#include "intact.h"
#include "kinect.h"
#include "object.h"
#include "region.h"
#include "timer.h"
#include "viewer.h"
#include "yolov5.h"
//#include "io.h"
//#include "searcher.h"
//#include "helpers.hpp"
//#include "calibration.h"

#define LOG_TRACE 0
#if LOG_TRACE == 1
#include "logger.h"
#define log(string) LOG(INFO) << string
#else
#define log
#endif

Intact::Intact(int& numPoints)
    : m_numPoints(numPoints)
{
    /** initialize infinite boundaries */
    Point segUB(__FLT_MIN__, __FLT_MIN__, __FLT_MIN__);
    Point segLB(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);
    Point ttopUB(__FLT_MIN__, __FLT_MIN__, __FLT_MIN__);
    Point ttopLB(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);

    m_segBoundary = { segUB, segLB };
    m_ttopBoundary = { ttopUB, ttopLB };

    sptr_run = std::make_shared<bool>(false);
    sptr_stop = std::make_shared<bool>(false);
    sptr_isClustered = std::make_shared<bool>(false);
    sptr_isSegmented = std::make_shared<bool>(false);
    sptr_isCalibrated = std::make_shared<bool>(false);
    sptr_isKinectReady = std::make_shared<bool>(false);
    sptr_isChromakeyed = std::make_shared<bool>(false);
    sptr_isEpsilonComputed = std::make_shared<bool>(false);

    int size(m_numPoints * 3);
    sptr_rawPcl = std::make_shared<std::vector<float>>(size);
    sptr_rawImg = std::make_shared<std::vector<uint8_t>>(size);
    sptr_rawPts = std::make_shared<std::vector<Point>>(size);

    sptr_segPcl = std::make_shared<std::vector<float>>(size);
    sptr_segImg = std::make_shared<std::vector<uint8_t>>(size);
    sptr_segPts = std::make_shared<std::vector<Point>>(size);

    sptr_clustPcl = std::make_shared<std::vector<float>>(size);
    sptr_clustImg = std::make_shared<std::vector<uint8_t>>(size);
    sptr_clustPts = std::make_shared<std::vector<Point>>(size);

    sptr_ttopPcl = std::make_shared<std::vector<float>>(size);
    sptr_ttopImg = std::make_shared<std::vector<uint8_t>>(size);
    sptr_ttopPts = std::make_shared<std::vector<Point>>(size);
}

int Intact::getNumPoints()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return m_numPoints;
}

void Intact::setDepthImgHeight(const int& height)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    m_depthHeight = height;
}

void Intact::setDepthImgWidth(const int& width)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    m_depthWidth = width;
}

int Intact::getDepthImgWidth()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return m_depthWidth;
}

int Intact::getDepthImgHeight()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return m_depthHeight;
}

void Intact::setSegBoundary(std::pair<Point, Point>& boundary)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    m_segBoundary = boundary;
}

std::pair<Point, Point> Intact::getSegBoundary()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return m_segBoundary;
}

void Intact::setTtpBoundary(std::pair<Point, Point>& boundary)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    m_ttopBoundary = boundary;
}

std::pair<Point, Point> Intact::getTtopBoundary()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return m_ttopBoundary;
}

void Intact::setSegPclBuf(
    int16_t* ptr_segPclBuf, int16_t* ptr_pclBuf, const int& pclSize)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    std::memcpy(ptr_segPclBuf, ptr_pclBuf, sizeof(int16_t) * pclSize);
    sptr_segPclBuf = std::make_shared<int16_t*>(ptr_segPclBuf);
}

std::shared_ptr<int16_t*> Intact::getSegPclBuf()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_segPclBuf;
}

void Intact::setSegImgBuf(
    uint8_t* ptr_segImgBuf, uint8_t* ptr_imgBuf, const int& imgSize)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    std::memcpy(ptr_segImgBuf, ptr_imgBuf, sizeof(uint8_t) * imgSize);
    sptr_segImgBuf = std::make_shared<uint8_t*>(ptr_segImgBuf);
}

std::shared_ptr<uint8_t*> Intact::getSegImgBuf()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_segImgBuf;
}

void Intact::setTtopImgBuf(
    uint8_t* ptr_ttpImgBuf, uint8_t* ptr_imgBuf, const int& imgSize)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    std::memcpy(ptr_ttpImgBuf, ptr_imgBuf, sizeof(uint8_t) * imgSize);
    sptr_ttopImgBuf = std::make_shared<uint8_t*>(ptr_ttpImgBuf);
}

std::shared_ptr<uint8_t*> Intact::getTtopImgBuf()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_ttopImgBuf;
}

void Intact::setRawPcl(const std::vector<float>& pcl)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_rawPcl = pcl;
}

std::shared_ptr<std::vector<float>> Intact::getRawPcl()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_rawPcl;
}

std::shared_ptr<std::vector<uint8_t>> Intact::getRawImg()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_rawImg;
}

void Intact::setRawImg(const std::vector<uint8_t>& img)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_rawImg = img;
}

void Intact::setRawPts(const std::vector<Point>& points) // todo what points?
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_rawPts = points;
}

std::shared_ptr<std::vector<float>> Intact::getSegPcl() // todo set?
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_segPcl;
}

std::shared_ptr<std::vector<uint8_t>> Intact::getSegImg() // todo set?
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_segImg;
}

void Intact::setSegFrame(cv::Mat& imgData) // todo check me please
{
    std::lock_guard<std::mutex> lck(m_mutex);
    sptr_segFrame = std::make_shared<cv::Mat>(imgData);
}

void Intact::setTtopFrame(cv::Mat& imgData)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    sptr_ttopFrame = std::make_shared<cv::Mat>(imgData);
}

std::shared_ptr<cv::Mat> Intact::getTtopFrame()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_ttopFrame;
}

std::shared_ptr<std::vector<Point>> Intact::getTtopPoints()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_ttopPts;
}

std::shared_ptr<std::vector<Point>> Intact::getSegPts()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_segPts;
}

void Intact::setSegPcl(const std::vector<float>& seg)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_segPcl = seg;
}

void Intact::setSegImg(const std::vector<uint8_t>& segment)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_segImg = segment;
}

void Intact::setSegPts(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_segPts = points;
}

void Intact::setClustPcl(const std::vector<float>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_clustPcl = points;
}

std::shared_ptr<std::vector<float>> Intact::getClustPcl()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_clustPcl;
}

void Intact::setClustImg(const std::vector<uint8_t>& img)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_clustImg = img;
}

std::shared_ptr<std::vector<uint8_t>> Intact::getClustImg()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_clustImg;
}

void Intact::setClustPts(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_clustPts = points;
}

std::shared_ptr<std::vector<Point>> Intact::getClustPts()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_clustPts;
}

void Intact::setTtopPcl(const std::vector<float>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_ttopPcl = points;
}

std::shared_ptr<std::vector<float>> Intact::getTtopPcl()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_ttopPcl;
}
void Intact::setTtopImg(const std::vector<uint8_t>& img)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_ttopImg = img;
}

std::shared_ptr<std::vector<uint8_t>> Intact::getTtopImg()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_ttopImg;
}

void Intact::setTtopPts(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_ttopPts = points;
}

// thread-safe semaphores
//
void Intact::raiseSegmentedFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_isSegmented = true;
}

void Intact::raiseChromakeyedFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_isChromakeyed = true;
}

void Intact::raiseClusteredFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_isClustered = true;
}

void Intact::raiseEpsilonFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_isEpsilonComputed = true;
}

void Intact::raiseRunFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_run = true;
}

void Intact::raiseStopFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_stop = true;
}

bool Intact::isRun()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_run;
}

bool Intact::isStop()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_stop;
}

void Intact::raiseKinectReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_isKinectReady = true;
}

bool Intact::isChromakeyed()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_isChromakeyed;
}

bool Intact::isKinectReady()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_isKinectReady;
}

bool Intact::isSegmented()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_isSegmented;
}

bool Intact::isClustered()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_isClustered;
}

// bool Intact::isCalibrated()
// {
//     std::lock_guard<std::mutex> lck(m_mutex);
//     return *sptr_isCalibrated;
// }
//
// void Intact::raiseCalibratedFlag()
// {
//     std::lock_guard<std::mutex> lck(m_mutex);
//     *sptr_isCalibrated = true;
// }

void Intact::stop()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_run = false;
}

#define WAIT_UNTIL_KINECT_READY                                                \
    while (!sptr_intact->isKinectReady()) {                                    \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define WAIT_UNTIL_SEGMENTATION_DONE                                           \
    while (!sptr_intact->isSegmented()) {                                      \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define WAIT_UNTIL_CLUSTERING_DONE                                             \
    while (!sptr_intact->isSegmented()) {                                      \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

#define WAIT_UNTIL_CHROMAKEY_DONE                                              \
    while (!sptr_intact->isChromakeyed()) {                                    \
        std::this_thread::sleep_for(std::chrono::milliseconds(3));             \
    }

void Intact::segment(std::shared_ptr<Intact>& sptr_intact)
{
#define SEGMENT 1
#if SEGMENT == 1

    bool init = true;
    WAIT_UNTIL_KINECT_READY; /*NOLINT*/
    while (sptr_intact->isRun()) {
        std::vector<Point> points = cast::toPoint(*sptr_intact->getRawPcl(),
            *sptr_intact->getRawImg(), sptr_intact->getNumPoints());
        std::vector<Point> seg = region::segment(points);
        std::pair<Point, Point> boundary = region::queryBoundary(seg);

        /** update shared raw points, tabletop points + boundary */
        sptr_intact->setRawPts(points);
        sptr_intact->setSegPts(seg);
        sptr_intact->setSegBoundary(boundary);

        /** update flow-control semaphores */
        if (init) {
            init = false;
            sptr_intact->raiseSegmentedFlag();
        }
    }
#endif
}

void Intact::calibrate(std::shared_ptr<Intact>& sptr_intact)
{
#define CALIBRATE 0
#if CALIBRATE == 1

    WAIT_UNTIL_KINECT_READY; /*NOLINT*/
    cv::Mat distanceCoefficients;
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    const std::string calibrationFile
        = "calibration.txt"; // calibration save file

    // uncomment #include "calibration.h"
    calibration::startChessBoardCalibration(cameraMatrix, distanceCoefficients);
    // This operation will loop infinitely until calibration
    // images have been taken! take at least 20 images of the
    // chessboard. A criteria for good calibration images is variable
    // chessboard poses equally across all 6 degrees of freedom.
    //

    /** grace period to successfully write calibration file */
    std::this_thread::sleep_for(std::chrono::seconds(5));

#endif

#define FIND_ARUCO 0
#if FIND_ARUCO == 1

    /** load the calibration */
    calibration::importCalibration(
        "calibration.txt", cameraMatrix, distanceCoefficients);
    /** detect aruco markers */
    calibration::findArucoMarkers(cameraMatrix, distanceCoefficients);

#endif
}

void Intact::approxEpsilon(const int& K, std::shared_ptr<Intact>& sptr_intact)
{
#define APPROXIMATE_EPSILON 0
#if APPROXIMATE_EPSILON

    WAIT_UNTIL_SEGMENTATION_DONE; /*NOLINT*/
    // ATM, dynamic epsilon estimation is unstable.
    // It must be done manually per specific projector-camera setup.
    //
    std::vector<Point> points = *sptr_intact->getSegPts();
    sptr_intact->raiseEpsilonFlag();

#endif
}

void Intact::cluster(
    const float& E, const int& N, std::shared_ptr<Intact>& sptr_intact)
{
#define WRITE_PLY_FILE 0
#if WRITE_PLY_FILE == 1
#include "ply.h"
#define WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(points) ply::write(points)
#else
#define WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(points)
#endif

#define CLUSTER 1
#if CLUSTER
    {
        WAIT_UNTIL_SEGMENTATION_DONE; /*NOLINT*/
        typedef std::pair<std::vector<float>, std::vector<uint8_t>> pclFmt;

        bool init = true;
        while (sptr_intact->isRun()) {
            std::vector<std::vector<Point>> clusters
                = dbscan::cluster(*sptr_intact->getSegPts(), E, N);

            /** convert clusters into objects */
            std::vector<Object> objects;
            for (auto& cluster : clusters) {
                Object object(cluster);
                objects.emplace_back(object);
            }

            /** points -> pcl format for rendering */
            std::vector<Point> tabletop = objects.front().m_points;
            pclFmt object = cast::toClusteredPcl(tabletop);

            std::vector<Point> allClusters = objects.back().m_points;
            pclFmt densityClusters = cast::toClusteredPcl(allClusters);

            std::pair<Point, Point> boundary
                = region::queryBoundary(objects.front().m_points);

            /** update shared density clusters and tabletop cluster */
            sptr_intact->setClustPcl(densityClusters.first);
            sptr_intact->setClustImg(densityClusters.second);
            sptr_intact->setClustPts(objects.back().m_points);
            sptr_intact->setTtopPcl(object.first);
            sptr_intact->setTtopImg(object.second);
            sptr_intact->setTtopPts(objects.front().m_points);
            sptr_intact->setTtpBoundary(boundary);

            /** update flow-control semaphore */
            if (init) {
                init = false;
                WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(
                    *sptr_intact->getClustPts()); /*NOLINT*/
                sptr_intact->raiseClusteredFlag();
            }
        }
    }

#endif
}

void Intact::render(std::shared_ptr<Intact>& sptr_intact)
{
#define RENDER 1
#if RENDER == 1

    WAIT_UNTIL_KINECT_READY; /*NOLINT*/
    viewer::draw(sptr_intact);

#endif
}

void Intact::chroma(std::shared_ptr<Intact>& sptr_intact)
{
#define CHROMAKEY 0
#if CHROMAKEY == 1

    int imgSize = sptr_intact->getNumPoints() * 4; // r, g, b, a
    int pclSize = sptr_intact->getNumPoints() * 3; // x, y, z

    auto* ptr_imgData = (uint8_t*)malloc(sizeof(uint8_t) * imgSize);
    auto* ptr_TabletopImgData = (uint8_t*)malloc(sizeof(uint8_t) * imgSize);

    auto* ptr_pclData = (int16_t*)malloc(sizeof(int16_t*) * pclSize);
    // auto* ptr_TabletopPclData = (int16_t*)malloc(sizeof(int16_t*) * pclSize);

    bool init = true;
    WAIT_UNTIL_CLUSTERING_DONE; /*NOLINT*/

    while (sptr_intact->isRun()) {
        std::memcpy(ptr_pclData, *sptr_intact->getSegPclBuf(),
            sizeof(int16_t) * pclSize);
        std::memcpy(ptr_imgData, *sptr_intact->getSegImgBuf(),
            sizeof(uint8_t) * imgSize);

        // todo:
        //   1. build pclVec
        //   2. build imgVec

        int numPoints = sptr_intact->getNumPoints();
        for (int i = 0; i < numPoints; i++) {
            if (ptr_imgData[4 * i + 0] == 0 && ptr_imgData[4 * i + 1] == 0
                && ptr_imgData[4 * i + 2] == 0 && ptr_imgData[4 * i + 3] == 0) {
                continue;
            }

            if (!vacant(i, ptr_pclData,
                    sptr_intact->getTabletopBoundary().first,
                    sptr_intact->getTabletopBoundary().second)) {
                continue;
            }
            chromaPixel(i, ptr_imgData);

            // float x = ptr_pclData[3 * i + 0];
            // float y = ptr_pclData[3 * i + 1];
            // float z = ptr_pclData[3 * i + 2];
            // Point queryPoint(x, y, z);
            // if (searcher::pointFound(*sptr_intact->getTabletopPoints(),
            // queryPoint)){
            //           chromaPixelData(i, ptr_imgData);
            // }
        }

        // sptr_intact->setTabletopPclData(
        //     ptr_TabletopPclData, ptr_pclData, pclSize);

        sptr_intact->setTtpImgBuf(ptr_TabletopImgData, ptr_imgData, imgSize);

        int height = sptr_intact->getDepthImgHeight();
        int width = sptr_intact->getDepthImgWidth();

        /** create image for segmented tabletop in cv::Mat format */
        cv::Mat frame = cv::Mat(height, width, CV_8UC4,
            (void*)*sptr_intact->getTtpImgBuf(), cv::Mat::AUTO_STEP)
                            .clone();

        sptr_intact->setTtpFrame(frame);

        /** update flow-control semaphore */
        if (init) {
            init = false;
            sptr_intact->raiseChromakeyedFlag();
        }
    }

#endif
}

void Intact::detectObjects(std::vector<std::string>& classnames,
    torch::jit::script::Module& module, std::shared_ptr<Intact>& sptr_intact)
{
#define DETECT_OBJECTS 0
#if DETECT_OBJECTS == 1

    WAIT_UNTIL_CHROMAKEY_DONE; /*NOLINT*/
    while (sptr_intact->isRun()) {
        clock_t start = clock();

        // cv::Mat frame = *sptr_intact->getSegmentedImgFrame();
        cv::Mat frame = *sptr_intact->getTtpFrame();

        cv::Mat img;
        /** prepare tensor input */
        cv::resize(frame, img, cv::Size(640, 384));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        torch::Tensor imgTensor = torch::from_blob(
            img.data, { img.rows, img.cols, 3 }, torch::kByte);
        imgTensor = imgTensor.permute({ 2, 0, 1 });
        imgTensor = imgTensor.toType(torch::kFloat);
        imgTensor = imgTensor.div(255);
        imgTensor = imgTensor.unsqueeze(0);

        // preds: [?, 15120, 9]
        torch::Tensor preds
            = module.forward({ imgTensor }).toTuple()->elements()[0].toTensor();
        std::vector<torch::Tensor> dets
            = yolo::non_max_suppression(preds, 0.4, 0.5);

        if (!dets.empty()) {
            /** show result */
            for (size_t i = 0; i < dets[0].sizes()[0]; ++i) {
                float left
                    = dets[0][i][0].item().toFloat() * (float)frame.cols / 640;
                float top
                    = dets[0][i][1].item().toFloat() * (float)frame.rows / 384;
                float right
                    = dets[0][i][2].item().toFloat() * (float)frame.cols / 640;
                float bottom
                    = dets[0][i][3].item().toFloat() * (float)frame.rows / 384;
                float score = dets[0][i][4].item().toFloat();
                int classID = dets[0][i][5].item().toInt();

                cv::rectangle(frame,
                    cv::Rect(left, top, (right - left), (bottom - top)),
                    cv::Scalar(0, 255, 0), 2);

                cv::putText(frame,
                    classnames[classID] + ": " + cv::format("%.2f", score),
                    cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX,
                    (right - left) / 200, cv::Scalar(0, 255, 0), 2);
            }
        }

#define SHOW_DETECTED_OBJECTS 1
#if SHOW_DETECTED_OBJECTS == 1
        cv::putText(frame,
            "FPS: " + std::to_string(int(1e7 / (double)(clock() - start))),
            cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
            cv::Scalar(0, 255, 0), 2);
        cv::imshow("", frame);
        if (cv::waitKey(1) == 27) {
            sptr_intact->raiseStopFlag();
        }
    }
#endif

#endif
}
