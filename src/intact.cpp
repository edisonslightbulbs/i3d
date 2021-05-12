#include <chrono>
#include <opencv2/core.hpp>
#include <thread>
#include <utility>

#include "calibration.h"
#include "cast.h"
#include "dbscan.h"
#include "intact.h"
#include "io.h"
#include "kinect.h"
#include "knn.h"
#include "logger.h"
#include "object.h"
#include "ply.h"
#include "region.h"
#include "timer.h"
#include "viewer.h"
#include "yolov5.h"

// Developer option:
// log tracing
//
#define LOG_TRACE 0
#if LOG_TRACE == 1
#define TRACE(string) LOG(INFO) << string
#else
#define TRACE
#endif

// number of point-cloud points form kinect
//
int Intact::getNumPoints()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return m_numPoints;
}

// raw point cloud thread-safe setters and getters
//
std::shared_ptr<std::vector<float>> Intact::getRaw()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_raw;
}

std::shared_ptr<std::vector<uint8_t>> Intact::getRawColor()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_rawColor;
}

void Intact::setRaw(const std::vector<float>& pcl)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_raw = pcl;
}

void Intact::setRawColor(const std::vector<uint8_t>& color)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_rawColor = color;
}

std::shared_ptr<std::vector<Point>> Intact::getRawPoints()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_rawPoints;
}

void Intact::setRawPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_rawPoints = points;
}

// interaction context thread-safe setters and getters
//
std::shared_ptr<std::vector<float>> Intact::getSegment()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_segment;
}

std::shared_ptr<std::vector<uint8_t>> Intact::getSegmentColor()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_segmentColor;
}

std::shared_ptr<std::vector<Point>> Intact::getSegmentPoints()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_segmentPoints;
}

void Intact::setSegment(const std::vector<float>& segment)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_segment = segment;
}

void Intact::setSegmentColor(const std::vector<uint8_t>& color)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_segmentColor = color;
}

void Intact::setSegmentPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_segmentPoints = points;
}

// point cloud density region thread-safe setters and getters
//
std::shared_ptr<std::vector<float>> Intact::getRegion()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_region;
}

std::shared_ptr<std::vector<uint8_t>> Intact::getRegionColor()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_regionColor;
}

void Intact::setRegion(const std::vector<float>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_region = points;
}
void Intact::setRegionColor(const std::vector<uint8_t>& color)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_regionColor = color;
}

void Intact::setRegionPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_regionPoints = points;
}

std::shared_ptr<std::vector<Point>> Intact::getRegionPoints()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_regionPoints;
}

// object point cloud  thread-safe setters and getters
//
void Intact::setObject(const std::vector<float>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_object = points;
}
void Intact::setObjectColor(const std::vector<uint8_t>& color)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_objectColor = color;
}

void Intact::setObjectPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_objectPoints = points;
}

std::shared_ptr<std::vector<float>> Intact::getObject()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_object;
}

std::shared_ptr<std::vector<uint8_t>> Intact::getObjectColor()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_objectColor;
}

// cross-thread semaphores
//
void Intact::raiseSegmentedFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_isContextSegmented = true;
}

void Intact::raiseClusteredFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_isContextClustered = true;
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

void Intact::stop()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_run = false;
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

bool Intact::isKinectReady()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_isKinectReady;
}

bool Intact::isSegmented()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_isContextSegmented;
}

bool Intact::isClustered()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_isContextClustered;
}

bool Intact::isEpsilonComputed()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_isEpsilonComputed;
}

bool Intact::isCalibrated()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return *sptr_isCalibrated;
}

void Intact::raiseCalibratedFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_isCalibrated = true;
}

std::pair<Point, Point> Intact::getSegmentBoundary()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return m_segmentBoundary;
}

#define SEGMENT 1
void Intact::segment(std::shared_ptr<Intact>& sptr_intact)
{
#if SEGMENT == 1
    bool init = true;
    while (!sptr_intact->isKinectReady()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    while (sptr_intact->isRun()) {
        /** cast point cloud to Point type definition for processing */
        std::vector<Point> points = cast::toPoint(*sptr_intact->getRaw(),
            *sptr_intact->getRawColor(), sptr_intact->getNumPoints());
        sptr_intact->setRawPoints(points); // <- update raw points

        /** segment tabletop interaction context ~15ms */
        std::vector<Point> seg = region::segment(points);
        std::pair<Point, Point> boundary = region::queryBoundary(seg);
        setSegmentBoundary(boundary);
        sptr_intact->setSegmentPoints(seg); // <- update segment points

        /** update flow control semaphores */
        if (init) {
            init = false;
            sptr_intact->raiseSegmentedFlag();
            TRACE("-- context segmented"); /*NOLINT*/
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#endif
}

void Intact::calibrate(std::shared_ptr<Intact>& sptr_intact)
{
    while (!sptr_intact->isKinectReady()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    cv::Mat distanceCoefficients;
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    // todo: move user specification to interface
    const float arucoSquareEdgeLength = 0.0565f;        // in meters
    const float calibrationSquareEdgeLength = 0.02500f; // in meters
    const std::string calibrationFile
        = "calibration.txt"; // external file for saving calibration

#define CALIBRATE 0
#if CALIBRATE == 1
    /** 1st calibrate the camera */
    calibration::startChessBoardCalibration(cameraMatrix, distanceCoefficients);
    // This operation will loop infinitely until calibration
    // images have been taken! take at least 20 images of the
    // chessboard. A criteria for good calibration images is variable
    // chessboard poses equally across all 6 degrees of freedom.
    //

    /** grace window for writing calibration file */
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
    std::cout << "-- calibration done!!" << std::endl;
}

// Developer option:
// print knn results
//
#define PRINT_KNN 0
void printKnn(int& i, std::vector<std::pair<Point, float>>& nn)
{
#if PRINT_KNN == 1
    std::cout << "#" << i << ",\t"
              << "dist: " << std::sqrt(nn[i].second) << ",\t"
              << "point: (" << nn[i].first.m_xyz[0] << ", "
              << nn[i].first.m_xyz[1] << ", " << nn[i].first.m_xyz[2] << ")"
              << std::endl;
#endif
}

// Developer option:
// write knn results to file
//
#define WRITE_KNN 0
void writeKnn(std::vector<float>& knnQuery)
{
#if WRITE_KNN == 1
    std::sort(knnQuery.begin(), knnQuery.end(), std::greater<>());
    const std::string file = io::pwd() + "/knn.csv";
    std::cout << "writing the knn (k=4) of every point to: ";
    std::cout << file << std::endl;
    io::write(knnQuery, file);
#endif
}

#define COMPUTE_EPSILON 1
void Intact::estimateEpsilon(const int& K, std::shared_ptr<Intact>& sptr_intact)
{
#if COMPUTE_EPSILON
    /** wait for segmented context */
    while (!sptr_intact->isSegmented()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    TRACE("-- evaluating k nearest neighbours"); /*NOLINT*/
    std::vector<Point> points = *sptr_intact->getSegmentPoints();

    const int testVal = 3;
    // testVal used for arbitrary test for release, use
    // points.size()) (computationally expensive task)
    //

    /** evaluate k=4th nearest neighbours for every point */
    std::vector<float> knnQuery;
    for (int i = 0; i < testVal; i++) {
        int indexOfQueryPoint = i;
        std::vector<std::pair<Point, float>> nn
            = knn::compute(points, K, indexOfQueryPoint);
        knnQuery.push_back(std::sqrt(nn[i].second));
        printKnn(i, nn);
    }
    writeKnn(knnQuery);
    sptr_intact->raiseEpsilonFlag();
#endif
}

// Developer option:
// write segmented context to ply file
//
#define WRITE_PLY_FILE 0
#if WRITE_PLY_FILE == 1
#define WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(points) ply::write(points)
#else
#define WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(points)
#endif

#define CLUSTER 1
void Intact::cluster(
    const float& E, const int& N, std::shared_ptr<Intact>& sptr_intact)
{
#if CLUSTER
    {
        /** wait for epsilon value */
        while (!sptr_intact->isEpsilonComputed()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        bool init = true;
        /** n.b., clustering loop takes ~130ms per iteration */
        while (sptr_intact->isRun()) {

            /** cluster segmented context ~130ms/loop iteration */
            std::vector<std::vector<Point>> clusters
                = dbscan::cluster(*sptr_intact->getSegmentPoints(), E, N);

            /** create objects using view clusters */
            std::vector<Object> objects;
            for (auto& cluster : clusters) {
                Object object(cluster);
                objects.emplace_back(object);
            }

            /** cast region points to pcl format  for rendering */
            std::pair<std::vector<float>, std::vector<uint8_t>> region
                = cast::toClusteredPcl(objects.back().m_points);
            sptr_intact->setRegion(region.first);
            sptr_intact->setRegionColor(region.second);
            sptr_intact->setRegionPoints(objects.back().m_points);

            /** cast object points to pcl format  for rendering */
            std::pair<std::vector<float>, std::vector<uint8_t>> object
                = cast::toClusteredPcl(objects.front().m_points);
            sptr_intact->setObject(object.first);
            sptr_intact->setObjectColor(object.second);
            sptr_intact->setObjectPoints(objects.front().m_points);

            /** sequence cross thread semaphore */
            if (init) {
                init = false;
                WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(
                    *sptr_intact->getRegionPoints()); /*NOLINT*/
                sptr_intact->raiseClusteredFlag();
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#endif
}

#define RENDER 0
void Intact::render(std::shared_ptr<Intact>& sptr_intact)
{
#if RENDER == 1
    while (!sptr_intact->isKinectReady()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    viewer::draw(sptr_intact);
#endif
}

#define DETECT_OBJECTS 1
void Intact::detectObjects(std::vector<std::string>& classnames,
    torch::jit::script::Module module, cv::VideoCapture& cap, cv::Mat& frame,
    cv::Mat& img, std::shared_ptr<Intact>& sptr_intact)
{

#if DETECT_OBJECTS == 1
    while (cap.isOpened()) {
        clock_t start = clock();
        cap.read(frame);

        // Preparing input tensor
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
            // Visualize result
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
        cv::putText(frame,
            "FPS: " + std::to_string(int(1e7 / (double)(clock() - start))),
            cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
            cv::Scalar(0, 255, 0), 2);
        cv::imshow("", frame);
        if (cv::waitKey(1) == 27) {
            sptr_intact->raiseStopFlag();
            break;
        }
    }
#endif
}
