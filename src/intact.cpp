#include <chrono>
#include <thread>
#include <utility>

#include "cast.h"
#include "dbscan.h"
#include "intact.h"
#include "kinect.h"
#include "knn.h"
#include "logger.h"
#include "object.h"
#include "region.h"
#include "timer.h"
#include "viewer.h"
// #include "io.h"
// #include "ply.h"

// Developer option:
// log tracing
//
#define LOG_TRACE 0
#if LOG_TRACE == 1
#define TRACE(string) LOG(INFO) << string
#else
#define TRACE
#endif

void Intact::adapt()
{
    std::vector<Point> points;
    {
        std::lock_guard<std::mutex> lck(m_mutex);
        points = *sptr_rawPoints;
    }
    std::pair<std::vector<float>, std::vector<uint8_t>> pcl
        = cast::toPcl(points);

    /** safely update context */
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_region = pcl.first;
    *sptr_regionColor = pcl.second;
}

void Intact::setRawPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_rawPoints = points;
}

void Intact::setSegmentPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_segmentPoints = points;
}

void Intact::setClusterPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_clusterPoints = points;
}

std::shared_ptr<std::vector<Point>> Intact::getPoints()
{
    std::shared_lock lock(s_mutex);
    return sptr_rawPoints;
}

int Intact::getNumPoints()
{
    std::shared_lock lock(s_mutex);
    return m_numPoints;
}

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

void Intact::setSegment(
    std::pair<std::vector<float>, std::vector<uint8_t>>& segment)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_segment = segment.first;
    *sptr_segmentColor = segment.second;
}

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

void Intact::raiseSegFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_isContextSegmented = true;
}

void Intact::raiseClustFlag()
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
    std::shared_lock lock(s_mutex);
    return *sptr_run;
}

bool Intact::isStop()
{
    std::shared_lock lock(s_mutex);
    return *sptr_stop;
}

void Intact::raiseIntactReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_isIntactReady = true;
}

bool Intact::isIntactReady()
{
    std::shared_lock lock(s_mutex);
    return *sptr_isIntactReady;
}

bool Intact::isSegmented()
{
    std::shared_lock lock(s_mutex);
    return *sptr_isContextSegmented;
}

bool Intact::isClustered()
{
    std::shared_lock lock(s_mutex);
    return *sptr_isContextClustered;
}

bool Intact::isEpsilonComputed()
{
    std::shared_lock lock(s_mutex);
    return *sptr_isEpsilonComputed;
}

#define SEGMENT 1
void Intact::segment(
    std::shared_ptr<Kinect>& sptr_kinect, std::shared_ptr<Intact>& sptr_intact)
{
#if SEGMENT
    bool init = true;
    while (!sptr_intact->isIntactReady()) {
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
        setBoundary(boundary);
        sptr_intact->setSegmentPoints(seg); // <- update segment points

        /** update flow control semaphores */
        if (init) {
            init = false;
            sptr_intact->raiseSegFlag();
            TRACE("-- context segmented"); /*NOLINT*/
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#endif
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

#define COMPUTE_EPSILON 0
void Intact::estimateEpsilon(const int& K, std::shared_ptr<Intact>& sptr_intact)
{
#if COMPUTE_EPSILON
    /** wait for segmented context */
    while (!sptr_intact->isSegmented()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }

    TRACE("-- evaluating k nearest neighbours"); /*NOLINT*/
    std::vector<Point> points = *sptr_intact->getPoints();

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
#define WRITE_PLY_FILE 1
#if WRITE_PLY_FILE == 1
#define WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(points) ply::write(points)
#else
#define WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(points)
#endif

#define CLUSTER 0
void Intact::cluster(
    const float& E, const int& N, std::shared_ptr<Intact>& sptr_intact)
{
#if CLUSTER
    {
        /** wait for epsilon value */
        while (!sptr_intact->isEpsilonComputed()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        bool firstRun = true;
        /** n.b., clustering loop takes ~130ms per iteration */
        while (RUN_SYSTEM) {

            /** cluster segmented context ~130ms/loop iteration */
            std::vector<std::vector<Point>> clusters
                = dbscan::cluster(*sptr_intact->getPoints(), E, N);

            /** create objects using density clusters */
            std::vector<Object> objects;
            for (auto& cluster : clusters) {
                Object object(cluster);
                objects.emplace_back(object);
            }

            /** preprocess points for rendering */
            sptr_intact->setPoints(objects[0].m_points);
            sptr_intact->adapt();

            if (firstRun) {
                firstRun = false;
                sptr_intact->raiseClustFlag();
                WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(
                    *sptr_intact->getPoints()); /*NOLINT*/
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
#endif
}

void Intact::render(
    std::shared_ptr<Kinect>& sptr_kinect, std::shared_ptr<Intact>& sptr_intact)
{
    while (!sptr_intact->isIntactReady()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    /** render context in real-time (un-clustered)  */
    viewer::draw(sptr_intact, sptr_kinect);
}
