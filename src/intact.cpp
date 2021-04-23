#include <chrono>
#include <thread>
#include <utility>

#include "cast.h"
#include "dbscan.h"
#include "intact.h"
#include "kinect.h"
#include "knn.h"
#include "logger.h"
#include "region.h"
#include "timer.h"
#include "viewer.h"

// Developer IO utility:
//
#include "io.h"
#include "ply.h"

// Developer option:
// epsilon evaluation
//
#define RENDER 1
#define CLUSTER 1
#define SEGMENT 1
#define COMPUTE_EPSILON 1

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
        points = *sptr_points;
    }
    std::pair<std::vector<float>, std::vector<uint8_t>> pcl
        = cast::toPcl(points);

    /** safely update context */
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_context = pcl.first;
    *sptr_color = pcl.second;
}

void Intact::setContextPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_points = points;
}

void Intact::setNumClusters(const int& clusters)
{
    std::lock_guard<std::mutex> lck(m_mutex);
    *sptr_numClusters = clusters;
}

std::shared_ptr<std::vector<Point>> Intact::getPoints()
{
    std::shared_lock lock(s_mutex);
    return sptr_points;
}

std::shared_ptr<std::vector<float>> Intact::getContext()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_context;
}

std::shared_ptr<std::vector<uint8_t>> Intact::getColor()
{
    std::lock_guard<std::mutex> lck(m_mutex);
    return sptr_color;
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

void Intact::segment(
    std::shared_ptr<Kinect>& sptr_kinect, std::shared_ptr<Intact>& sptr_intact)
{
#if SEGMENT
    /** capture point cloud using rgb-depth transformation */
    sptr_kinect->record(RGB_TO_DEPTH);
    bool firstRun = true;

    while (RUN_SYSTEM) {

        /** parse point cloud data into <Point> type */
        std::vector<Point> points = cast::toPoint(*sptr_kinect->getPcl(),
            *sptr_kinect->getColor(), sptr_kinect->m_numPoints);

        /** segment tabletop interaction context ~15ms */
        std::vector<Point> seg = region::segment(points);
        sptr_intact->setContextPoints(seg);

        /** update flow control semaphores */
        if (firstRun) {
            firstRun = false;
            sptr_intact->raiseSegFlag();
            TRACE("-- context segmented"); /*NOLINT*/
        }

        /** query interaction context boundary */
        std::pair<Point, Point> contextBoundary = region::queryBoundary(seg);

        /** register interaction context */
        sptr_kinect->setContextBounds(contextBoundary);

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

void Intact::cluster(
    const float& E, const int& N, std::shared_ptr<Intact>& sptr_intact)
{
#if CLUSTER
    {
        /** wait for epsilon value */
        while (!sptr_intact->isEpsilonComputed()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        /** cluster segmented interaction context ~130ms/ loop run-through */
        std::vector<Point> points;
        std::pair<std::vector<Point>, int> clusteredContext;
        bool firstRun = true;
        while (RUN_SYSTEM) {
            points = *sptr_intact->getPoints();
            clusteredContext = dbscan::cluster(points, E, N);
            sptr_intact->setContextPoints(clusteredContext.first);
            sptr_intact->setNumClusters(clusteredContext.second);
            sptr_intact->adapt();
            TRACE("-- context points clustered and synchronized"); /*NOLINT*/

            if (firstRun) {
                firstRun = false;
                sptr_intact->raiseClustFlag();
                WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(
                    *sptr_intact->getPoints()); /*NOLINT*/
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#endif
}

// Developer options:
// option (1) render segment
// option (2) render clusters
//
#define RENDER_SEGMENTED 0
#define RENDER_CLUSTERED 1

void Intact::render(
    std::shared_ptr<Kinect>& sptr_kinect, std::shared_ptr<Intact>& sptr_intact)
{
#if RENDER
#if RENDER_SEGMENTED
    /** render interaction context in real-time (un-clustered)  */
    viewer::draw(sptr_kinect);
#endif
    /** wait for epsilon value */
    while (!sptr_intact->isClustered()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

#if RENDER_CLUSTERED
    /** render interaction context in real-time (clustered)  */
    viewer::draw(sptr_intact, sptr_kinect);
#endif
#endif
}
