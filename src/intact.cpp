#include <Eigen/Dense>
#include <chrono>
#include <thread>
#include <utility>

#include "dbscan.h"
#include "intact.h"
#include "io.h"
#include "kinect.h"
#include "knn.h"
#include "logger.h"
#include "outliers.h"
#include "ply.h"
#include "proposal.h"
#include "svd.h"
#include "timer.h"
#include "viewer.h"

// Developer options to disable/enable segmenting,
// epsilon evaluation, clustering and rendering
//
#define RENDER 1
#define CLUSTER 1
#define SEGMENT 1
#define COMPUTE_EPSILON 1

// Developer options to disable/enable,
// log tracing
//
#define LOG_TRACE 0
#if LOG_TRACE == 1
#define TRACE(string) LOG(INFO) << string
#else
#define TRACE
#endif

std::vector<Point> find(std::vector<Point>& points)
{
    /** filter out outliers */
    std::vector<Point> denoisedPcl = outliers::filter(points);

    /** compute svd */
    std::pair<Eigen::JacobiSVD<Eigen::MatrixXf>, Eigen::MatrixXf> USV;
    USV = svd::compute(denoisedPcl);

    /** coarse segment */
    std::vector<Point> coarseSeg = proposal::grow(USV, denoisedPcl);

    /** final segment */
    std::vector<Point> finalSeg = outliers::filter(coarseSeg);

    /** return interaction context */
    return finalSeg;
}

std::pair<Point, Point> Intact::queryContextBoundary(
    std::vector<Point>& context)
{
    std::vector<float> X(context.size());
    std::vector<float> Y(context.size());
    std::vector<float> Z(context.size());
    for (auto& point : context) {
        X.push_back(point.m_xyz[0]);
        Y.push_back(point.m_xyz[1]);
        Z.push_back(point.m_xyz[2]);
    }
    /** find the max and min points, viz,
     *   upper and lower point boundaries */
    float xMax = *std::max_element(X.begin(), X.end());
    float xMin = *std::min_element(X.begin(), X.end());
    float yMax = *std::max_element(Y.begin(), Y.end());
    float yMin = *std::min_element(Y.begin(), Y.end());
    float zMax = *std::max_element(Z.begin(), Z.end());
    float zMin = *std::min_element(Z.begin(), Z.end());
    Point min(xMin, yMin, zMin);
    Point max(xMax, yMax, zMax);
    return { min, max };
}

std::vector<Point> Intact::castPclToPoints(std::shared_ptr<Kinect>& sptr_kinect)
{
    std::vector<Point> pcl;
    for (int i = 0; i < sptr_kinect->m_numPoints; i++) {
        float x = (*sptr_kinect->getPcl())[3 * i + 0];
        float y = (*sptr_kinect->getPcl())[3 * i + 1];
        float z = (*sptr_kinect->getPcl())[3 * i + 2];

        if (x == 0 || y == 0 || z == 0) {
            continue;
        }
        std::vector<float> rgb(3);
        rgb[0] = (*sptr_kinect->getColor())[3 * i + 0];
        rgb[1] = (*sptr_kinect->getColor())[3 * i + 1];
        rgb[2] = (*sptr_kinect->getColor())[3 * i + 2];
        Point point(x, y, z);
        point.setColor(rgb);
        pcl.push_back(point);
    }
    return pcl;
}

void Intact::segmentContext(
    std::shared_ptr<Kinect>& sptr_kinect, std::shared_ptr<Intact>& sptr_intact)
{
#if SEGMENT
    /** capture point cloud using rgb-depth transformation */
    sptr_kinect->record(RGB_TO_DEPTH);
    bool firstRun = true;

    while (RUN_SYSTEM) {

        /** parse point cloud data into <Point> type */
        std::vector<Point> points = castPclToPoints(sptr_kinect);

        /** segment tabletop interaction context ~15ms */
        std::vector<Point> segment = find(points);
        sptr_intact->setContextPoints(segment);

        /** update flow control semaphores */
        if (firstRun) {
            firstRun = false;
            sptr_intact->setSegmentedFlag();
            TRACE("-- context segmented"); /*NOLINT*/
        }

        /** query interaction context boundary */
        std::pair<Point, Point> contextBoundary = queryContextBoundary(segment);

        /** register interaction context */
        sptr_kinect->setContextBounds(contextBoundary);

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
#endif
}

// Developer option to enable/disable
// printing knn results to terminal
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

// Developer option to enable/disable
// printing knn results to terminal
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
    sptr_intact->setEpsilonComputedFlag();
#endif
}

// Developer option to enable/disable
// writing ply file of segmented context
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
            sptr_intact->castPointsToPcl();
            TRACE("-- context points clustered and synchronized"); /*NOLINT*/

            if (firstRun) {
                firstRun = false;
                sptr_intact->setClusteredFlag();
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    WRITE_CLUSTERED_SEGMENT_TO_PLY_FILE(*sptr_intact->getPoints());
        /*NOLINT*/ // todo: testing
#endif
}

// Developer options to render *either
// segmented point cloud *or (not both)
// segmented + density clustered point cloud
//
#define RENDER_SEGMENTED 0
#define RENDER_CLUSTERED 1

void Intact::renderContext(
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
