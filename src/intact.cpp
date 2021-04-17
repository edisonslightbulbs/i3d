#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>

#include "dbscan.h"
#include "intact.h"
#include "io.h"
#include "kinect.h"
#include "knn.h"
#include "logger.h"
#include "outliers.h"
#include "proposal.h"
#include "svd.h"
#include "viewer.h"

std::mutex intact_mutex;
std::shared_mutex sharedIntact_mutex;
[[maybe_unused]] Context CONTEXT; /*NOLINT*/

Context* intact::getContext()
{
    /** allow multiple threads to read context */
    std::shared_lock lock(sharedIntact_mutex);
    if (CONTEXT.m_points.empty()) {
        return nullptr;
    }
    return &CONTEXT;
}

extern std::shared_ptr<bool> RUN_SYSTEM;

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

std::vector<Point> intact::parsePcl(std::shared_ptr<Kinect>& sptr_kinect)
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

std::pair<Point, Point> intact::queryContextBoundary(
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

void intact::segmentContext(std::shared_ptr<Kinect>& sptr_kinect)
{
    /** capture point cloud using rgb-depth transformation */
    sptr_kinect->record(RGB_TO_DEPTH);
    while (RUN_SYSTEM) {

        /** parse point cloud data into <Point> type */
        std::vector<Point> points = parsePcl(sptr_kinect);

        /** segment tabletop interaction context */
        std::vector<Point> segment = find(points);
        {
            /** block threads from accessing
             *  CONTEXT during update */
            std::lock_guard<std::mutex> lck(intact_mutex);
            CONTEXT = Context(segment);
        }

        /** query interaction context boundary */
        std::pair<Point, Point> contextBoundary = queryContextBoundary(segment);

        /** register interaction context */
        sptr_kinect->setContextBounds(contextBoundary);

        /** update interaction context constraints every 40 milliseconds */
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}

void intact::render(std::shared_ptr<Kinect>& sptr_kinect)
{
    /** render in real-time */
    viewer::draw(sptr_kinect);
}

float estimateEpsilon(std::vector<Point>& points)
{
    /** query the K nearest neighbour of every point */
    std::vector<float> knnQuery;
    int k = 5;
    const int testVal = 3;
    // testVal corresponds to an arbitrary number of points for testing
    // in theory, we should find the knns of every point. This is
    // typically expensive, even with flann's impressive optimizations.
    //
    for (int i = 0; i < testVal; i++) {
        int indexOfQueryPoint = i;
        std::vector<std::pair<Point, float>> nn
            = knn::compute(points, k, indexOfQueryPoint);
        knnQuery.push_back(std::sqrt(nn[i].second));

        std::cout << "#" << i << ",\t"
                  << "dist: " << std::sqrt(nn[i].second) << ",\t"
                  << "point: (" << nn[i].first.m_xyz[0] << ", "
                  << nn[i].first.m_xyz[1] << ", " << nn[i].first.m_xyz[2]
                  << ") " << std::endl;
    }

    std::sort(knnQuery.begin(), knnQuery.end(), std::greater<>());
    const std::string file = io::pwd() + "/knn.csv";
    std::cout << "writing the knn (k=4) of every point to: ";
    std::cout << file << std::endl;
    io::write(knnQuery, file);
    // an important next step here is automate determining epsilon
    // Currently the 4th nearest neighbour distance are output  to
    // to a file and analyzed in matlab to visually extract the maximum
    // curvature elbow.
    //
    return 0;
}

void densityScan(std::vector<Point>& points)
{
    // update clustered flag here
    // update CONTEXT points here
}

void intact::cluster(const float& epsilon)
{
    std::cout << "hello brave new clustering thread !!! " << std::endl;
    while (getContext() == nullptr) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (getContext() != nullptr) {
        std::vector<Point> points = getContext()->m_points;

        /** estimate epsilon value */
        // estimateEpsilon(points);

        /** cluster point cloud */
        std::pair<std::vector<Point>, int> clusters = dbscan::original(points);
        // then do dbscan
    }
}
