#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>

#include "intact.h"
#include "kinect.h"
#include "logger.h"
#include "outliers.h"
#include "proposal.h"
#include "svd.h"
#include "viewer.h"

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
    /** capturing point cloud and use rgb to depth transformation */
    sptr_kinect->record(RGB_TO_DEPTH);

    while (RUN_SYSTEM) {

        /** parse point cloud data into <Point> type */
        std::vector<Point> points = parsePcl(sptr_kinect);

        /** segment tabletop interaction context */
        std::vector<Point> context = find(points);

        /** query interaction context boundary */
        std::pair<Point, Point> contextBoundary = queryContextBoundary(context);

        /** register interaction context */
        sptr_kinect->setContextBounds(contextBoundary);

        /** update interaction context constraints every 40 milliseconds */
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
}

void intact::render(std::shared_ptr<Kinect>& sptr_kinect)
{
    /** render in real-time */
    viewer::draw(sptr_kinect);
}
