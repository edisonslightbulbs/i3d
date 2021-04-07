#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <thread>
#include <utility>

#include "grow.h"
#include "intact.h"
#include "kinect.h"
#include "outliers.h"
#include "svd.h"
#include "viewer.h"
//#include "io.h"

extern std::mutex SYNCHRONIZE;
extern std::shared_ptr<bool> RUN_SYSTEM;

std::vector<Point> find(std::vector<Point>& points)
{
    /** copy unmodified point cloud */
    std::vector<Point> clone = points;

    /** filter out outliers */
    std::vector<Point> denoisedPcl = outliers::filter(points);

    /** compute svd */
    std::pair<Eigen::JacobiSVD<Eigen::MatrixXd>, Eigen::MatrixXd> USV;
    USV = svd::compute(denoisedPcl);

    /** coarse segment */
    std::vector<Point> coarseSeg = grow::propose(USV, denoisedPcl);

    /** final segment */
    std::vector<Point> finalSeg = outliers::filter(coarseSeg);

    /** output interaction context as ply file */
    // io::ply(points, finalSeg);

    /** return interaction context */
    return finalSeg;
}

std::vector<Point> intact::parse(
    std::shared_ptr<Kinect>& sptr_kinect, std::vector<float>& pcl)
{

    /** parse <float> cloud into <Point> cloud */
    std::vector<Point> pclPoints; // <- size not known ahead of time!!
    for (int i = 0; i < sptr_kinect->getNumPoints(); i++) {
        if (pcl[3 * i + 0] == 0 || pcl[3 * i + 1] == 0 || pcl[3 * i + 2] == 0) {
            continue;
        }
        float x = pcl[3 * i + 0];
        float y = pcl[3 * i + 1];
        float z = pcl[3 * i + 2];
        Point point(x, y, z);
        pclPoints.push_back(point);
    }
    return pclPoints;
}

std::pair<Point, Point> intact::queryContextBoundary(
    std::vector<Point>& context)
{

    std::vector<float> X(context.size());
    std::vector<float> Y(context.size());
    std::vector<float> Z(context.size());

    for (auto& point : context) {
        X.push_back(point.m_x);
        Y.push_back(point.m_y);
        Z.push_back(point.m_z);
    }

    /** find the max point and min point, viz,
     *   upper and lower context point boundaries */
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

void intact::segment(std::shared_ptr<Kinect>& sptr_kinect)
{
    /** capture point cloud */
    sptr_kinect->capturePcl();

    while (RUN_SYSTEM) {
        std::vector<float> pcl = *sptr_kinect->getPcl();

        /** parse point cloud data into Point type definitions */
        std::vector<Point> pclPoints = parse(sptr_kinect, pcl);

        /** segment tabletop interaction context */
        std::vector<Point> context = find(pclPoints);

        /** query interaction context boundary */
        std::pair<Point, Point> contextBoundary = queryContextBoundary(context);

        /** register interaction context */
        sptr_kinect->setContextBounds(contextBoundary);

        /** update interaction context constraints every second */
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}
void intact::render(std::shared_ptr<Kinect>& sptr_kinect)
{
    viewer::draw(sptr_kinect);
}
