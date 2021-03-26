#include "proposal.h"
#include <algorithm>

std::vector<Point> proposal::grow(
    const std::pair<Eigen::JacobiSVD<Eigen::MatrixXd>, Eigen::MatrixXd>&
        solution,
    std::vector<Point>& points)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd = solution.first;
    Eigen::MatrixXd pointsMat = solution.second;

    /** get normal */
    Eigen::MatrixXd normalX = svd.matrixU().col(0);
    Eigen::MatrixXd normalY = svd.matrixU().col(1);
    Eigen::MatrixXd normalZ = svd.matrixU().col(2);

    /** container for growing region of interest */
    std::vector<Point> region;

    /** define Argmin constraints */
    const float ARMGMIN_UPPER = 4.2;
    const float ARMGMIN_LOWER = -5.2;

    /** Iff point-norm form satisfied, point exists in
     *  cuboid proposal of tabletop interaction context */
    const int COL = 0;
    int index = 0;
    for (auto point : points) {
        Eigen::Vector3d v(pointsMat(index, xCol), pointsMat(index, yCol),
            pointsMat(index, zCol));
        Eigen::Vector3d n(
            normalX(index, COL), normalY(index, COL), normalZ(index, COL));
        float arg = n.dot(v);

        /** grow region */
        if (arg < ARMGMIN_UPPER && arg > ARMGMIN_LOWER) {
            region.push_back(point);
        }
        index++;
    }

    const float TABLETOP_CONTEXT_SIZE = 100;
    Point centroid = Point::centroid(region);
    std::vector<Point> segment;
    for (auto& point : region) {
        if (point.m_z < centroid.m_z + TABLETOP_CONTEXT_SIZE) {
            segment.push_back(point);
        }
    }
    return segment;
}
