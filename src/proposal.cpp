#include "proposal.h"

std::vector<Point> proposal::grow(const std::pair<Eigen::JacobiSVD<Eigen::MatrixXd>, Eigen::MatrixXd>& svd, std::vector<Point>& points)
{
    /** get normal */
    Eigen::MatrixXd normalMat = svd.first.matrixV().col(2);
    Eigen::Vector3d n(normalMat(0, 0), normalMat(1, 0), normalMat(2, 0));

    /** container for growing region of interest */
    std::vector<Point> region;

    /** define Argmin constraints */
    const double E_MAX = 350.0;
    const double E_MIN = -350.0;

    /** Iff point-norm form satisfied, point exists in
     *  cuboid proposal of tabletop interaction context */
    int index = 0;
    for (auto point : points) {
        Eigen::Vector3d v(svd.second(index, xCol), svd.second(index, yCol),
            svd.second(index, zCol));

        /** grow region */
        if (n.dot(v) < E_MAX && n.dot(v) > E_MIN) {
            region.push_back(point);
        }
        index++;
    }
    return region;
}