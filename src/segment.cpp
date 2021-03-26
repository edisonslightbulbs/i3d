#include <Eigen/Dense>
#include <utility>

#include "dbscan.h"
#include "jct.h"
#include "outliers.h"
#include "proposal.h"
#include "segment.h"
#include "svd.h"
#include "timer.h"

std::vector<Point> segment::cut(std::vector<Point>& points)
{
    Timer timer;
    float raw = points.size();

    /** filter out outliers */
    std::vector<Point> filtered = outliers::remove(points);
    std::string filterTime = timer.getDuration();

    /** compute svd */
    std::pair<Eigen::JacobiSVD<Eigen::MatrixXd>, Eigen::MatrixXd> solution;
    solution = svd::compute(filtered);

    /** coarse segment using svd solution */
    std::vector<Point> coarseSeg = proposal::grow(solution, filtered);

    /** deprecated: coarse segment using dbscan */
    // std::vector<Point> finalSeg = dbscan::run(filtered);

    /** deprecated: coarse segment using jordan curve theorem: */
    // std::vector<Point> finalSeg = jct::polygon(filtered);
    std::string coarseSegTime = timer.getDuration();

    /** remove straggling points */
    std::vector<Point> finalSeg = outliers::remove(coarseSeg);
    std::string finalSegTime = timer.getDuration();

    /** return final segment of tabletop interaction context*/
    return finalSeg;
}

/** log performance */
// io::performance(raw, filtered.size(), filterTime, coarseSeg.size(),
// coarseSegTime, finalSeg.size(), finalSegTime, timer.getDuration());
