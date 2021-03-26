#include <Eigen/Dense>
#include <utility>

#include "dbscan.h"
#include "edge.h"
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

    /** grow course segment using svd solution */
    std::vector<Point> region = proposal::grow(solution, filtered);
    std::string growTime = timer.getDuration();

    /** option 1 (robust and fast): do final segmentation using vanishing points
     */
    std::vector<Point> finalSeg = edge::detect(region);

    // /** option 2 (moderately robust & slow): do final segmentation using
    // dbscan */ std::vector<Point> finalSeg = dbscan::run(region);

    // /** option 3 (not robust & slow): do final segmentation using JCT (Jordan
    // curve theorem) */ std::vector<Point> finalSeg = jct::polygon(region);

    /** remove straggling points */
    std::vector<Point> denoisedFinalSeg = outliers::remove(finalSeg);
    std::string finalSegTime = timer.getDuration();

    /** return final segment of tabletop interaction context*/
    return denoisedFinalSeg;
}

/** log performance */
// io::performance(raw, filtered.size(), filterTime, region.size(),
//                growTime, finalSeg.size(), finalSegTime,
//                timer.getDuration());
