#include "segment.h"
#include "logger.h"
#include "outliers.h"
#include "polygon.h"
#include "scan.h"
#include "timer.h"

std::vector<Point> segment::partition(std::vector<Point>& points)
{
    Timer timer;
    /** coarse segmentation  */
    std::vector<Point> partition = polygon::fit(points);
    std::vector<Point> proposal = outliers::remove(partition);

    /** final proposal */
    //std::vector<Point> context = scan::density(proposal);

    LOG(INFO) << timer.getDuration() << " ms: final segmentation";
    //return context;
    return proposal;
}
