#include <vector>
#include "outliers.h"
#include "segment.h"
#include "timer.h"
#include "logger.h"
#include "edge.h"
#include "scan.h"


std::vector<Point> segment::partition(std::vector<Point>& points)
{
    Timer timer;
    /** partition proposal */
    //std::vector<Point> proposal = edge::find(points);
    std::vector<Point> proposal = scan::density(points);

    /** remove ``residual'' outliers in final context segment */
    std::vector<Point> context = outliers::remove(proposal);
    LOG(INFO) << timer.getDuration() << " ms: final segmentation";
    return context;
}