#include "segment.h"
#include "edge.h"
#include "outliers.h"
#include "svd.h"
#include "timer.h"

std::vector<Point> segment::cut(std::vector<Point>& points)
{
    Timer timer;
    float raw = points.size();

    /** remove outliers*/
    std::vector<Point> filtered = outliers::remove(points);
    std::string filterTime = timer.getDuration();

    /** grow course segment  */
    std::vector<Point> region = svd::compute(filtered);
    std::string growTime = timer.getDuration();

    /** final segmentation: extrapolating 'vanishing' depth values */
    std::vector<Point> proposal = edge::detect(region);

    /** removing straggling points */
    std::vector<Point> finalSeg = outliers::remove(proposal);
    std::string finalSegTime = timer.getDuration();

    /** log performance */
    // io::performance(raw, filtered.size(), filterTime, region.size(),
    //                growTime, finalSeg.size(), finalSegTime,
    //                timer.getDuration());

    /** return segment of tabletop interaction context*/
    return finalSeg;
}
