#include "segment.h"
#include "logger.h"
#include "edge.h"
#include "outliers.h"
#include "polygon.h"
#include "scan.h"
#include "timer.h"

std::vector<Point> segment::cut(std::vector<Point>& points)
{
    Timer timer;
    /** final segmentation: extrapolating 'vanishing' depth values,  very fast [ robust ]  */
    std::vector<Point> proposal = edge::detect(points);

    /** final segmentation:  jordan curve-based, slow [ unreliable ] */
    //std::vector<Point> proposal = polygon::fit(points);

    /** final segmentation: dbscan-based, slow [ robust ] */
    //std::vector<Point> proposal = scan::density(proposal);

    /** removing floating points */
    std::vector<Point> denoised = outliers::remove(points);

    LOG(INFO) << timer.getDuration() << " ms: final segmentation";
    return denoised;
}
