#include "segment.h"
#include "edge.h"
#include "logger.h"
#include "outliers.h"
#include "polygon.h"
#include "scan.h"
#include "timer.h"

std::vector<Point> segment::cut(std::vector<Point>& points)
{
    Timer timer;
    /** final segmentation: extrapolating 'vanishing' depth values,  very fast [
     * robust ]  */
    std::vector<Point> proposal = edge::detect(points);

    /** removing floating points */
    std::vector<Point> denoised = outliers::remove(proposal);

    /** final segmentation: dbscan-based, slow [ robust ] */
    //std::vector<Point> clusters = scan::density(proposal);

    /** final segmentation:  jordan curve-based, slow [ unreliable ] */
    // std::vector<Point> proposal = polygon::fit(points);

    LOG(INFO) << timer.getDuration() << " ms: final segmentation";
    return denoised;
}
