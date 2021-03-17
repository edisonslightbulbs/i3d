#include "sampler.h"
#include "logger.h"
#include "timer.h"


std::vector<Point> sampler::sample(std::vector<Point> points)
{
    Timer timer;
    const int SAMPLE_SIZE = 1000;

    /** find centroid */
    Point center = Point::centroid(points);

    /** segment distances from centroid for each point */
    for (auto& point : points) {
        float distance = point.distance(center);
        point.m_distance.second = distance;
    }

    /** sort points */
    Point::sort(points);

    /** sample edge of run points */
    std::vector<Point> sample
            = std::vector<Point>(points.end() - SAMPLE_SIZE, points.end());

    LOG(INFO) << timer.getDuration() << " ms: sampling runtime (!unoptimized!)";

    return sample;
}
