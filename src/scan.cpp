#include "scan.h"
#include "dbscan.h"
#include "knn.h"
#include "logger.h"
#include "timer.h"

std::vector<Point> largestDensity(std::pair<std::vector<Point>, int>& points)
{
    std::vector<std::vector<Point>> clusters(points.second);

    for (int i = 0; i <= points.second; ++i) {
        for (auto& point : points.first) {
            if (i == point.m_cluster) {
                clusters[i].push_back(point);
            }
        }
    }

    std::vector<Point> proposal;
    for (auto& cluster : clusters) {
        if (cluster.size() > proposal.size()) {
            proposal.clear();
            proposal = cluster;
        }
    }
    return proposal;
}

std::vector<Point> sample(std::vector<Point> points)
{
    Timer timer;
    const int SAMPLE_SIZE = 1000;

    /** find centroid */
    Point center = Point::centroid(points);

    /** compute distances from centroid for each point */
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

std::vector<Point> scan::density(std::vector<Point>& points)
{

    /** sample grown region */
    std::vector<Point> samplePoints = sample(points);

    /** accumulate distances to 4th nearest neighbours */
    std::vector<float> knn4 = knn::compute(samplePoints);

    /** find epsilon query size */
    // float epsilon = elbow::find(knn4);
    float epsilon = 90;
    LOG(INFO) << epsilon << ": ≈ epsilon";

    /** run: return clustered points and number of clusters */
    const int MIN_POINTS = 4;
    std::pair<std::vector<Point>, int> clusters
        = dbscan::cluster(points, MIN_POINTS, epsilon);

    std::vector<Point> proposal = largestDensity(clusters);
    return proposal;
}
