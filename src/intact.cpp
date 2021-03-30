#include <Eigen/Dense>
#include <utility>
#include <mutex>
#include <atomic>

#include "dbscan.h"
#include "jct.h"
#include "outliers.h"
#include "io.h"
#include "grow.h"
#include "intact.h"
#include "svd.h"
#include "kinect.h"
#include "timer.h"
#include "viewer.h"

std::atomic<bool> RUN(true);

std::vector<Point> update(std::vector<Point>& points)
{
    //kinect.capture();
    //kinect.pclImage();
    float raw = points.size();

    Timer timer;
    /** filter out outliers */
    std::vector<Point> filtered = outliers::remove(points);
    std::string filterTime = timer.getDuration();

    /** compute svd */
    std::pair<Eigen::JacobiSVD<Eigen::MatrixXd>, Eigen::MatrixXd> solution;
    solution = svd::compute(filtered);

    /** define coarse segment */
    std::vector<Point> coarseSeg = grow::propose(solution, filtered);
    std::string coarseSegTime = timer.getDuration();

    /** deprecated: coarse segment using dbscan */
    // std::vector<Point> finalSeg = dbscan::run(filtered);

    /** deprecated: coarse segment using jordan curve theorem: */
    // std::vector<Point> finalSeg = jct::polygon(filtered);

    /** remove straggling points */
    std::vector<Point> finalSeg = outliers::remove(coarseSeg);
    std::string finalSegTime = timer.getDuration();

    /** output final segment of tabletop interaction context*/
    // io::ply(kinect.m_points, finalSeg);
    return finalSeg;
}

/** log performance */
// io::performance(raw, filtered.size(), filterTime, coarseSeg.size(),
// coarseSegTime, finalSeg.size(), finalSegTime, timer.getDuration());

void intact::segment(std::mutex& m, Kinect& kinect, const int& numPoints,
                std::shared_ptr<std::vector<float>>& sptr_points, std::shared_ptr<std::pair<Point, Point>>& sptr_threshold)
{
    /** capture once and hand over capturing task to renderer */
    kinect.capture();
    kinect.pclImage(sptr_points);

    while(RUN){
        /** capture using kinect */
        std::vector<Point> pcl;

        std::vector<float> raw;
        if (m.try_lock()) {
            kinect.capture();
            kinect.pclImage(sptr_points);
            raw = *sptr_points;
            m.unlock();
        }

        /** parse point cloud into std::vector<Point> */
        int width = k4a_image_get_width_pixels(kinect.m_pclImage);
        int height = k4a_image_get_height_pixels(kinect.m_pclImage);

        for (int i = 0; i < width * height; i++) {
            if (raw[3 * i + 0] == 0  || raw[3 * i + 1] == 0
                || raw[3 * i + 2] == 0){
                continue;
            }
         float x = raw[3 * i + 0];
         float y = raw[3 * i + 1];
         float z = raw[3 * i + 2];
         Point point(x, y, z);
         pcl.push_back(point);
        }

        /** segment tabletop interaction context */
        std::vector<Point> points = update(pcl);
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> z;
        for (const auto& point : points) {
            x.push_back(point.m_x);
            y.push_back(point.m_y);
            z.push_back(point.m_z);
        }
        float xMax = *std::max_element(x.begin(), x.end());
        float xMin = *std::min_element(x.begin(), x.end());
        float yMax = *std::max_element(y.begin(), y.end());
        float yMin = *std::min_element(y.begin(), y.end());
        float zMax = *std::max_element(z.begin(), z.end());
        float zMin = *std::min_element(z.begin(), z.end());

        Point min(xMin, yMin, zMin);
        Point max(xMax, yMax, zMax);

        /** segment tabletop interaction context */
        if (m.try_lock()) {
            sptr_threshold->first = min;
            sptr_threshold->second = max;
            m.unlock();
        }

        /** update tabletop interaction context x, y, z every sec */
        usleep(1000000);
    }
}

void intact::render(std::mutex& m, Kinect kinect, int numPoints,
                    std::shared_ptr<std::vector<float>>& sptr_points, std::shared_ptr<std::pair<Point, Point>>& sptr_threshold){

    viewer::draw(m, std::move(kinect), numPoints, sptr_points, sptr_threshold);
}