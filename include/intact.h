#ifndef INTACT_H
#define INTACT_H

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "kinect.h"
#include "point.h"

extern std::shared_ptr<bool> RUN_SYSTEM;

class Intact {

public:
    int m_numPoints;

    /** shared pointers with context rendering format -- kinect pcl format */
    std::shared_ptr<std::vector<float>> sptr_context = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_color = nullptr;

    /** shared pointers with 3DINTACT's point typedef */
    std::shared_ptr<int> sptr_numClusters = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_points = nullptr;

    /** multi-threading  mutual exclusion control */
    std::mutex m_mutex;
    std::shared_mutex s_mutex;

    /** multi-threading  flow-control semaphores */
    std::shared_ptr<bool> sptr_isContextClustered;
    std::shared_ptr<bool> sptr_isEpsilonComputed;
    std::shared_ptr<bool> sptr_isContextSegmented;

    explicit Intact(int& numPoints)
        : m_numPoints(numPoints)
    {
        sptr_numClusters = std::make_shared<int>(0);

        sptr_isContextClustered = std::make_shared<bool>(false);
        sptr_isEpsilonComputed = std::make_shared<bool>(false);
        sptr_isContextSegmented = std::make_shared<bool>(false);

        sptr_context = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_color = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);
        sptr_points = std::make_shared<std::vector<Point>>(m_numPoints * 3);
    }

    void castPointsToPcl(std::vector<Point>& points)
    {
        std::vector<float> pcl;
        std::vector<uint8_t> pclRgb;
        std::string delimiter = " ";
        for (auto& point : points) {
            pcl.push_back(point.m_xyz[0]);
            pcl.push_back(point.m_xyz[1]);
            pcl.push_back(point.m_xyz[2]);

            /** cast Point color to pcl-point color */
            std::string colorStr = point.m_clusterColor;
            colorStr.erase(0, 1);
            size_t last = 0;
            size_t next = 0;
            int rgb[3];
            int index = 0;
            while (
                (next = colorStr.find(delimiter, last)) != std::string::npos) {
                int colorVal = std::stoi(colorStr.substr(last, next - last));
                rgb[index] = colorVal;
                last = next + 1;
                index++;
            }
            int colorVal = std::stoi(colorStr.substr(last));
            rgb[index] = colorVal;

            pclRgb.push_back(rgb[0]);
            pclRgb.push_back(rgb[1]);
            pclRgb.push_back(rgb[2]);
        }
        std::lock_guard<std::mutex> lck(m_mutex);
        *sptr_context = pcl;
        *sptr_color = pclRgb;
    }

    void setContextPoints(const std::vector<Point>& points)
    {
        {
            std::lock_guard<std::mutex> lck(m_mutex);
            *sptr_points = points;
        }
    }

    void setNumClusters(const int& clusters)
    {
        {
            std::lock_guard<std::mutex> lck(m_mutex);
            *sptr_numClusters = clusters;
        }
    }

    void castPointsToPcl()
    {
        std::vector<Point> points;
        {
            std::lock_guard<std::mutex> lck(m_mutex);
            points = *sptr_points;
        }
        castPointsToPcl(points);
    }

    std::shared_ptr<std::vector<Point>> getPoints()
    {
        std::shared_lock lock(s_mutex);
        return sptr_points;
    }

    std::shared_ptr<std::vector<float>> getContext()
    {
        std::lock_guard<std::mutex> lck(m_mutex);
        return sptr_context;
    }

    std::shared_ptr<std::vector<uint8_t>> getColor()
    {
        std::lock_guard<std::mutex> lck(m_mutex);
        return sptr_color;
    }

    void setSegmentedFlag()
    {
        std::lock_guard<std::mutex> lck(m_mutex);
        *sptr_isContextSegmented = true;
    }

    void setClusteredFlag()
    {
        std::lock_guard<std::mutex> lck(m_mutex);
        *sptr_isContextClustered = true;
    }

    void setEpsilonComputedFlag()
    {
        std::lock_guard<std::mutex> lck(m_mutex);
        *sptr_isEpsilonComputed = true;
    }

    bool isSegmented()
    {
        std::shared_lock lock(s_mutex);
        return *sptr_isContextSegmented;
    }

    bool isClustered()
    {
        std::shared_lock lock(s_mutex);
        return *sptr_isContextClustered;
    }

    bool isEpsilonComputed()
    {
        std::shared_lock lock(s_mutex);
        return *sptr_isEpsilonComputed;
    }

    /**
     * segmentContext
     *   Initiates pipeline operation for segmenting context.
     *
     * @param sptr_kinect
     *   Kinect device.
     */
    static void segmentContext(std::shared_ptr<Kinect>& sptr_kinect,
        std::shared_ptr<Intact>& sptr_intact);

    /**
     * render
     *   Renders point cloud in real-time.
     *
     * @param sptr_kinect
     *   Kinect device.
     */
    static void renderContext(std::shared_ptr<Kinect>& sptr_kinect,
        std::shared_ptr<Intact>& sptr_intact);

    /**
     * queryContextBoundary
     *   Queries the min max point boundaries of segmented context.
     *
     * @param context
     *   Segmented interaction context.
     *
     * @retval
     *    { Point_min, Point_max }
     */
    static std::pair<Point, Point> queryContextBoundary(
        std::vector<Point>& context);

    /**
     * castPclToPoints
     *   Parses point cloud from std::vector<float> to std::vector<Point>
     *
     * @param sptr_kinect
     *   Kinect device.
     *
     * @retval
     *    Point cloud points.
     */
    static std::vector<Point> castPclToPoints(
        std::shared_ptr<Kinect>& sptr_kinect);

    static void cluster(
        const float& E, const int& N, std::shared_ptr<Intact>& sptr_intact);

    static void estimateEpsilon(
        const int& K, std::shared_ptr<Intact>& sptr_intact);
};
#endif /* INTACT_H */
