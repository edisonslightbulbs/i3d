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

    /** for context rendering */
    std::shared_ptr<std::vector<float>> sptr_context = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_color = nullptr;

    /** for context processing */
    std::shared_ptr<int> sptr_numClusters = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_points = nullptr;

    /** mutual exclusion */
    std::mutex m_mutex;
    std::shared_mutex s_mutex;

    /** flow-control semaphores */
    std::shared_ptr<bool> sptr_isContextClustered;
    std::shared_ptr<bool> sptr_isEpsilonComputed;
    std::shared_ptr<bool> sptr_isContextSegmented;

    /** initialize API */
    explicit Intact(int& numPoints)
        : m_numPoints(numPoints)
    {
        sptr_numClusters = std::make_shared<int>(0);

        sptr_isEpsilonComputed = std::make_shared<bool>(false);
        sptr_isContextClustered = std::make_shared<bool>(false);
        sptr_isContextSegmented = std::make_shared<bool>(false);

        sptr_points = std::make_shared<std::vector<Point>>(m_numPoints * 3);
        sptr_context = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_color = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);
    }

    /**
     * segment
     *   Segments context.
     *
     * @param sptr_kinect
     *   Kinect device.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void segment(std::shared_ptr<Kinect>& sptr_kinect,
        std::shared_ptr<Intact>& sptr_intact);
    /**
     * render
     *   Renders point cloud.
     *
     * @param sptr_kinect
     *   Kinect device.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void render(std::shared_ptr<Kinect>& sptr_kinect,
        std::shared_ptr<Intact>& sptr_intact);
    /**
     * cluster
     *   Clusters segmented context.
     *
     * @param E
     *   Epsilon parameter.
     *
     * @param N
     *   Number of epsilon-neighbourhood neighbours.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void cluster(
        const float& E, const int& N, std::shared_ptr<Intact>& sptr_intact);

    /**
     * estimateEpsilon
     *   Estimates size of epsilon neighbourhood using knn.
     *
     * @param K
     *   K parameter.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void estimateEpsilon(
        const int& K, std::shared_ptr<Intact>& sptr_intact);

    /**
     * adapt
     *  Adapts std::vector<Point> to point cloud.
     */
    void adapt();

    /** helpers:
     *   Setters. */
    void setNumClusters(const int& clusters);

    void setContextPoints(const std::vector<Point>& points);

    /** helpers:
     *   Getters. */
    std::shared_ptr<std::vector<Point>> getPoints();

    std::shared_ptr<std::vector<float>> getContext();

    std::shared_ptr<std::vector<uint8_t>> getColor();

    /** helpers:
     *  Thread-safe-semaphore controllers */
    bool isSegmented();

    bool isClustered();

    void raiseSegFlag();

    void raiseClustFlag();

    void raiseEpsilonFlag();

    bool isEpsilonComputed();
};
#endif /* INTACT_H */
