#ifndef INTACT_H
#define INTACT_H

#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <vector>

#include "kinect.h"
#include "point.h"

class Intact {

public:
    int m_numPoints;

    /** initial pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_pcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_img = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_points = nullptr;

    /** segmented pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_segmentPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_segmentImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_segmentPoints = nullptr;

    /** clustered pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_clustersPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_clustersImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_clustersPoints = nullptr;

    /** tabletop pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_tabletopPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_tabletopImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_tabletopPoints = nullptr;

    /** for pcl resource management */
    std::mutex m_mutex;

    /** flow-control semaphores */
    std::shared_ptr<bool> sptr_run;
    std::shared_ptr<bool> sptr_stop;
    std::shared_ptr<bool> sptr_isCalibrated;
    std::shared_ptr<bool> sptr_isKinectReady;
    std::shared_ptr<bool> sptr_isContextClustered;
    std::shared_ptr<bool> sptr_isEpsilonComputed;
    std::shared_ptr<bool> sptr_isContextSegmented;

    std::pair<Point, Point> m_segmentBoundary {};

    void setSegmentBoundary(std::pair<Point, Point>& boundary)
    {
        std::lock_guard<std::mutex> lck(m_mutex);
        m_segmentBoundary = boundary;
    }

    /** initialize API */
    explicit Intact(int& numPoints)
        : m_numPoints(numPoints)
    {
        /** initialize infinite boundary */
        Point lower(__FLT_MIN__, __FLT_MIN__, __FLT_MIN__);
        Point upper(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);

        m_segmentBoundary = { lower, upper };

        sptr_run = std::make_shared<bool>(false);
        sptr_stop = std::make_shared<bool>(false);
        sptr_isCalibrated = std::make_shared<bool>(false);
        sptr_isKinectReady = std::make_shared<bool>(false);
        sptr_isEpsilonComputed = std::make_shared<bool>(false);
        sptr_isContextClustered = std::make_shared<bool>(false);
        sptr_isContextSegmented = std::make_shared<bool>(false);

        sptr_pcl = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_img = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);
        sptr_points = std::make_shared<std::vector<Point>>(m_numPoints * 3);

        sptr_segmentPcl = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_segmentImg
            = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);
        sptr_segmentPoints
            = std::make_shared<std::vector<Point>>(m_numPoints * 3);

        sptr_clustersPcl
            = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_clustersImg
            = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);
        sptr_clustersPoints
            = std::make_shared<std::vector<Point>>(m_numPoints * 3);

        sptr_tabletopPcl
            = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_tabletopImg
            = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);
        sptr_tabletopPoints
            = std::make_shared<std::vector<Point>>(m_numPoints * 3);
    }
    /**
     * segment
     *   Segments context.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    void segment(std::shared_ptr<Intact>& sptr_intact);

    static void calibrate(std::shared_ptr<Intact>& sptr_intact);

    /**
     * render
     *   Renders point cloud.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void render(std::shared_ptr<Intact>& sptr_intact);
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

    int getNumPoints();

    /** initial pcl, image, and points */
    void setPcl(const std::vector<float>& pcl);
    void setImg(const std::vector<uint8_t>& color);
    void setPoints(const std::vector<Point>& points);

    std::shared_ptr<std::vector<float>> getPcl();
    std::shared_ptr<std::vector<uint8_t>> getImg();
    std::shared_ptr<std::vector<Point>> getPoints();

    /** segmented pcl, image, and points */
    void setSegmentPcl(const std::vector<float>& segment);
    void setSegmentImg(const std::vector<uint8_t>& segment);
    void setSegmentPoints(const std::vector<Point>& points);

    std::shared_ptr<std::vector<float>> getSegmentPcl();
    std::shared_ptr<std::vector<uint8_t>> getSegmentImg();
    std::shared_ptr<std::vector<Point>> getSegmentPoints();

    /** clustered pcl, image, and points */
    void setClustersPcl(const std::vector<float>& points);
    void setClustersImg(const std::vector<uint8_t>& color);
    void setClustersPoints(const std::vector<Point>& points);

    std::shared_ptr<std::vector<float>> getClustersPcl();
    std::shared_ptr<std::vector<uint8_t>> getClustersImg();
    std::shared_ptr<std::vector<Point>> getClustersPoints();

    /** tabletop pcl, image, and points */
    void setTabletopPcl(const std::vector<float>& points);
    void setTabletopImg(const std::vector<uint8_t>& color);
    void setTabletopPoints(const std::vector<Point>& points);

    std::shared_ptr<std::vector<float>> getTabletopPcl();
    std::shared_ptr<std::vector<uint8_t>> getTabletopImg();

    /** segment boundary */
    std::pair<Point, Point> getSegmentBoundary();

    /** asynchronous flow-control semaphores */
    void stop();

    bool isRun();

    bool isStop();

    bool isSegmented();

    bool isClustered();

    bool isCalibrated();

    void raiseRunFlag();

    void raiseStopFlag();

    bool isKinectReady();

    void raiseEpsilonFlag();

    bool isEpsilonComputed();

    void raiseSegmentedFlag();

    void raiseClusteredFlag();

    void raiseKinectReadyFlag();

    void raiseCalibratedFlag();

    static void detectObjects(std::vector<std::string>& classnames,
        torch::jit::Module& module, std::shared_ptr<Intact>& sptr_intact,
        std::shared_ptr<Kinect>& sptr_kinect);
};
#endif /* INTACT_H */
