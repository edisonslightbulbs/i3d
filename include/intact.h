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
    int m_depthWidth;
    int m_depthHeight;

    /** initial data */
    std::shared_ptr<int16_t*> sptr_segmentedPclData = nullptr; // todo:check
    std::shared_ptr<uint8_t*> sptr_segmentedImgData = nullptr; // todo:check

    /** pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_pclVec = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_imgVec = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_points = nullptr;

    /** segmented pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_segmentedPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_segmentedImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_segmentedPoints = nullptr;
    std::shared_ptr<cv::Mat> sptr_segmentedImgFrame = nullptr; // todo fixme

    /** clustered pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_clusteredPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_clusteredImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_clusteredPoints = nullptr;

    /** tabletop pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_tabletopPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_tabletopImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_tabletopPoints = nullptr;
    std::shared_ptr<cv::Mat> sptr_tabletopImgData = nullptr; // todo fixme

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
    std::shared_ptr<bool> sptr_isChromakeyed; // todo:check

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
        sptr_isChromakeyed = std::make_shared<bool>(false); // todo:check
        sptr_isEpsilonComputed = std::make_shared<bool>(false);
        sptr_isContextClustered = std::make_shared<bool>(false);
        sptr_isContextSegmented = std::make_shared<bool>(false);

        sptr_pclVec = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_imgVec = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);
        sptr_points = std::make_shared<std::vector<Point>>(m_numPoints * 3);

        sptr_segmentedPcl
            = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_segmentedImg
            = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);
        sptr_segmentedPoints
            = std::make_shared<std::vector<Point>>(m_numPoints * 3);

        sptr_clusteredPcl
            = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_clusteredImg
            = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);
        sptr_clusteredPoints
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

    /** initial data */
    void setSegmentedPclData(int16_t* pcl);
    void setSegmentedImgData(uint8_t* image);

    std::shared_ptr<int16_t*> getPclData();
    std::shared_ptr<uint8_t*> getImgData();

    /** pcl, image, and points */
    void setPclVec(const std::vector<float>& pcl);
    void setImgVec(const std::vector<uint8_t>& img);
    void setPoints(const std::vector<Point>& points);

    std::shared_ptr<std::vector<float>> getPcl();
    std::shared_ptr<std::vector<uint8_t>> getImg();
    std::shared_ptr<std::vector<Point>> getPoints();

    /** segmented pcl, image, and points */
    void setSegmentedPcl(const std::vector<float>& segment);
    void setSegmentedImg(const std::vector<uint8_t>& segment);
    void setSegmentedPoints(const std::vector<Point>& points);
    void setSegmentedImgFrame(cv::Mat& imgData); // todo: check me

    std::shared_ptr<std::vector<float>> getSegmentedPcl();
    std::shared_ptr<std::vector<uint8_t>> getSegmentedImg();
    std::shared_ptr<std::vector<Point>> getSegmentedPoints();
    std::shared_ptr<cv::Mat> getSegmentedImgFrame(); // todo: check me

    /** clustered pcl, image, and points */
    void setClusteredPcl(const std::vector<float>& points);
    void setClusteredImg(const std::vector<uint8_t>& color);
    void setClusteredPoints(const std::vector<Point>& points);

    std::shared_ptr<std::vector<float>> getClusteredPcl();
    std::shared_ptr<std::vector<uint8_t>> getClusteredImg();
    std::shared_ptr<std::vector<Point>> getClusteredPoints();

    /** tabletop pcl, image, and points */
    void setTabletopPcl(const std::vector<float>& points);
    void setTabletopImg(const std::vector<uint8_t>& color);
    void setTabletopPoints(const std::vector<Point>& points);
    void setTabletopImgData(cv::Mat& imgData); // todo: check me

    std::shared_ptr<cv::Mat> getTabletopImgData(); // todo: check me
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

    void raiseChromakeyedFlag();

    bool isEpsilonComputed();

    void raiseSegmentedFlag();

    void raiseClusteredFlag();

    void raiseKinectReadyFlag();

    void raiseCalibratedFlag();

    static void detectObjects(std::vector<std::string>& classnames,
        torch::jit::Module& module, std::shared_ptr<Intact>& sptr_intact);

    int getDepthImgWidth();

    int getDepthImgHeight();

    void setDepthImgHeight(const int& height);

    void setDepthImgWidth(const int& width);
};
#endif /* INTACT_H */
