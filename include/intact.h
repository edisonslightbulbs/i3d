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
private:
public:
    std::mutex m_mutex;

    int m_numPoints;
    int m_depthWidth {};
    int m_depthHeight {};

    /** pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_rawPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_rawImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_rawPts = nullptr;

    /** segmented pcl, image, and points */
    std::shared_ptr<cv::Mat> sptr_segFrame = nullptr;
    std::shared_ptr<int16_t*> sptr_segPclBuf = nullptr;
    std::shared_ptr<uint8_t*> sptr_segImgBuf = nullptr;
    std::shared_ptr<std::vector<float>> sptr_segPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_segImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_segPts = nullptr;

    /** clustered pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_clustPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_clustImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_clustPts = nullptr;

    /** tabletop pcl, image, and points */
    std::shared_ptr<cv::Mat> sptr_ttopFrame = nullptr;
    std::shared_ptr<int16_t*> sptr_ttopPclBuf = nullptr;
    std::shared_ptr<uint8_t*> sptr_ttopImgBuf = nullptr;
    std::shared_ptr<std::vector<float>> sptr_ttopPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_ttopImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_ttopPts = nullptr;

    /** flow-control semaphores */
    std::shared_ptr<bool> sptr_run;
    std::shared_ptr<bool> sptr_stop;
    std::shared_ptr<bool> sptr_isClustered;
    std::shared_ptr<bool> sptr_isSegmented;
    std::shared_ptr<bool> sptr_isCalibrated;
    std::shared_ptr<bool> sptr_isKinectReady;
    std::shared_ptr<bool> sptr_isChromakeyed;
    std::shared_ptr<bool> sptr_isEpsilonComputed;

    std::pair<Point, Point> m_segBoundary {};
    std::pair<Point, Point> m_ttopBoundary {};

    explicit Intact(int& numPoints)
        : m_numPoints(numPoints)
    {
        /** initialize infinite boundaries */
        Point segUB(__FLT_MIN__, __FLT_MIN__, __FLT_MIN__);
        Point segLB(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);
        Point ttopUB(__FLT_MIN__, __FLT_MIN__, __FLT_MIN__);
        Point ttopLB(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);

        m_segBoundary = { segUB, segLB };
        m_ttopBoundary = { ttopUB, ttopLB };

        sptr_run = std::make_shared<bool>(false);
        sptr_stop = std::make_shared<bool>(false);
        sptr_isClustered = std::make_shared<bool>(false);
        sptr_isSegmented = std::make_shared<bool>(false);
        sptr_isCalibrated = std::make_shared<bool>(false);
        sptr_isKinectReady = std::make_shared<bool>(false);
        sptr_isChromakeyed = std::make_shared<bool>(false);
        sptr_isEpsilonComputed = std::make_shared<bool>(false);

        int size(m_numPoints * 3);
        sptr_rawPcl = std::make_shared<std::vector<float>>(size);
        sptr_rawImg = std::make_shared<std::vector<uint8_t>>(size);
        sptr_rawPts = std::make_shared<std::vector<Point>>(size);

        sptr_segPcl = std::make_shared<std::vector<float>>(size);
        sptr_segImg = std::make_shared<std::vector<uint8_t>>(size);
        sptr_segPts = std::make_shared<std::vector<Point>>(size);

        sptr_clustPcl = std::make_shared<std::vector<float>>(size);
        sptr_clustImg = std::make_shared<std::vector<uint8_t>>(size);
        sptr_clustPts = std::make_shared<std::vector<Point>>(size);

        sptr_ttopPcl = std::make_shared<std::vector<float>>(size);
        sptr_ttopImg = std::make_shared<std::vector<uint8_t>>(size);
        sptr_ttopPts = std::make_shared<std::vector<Point>>(size);
    }
    /**
     * segment
     *   Segments context.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void segment(std::shared_ptr<Intact>& sptr_intact);

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
     * approxEpsilon
     *   Estimates size of epsilon neighbourhood using knn.
     *
     * @param K
     *   K parameter.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void approxEpsilon(
        const int& K, std::shared_ptr<Intact>& sptr_intact);

    static void detectObjects(std::vector<std::string>& classnames,
        torch::jit::Module& module, std::shared_ptr<Intact>& sptr_intact);

    void chroma(std::shared_ptr<Intact>& sptr_intact);

    /** segment & cluster boundaries */
    std::pair<Point, Point> getSegBoundary();
    std::pair<Point, Point> getTtopBoundary();
    void setSegBoundary(std::pair<Point, Point>& boundary);
    void setTtpBoundary(std::pair<Point, Point>& boundary);

    /** raw pcl, image, and points */
    void setRawPcl(const std::vector<float>& pcl);
    void setRawImg(const std::vector<uint8_t>& img);
    void setRawPts(const std::vector<Point>& points);
    std::shared_ptr<std::vector<float>> getRawPcl();
    std::shared_ptr<std::vector<uint8_t>> getRawImg();

    /** segmented pcl, image, and points */
    void setSegFrame(cv::Mat& imgData);
    void setSegPts(const std::vector<Point>& points);
    void setSegPcl(const std::vector<float>& seg);
    void setSegImg(const std::vector<uint8_t>& segment);
    void setSegImgBuf(
        uint8_t* ptr_segImgBuf, uint8_t* ptr_imgBuf, const int& imgSize);
    void setSegPclBuf(
        int16_t* ptr_segPclBuf, int16_t* ptr_pclBuf, const int& pclSize);
    std::shared_ptr<int16_t*> getSegPclBuf();
    std::shared_ptr<uint8_t*> getSegImgBuf();
    std::shared_ptr<std::vector<Point>> getSegPts();
    std::shared_ptr<std::vector<float>> getSegPcl();
    std::shared_ptr<std::vector<uint8_t>> getSegImg();

    /** clustered pcl, image, and points */
    void setClustPcl(const std::vector<float>& points);
    void setClustImg(const std::vector<uint8_t>& img);
    void setClustPts(const std::vector<Point>& points);
    std::shared_ptr<std::vector<Point>> getClustPts();
    std::shared_ptr<std::vector<float>> getClustPcl();
    std::shared_ptr<std::vector<uint8_t>> getClustImg();

    /** tabletop pcl, image, and points */
    void setTtopImgBuf(
        uint8_t* ptr_ttpImgBuf, uint8_t* ptr_imgBuf, const int& imgSize);
    void setTtopFrame(cv::Mat& imgData);
    void setTtopPcl(const std::vector<float>& points);
    void setTtopImg(const std::vector<uint8_t>& img);
    void setTtopPts(const std::vector<Point>& points);
    std::shared_ptr<uint8_t*> getTtopImgBuf();
    std::shared_ptr<cv::Mat> getTtopFrame();
    std::shared_ptr<std::vector<float>> getTtopPcl();
    std::shared_ptr<std::vector<uint8_t>> getTtopImg();
    std::shared_ptr<std::vector<Point>> getTtopPoints();

    /** asynchronous flow-control semaphores */
    void stop();
    bool isRun();
    bool isStop();
    bool isSegmented();
    bool isClustered();
    // bool isCalibrated();
    bool isChromakeyed();
    bool isKinectReady();

    void raiseRunFlag();
    void raiseStopFlag();
    void raiseEpsilonFlag();
    void raiseSegmentedFlag();
    void raiseClusteredFlag();
    // void raiseCalibratedFlag();
    void raiseKinectReadyFlag();
    void raiseChromakeyedFlag();

    int getNumPoints();
    int getDepthImgWidth();
    int getDepthImgHeight();
    void setDepthImgHeight(const int& height);
    void setDepthImgWidth(const int& width);
};
#endif /* INTACT_H */
