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
    std::mutex m_bufMutex;
    std::mutex m_updateMutex;
    std::mutex m_flagMutex;

    int m_numPoints;
    int m_depthWidth {};
    int m_depthHeight {};

    /** for raw pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_rawPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_rawImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_rawPts = nullptr;

    /** for segmented pcl, image, and points */
    std::shared_ptr<cv::Mat> sptr_segFrame = nullptr;
    std::shared_ptr<int16_t*> sptr_segPclBuf = nullptr;
    std::shared_ptr<uint8_t*> sptr_segImgBuf = nullptr;
    std::shared_ptr<std::vector<float>> sptr_segPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_segImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_segPts = nullptr;

    /** for clustered pcl, image, and points */
    std::shared_ptr<std::vector<float>> sptr_clustPcl = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_clustImg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_clustPts = nullptr;

    /** for tabletop pcl, image, and points */
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

    /** for segment and cluster boundaries */
    std::pair<Point, Point> m_segBoundary {};
    std::pair<Point, Point> m_ttopBoundary {};

public:
    /**
     * Intact
     *   Constructs instance of API
     *
     * @param numPoints
     *   Max number of point cloud points.
     */
    explicit Intact(int& numPoints);

    /**
     * segment
     *   Segments context.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void segment(std::shared_ptr<Intact>& sptr_intact);

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
     * @param N
     *   Number of epsilon-neighbourhood neighbours.
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

    /**
     * calibrate
     *   Calibrates projector camera system.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void calibrate(std::shared_ptr<Intact>& sptr_intact);

    /**
     * detectObjects
     *   Detects objects in a given cv::Mat frame.
     *
     * @param classnames
     *   Object class names.
     * @param module
     *   torch module
     * @param sptr_intact
     *   Instance of API call.
     */
    static void detectObjects(std::vector<std::string>& classnames,
        torch::jit::Module& module, std::shared_ptr<Intact>& sptr_intact);

    /**
     * chroma
     *   Chroma keys tabletop.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    void chroma(std::shared_ptr<Intact>& sptr_intact);

    /** for segment and cluster boundaries */
    std::pair<Point, Point> getSegBoundary();
    std::pair<Point, Point> getTtopBoundary();
    void setSegBoundary(std::pair<Point, Point>& boundary);
    void setTtpBoundary(std::pair<Point, Point>& boundary);

    /** for raw pcl, image, and points */
    void setRawPcl(const std::vector<float>& pcl);
    void setRawImg(const std::vector<uint8_t>& img);
    void setRawPts(const std::vector<Point>& points);
    std::shared_ptr<std::vector<float>> getRawPcl();
    std::shared_ptr<std::vector<uint8_t>> getRawImg();

    /** for segmented pcl, image, and points */
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

    /** for clustered pcl, image, and points */
    void setClustPcl(const std::vector<float>& points);
    void setClustImg(const std::vector<uint8_t>& img);
    void setClustPts(const std::vector<Point>& points);
    std::shared_ptr<std::vector<Point>> getClustPts();
    std::shared_ptr<std::vector<float>> getClustPcl();
    std::shared_ptr<std::vector<uint8_t>> getClustImg();

    /** for tabletop pcl, image, and points */
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

    /** for image and point cloud size */
    int getNumPoints();
    int getDepthImgWidth();
    int getDepthImgHeight();
    void setDepthImgHeight(const int& height);
    void setDepthImgWidth(const int& width);

    /** for asynchronous flow-control semaphores */
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
};
#endif /* INTACT_H */
