#ifndef INTACT_H
#define INTACT_H

#include <memory>
#include <mutex>
#include <torch/script.h>
#include <vector>

#include "kinect.h"
#include "point.h"

class Intact {

public:
    int m_numPoints;
    int m_pclsize;
    int m_imgsize;
    int m_depthWidth {};
    int m_depthHeight {};

private:
    // hierarchical mutual exclusion
    std::mutex m_sensorMutex;
    std::mutex m_intactMutex;
    std::mutex m_bkgdMutex;
    std::mutex m_semaphoreMutex;

    // point clouds
    std::shared_ptr<std::vector<Point>> sptr_unrefinedPoints = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_refinedPoints = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_segmentedPoints = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_chromaBkgdPoints = nullptr;

    // shared sensor resources
    std::shared_ptr<uint8_t*> sptr_sensorImg_GL = nullptr;
    std::shared_ptr<uint8_t*> sptr_sensorImg_CV = nullptr;
    std::shared_ptr<int16_t*> sptr_sensorPcl = nullptr;

    // shared api resources
    std::pair<Point, Point> m_intactBoundary {};
    std::shared_ptr<uint8_t*> sptr_intactImg_GL = nullptr;
    std::shared_ptr<uint8_t*> sptr_intactImg_CV = nullptr;
    std::shared_ptr<int16_t*> sptr_intactPcl = nullptr;

    std::shared_ptr<int16_t*> sptr_chromaBkgdPcl = nullptr;
    std::shared_ptr<uint8_t*> sptr_chromaBkgdImg_GL = nullptr;
    std::shared_ptr<uint8_t*> sptr_chromaBkgdImg_CV = nullptr;

    // semaphores for asynchronous threads
    std::shared_ptr<bool> sptr_run;
    std::shared_ptr<bool> sptr_stop;
    std::shared_ptr<bool> sptr_isClustered;
    std::shared_ptr<bool> sptr_isSegmented;
    std::shared_ptr<bool> sptr_isKinectReady;
    std::shared_ptr<bool> sptr_isIntactReady;

public:
    /**
     * Intact
     *   Constructs instance of the 3dintact API
     *
     * @param numPts
     *   Max number of point-cloud points.
     */
    explicit Intact(int& numPts);

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
     * @param epsilon
     *   Epsilon parameter.
     * @param minPoints
     *   Number of epsilon-neighbourhood neighbours.
     * @param sptr_intact
     *   Instance of API call.
     */
    static void cluster(const float& epsilon, const int& minPoints,
        std::shared_ptr<Intact>& sptr_intact);

    /**
     * showObjects
     *   Detects objects in a given cv::Mat frame.
     *
     * @param classnames
     *   Object class names.
     * @param module
     *   torch module
     * @param sptr_intact
     *   Instance of API call.
     */
    static void showObjects(std::vector<std::string>& classnames,
        torch::jit::Module& module, std::shared_ptr<Intact>& sptr_intact);

    //////////////////////////////////////////////////////////////////////////
    //                semaphores for asynchronous threads
    //////////////////////////////////////////////////////////////////////////
    void stop();
    bool isRun();
    bool isStop();
    bool isSegmented();
    bool isClustered();
    bool isKinectReady();
    bool isIntactReady();

    void raiseRunFlag();
    void raiseStopFlag();
    void raiseSegmentedFlag();
    void raiseClusteredFlag();
    void raiseKinectReadyFlag();
    void raiseIntactReadyFlag();

    //////////////////////////////////////////////////////////////////////////
    //                     depth image width and height
    //////////////////////////////////////////////////////////////////////////
    int getDepthImgWidth();
    int getDepthImgHeight();
    void setDepthImgHeight(const int& height);
    void setDepthImgWidth(const int& width);

    ///////////////////////////////////////////////////////////////////////////////
    //                         shared point-cloud points
    ///////////////////////////////////////////////////////////////////////////////
    void setUnrefinedPoints(const std::vector<Point>& points);
    std::shared_ptr<std::vector<Point>> getUnrefinedPoints();

    void setRefinedPoints(const std::vector<Point>& points);
    std::shared_ptr<std::vector<Point>> getRefinedPoints();

    void setChromaBkgdPoints(const std::vector<Point>& points);
    std::shared_ptr<std::vector<Point>> getChromaBkgdPoints();

    //////////////////////////////////////////////////////////////////////////
    //                         shared sensor resources
    //////////////////////////////////////////////////////////////////////////
    void setSensorPcl(int16_t* ptr_pcl);
    std::shared_ptr<int16_t*> getSensorPcl();

    void setSensorImg_GL(uint8_t* ptr_img);
    std::shared_ptr<uint8_t*> getSensorImg_GL();

    void setSensorImg_CV(uint8_t* ptr_img);
    std::shared_ptr<uint8_t*> getSensorImg_CV();

    //////////////////////////////////////////////////////////////////////////
    //                         shared API resources
    //////////////////////////////////////////////////////////////////////////
    void setIntactPcl(int16_t* ptr_pcl);
    std::shared_ptr<int16_t*> getIntactPcl();

    void setIntactImg_GL(uint8_t* ptr_img);
    std::shared_ptr<uint8_t*> getIntactImg_GL();

    void setSegmentPoints(const std::vector<Point>& points);
    std::shared_ptr<std::vector<Point>> getSegmentPoints();

    void setIntactImg_CV(uint8_t* ptr_img);
    std::shared_ptr<uint8_t*> getIntactImg_CV();

    void setIntactBoundary(std::pair<Point, Point>& boundary);
    std::pair<Point, Point> getIntactBoundary();

    ///////////////////////////////////////////////////////////////////////////////
    //                           chroma background
    ///////////////////////////////////////////////////////////////////////////////
    void setChromaBkgdPcl(int16_t* ptr_pcl);
    std::shared_ptr<int16_t*> getChromaBkgdPcl();

    void setChromaBkgdImg_GL(uint8_t* ptr_img);
    std::shared_ptr<uint8_t*> getChromaBkgdImg_GL();

    void setChromaBkgdImg_CV(uint8_t* ptr_img);
    std::shared_ptr<uint8_t*> getChromaBkgdImg_CV();

    void sift(std::shared_ptr<Intact>& sptr_intact);
};
#endif /* INTACT_H */
