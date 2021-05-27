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
    int m_depthWidth {};
    int m_depthHeight {};

private:
    // hierarchical mutual exclusion
    std::mutex m_depthDimensions;

    std::mutex m_sensorTableDataMutex;
    std::mutex m_sensorImgDataMutex;
    std::mutex m_sensorDepthDataMutex;
    std::mutex m_sensorPCloudDataMutex;

    std::mutex m_pCloudMutex;
    std::mutex m_pCloudSegMutex;

    std::mutex m_pCloudFrameMutex;
    std::mutex m_pCloudSegFrameMutex;

    std::mutex m_pCloud2x2BinMutex;
    std::mutex m_pCloudSeg2x2BinMutex;

    std::mutex m_imgFrameMutex_GL;

    std::mutex m_imgSegFrameMutex_GL;
    std::mutex m_imgSegFrameMutex_CV;

    std::mutex m_boundaryMutex;

    // std::mutex m_bkgdMutex;
    std::mutex m_flagMutex;
    std::mutex m_clusterMutex;

    // point clouds
    std::shared_ptr<std::vector<Point>> sptr_pCloud = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_pCloudSeg = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_pCloud2x2Bin = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_pCloudSeg2x2Bin = nullptr;
    // std::shared_ptr<std::vector<Point>> sptr_chromaBkgdPoints = nullptr;

    // sensor resources
    k4a_float2_t* ptr_sensorTableData = nullptr;
    std::shared_ptr<int16_t*> sptr_sensorPCloudData = nullptr;
    std::shared_ptr<uint8_t*> sptr_sensorImgData = nullptr;
    std::shared_ptr<uint16_t*> sptr_sensorDepthData = nullptr;

    // core i3d handlers
    std::pair<Point, Point> m_i3dBoundary {};
    std::shared_ptr<std::vector<uint8_t>> sptr_imgFrame_CV = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_imgFrame_GL = nullptr;
    std::shared_ptr<std::vector<int16_t>> sptr_pCloudFrame = nullptr;
    std::shared_ptr<std::vector<int16_t>> sptr_i3dPClSegFrame = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_i3dImgSegFrame_GL = nullptr;

    // std::shared_ptr<int16_t*> sptr_chromaBkgdPcl = nullptr;
    // std::shared_ptr<uint8_t*> sptr_chromaBkgdImg_GL = nullptr;
    // std::shared_ptr<uint8_t*> sptr_chromaBkgdImg_CV = nullptr;

    // typedef alias for clusters
    typedef std::pair<std::vector<Point>,
        std::vector<std::vector<unsigned long>>>
        t_clusters;
    std::shared_ptr<t_clusters> sptr_clusters = nullptr;

    // semaphores for asynchronous threads
    std::shared_ptr<bool> sptr_run;
    std::shared_ptr<bool> sptr_stop;
    std::shared_ptr<bool> sptr_pCloudReady;
    std::shared_ptr<bool> sptr_clustered;
    std::shared_ptr<bool> sptr_boundarySet;
    std::shared_ptr<bool> sptr_segmented;
    std::shared_ptr<bool> sptr_resourcesReady;
    std::shared_ptr<bool> sptr_framesReady;
    std::shared_ptr<bool> sptr_isBackgroundReady;

public:
    /**
     * Intact
     *   Constructs instance of the 3dintact API
     */
    Intact();

    /**
     * segment
     *   Segments context.
     *
     * @param sptr_i3d
     *   Instance of API call.
     */
    static void region(std::shared_ptr<Intact>& sptr_i3d);

    /**
     * render
     *   Renders point cloud.
     *
     * @param sptr_i3d
     *   Instance of API call.
     */
    static void render(std::shared_ptr<Intact>& sptr_i3d);

    static void buildPCloud(std::shared_ptr<Intact>& sptr_i3d);

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
     * findObjects
     *   Detects objects in a given cv::Mat frame.
     *
     * @param classnames
     *   Object class names.
     * @param module
     *   torch module
     * @param sptr_i3d
     *   Instance of API call.
     */
    static void findObjects(std::vector<std::string>& classnames,
        torch::jit::Module& module, std::shared_ptr<Intact>& sptr_i3d);

    // ---------------------- asynchronous semaphores ---------------------//

    void stop();
    bool isRun();
    bool isStop();
    bool isPCloudReady();
    bool isSegmented();
    bool isBoundarySet();
    bool framesReady();
    bool isClustered();
    bool isSensorReady();
    bool isBackgroundReady();

    void raiseRunFlag();
    void raiseStopFlag();
    void raisePCloudReadyFlag();
    void raiseSegmentedFlag();
    void raiseBoundarySetFlag();
    void raiseClusteredFlag();
    void raiseSensorReadyFlag();
    void raiseFramesReadyFlag();
    void raiseBackgroundReadyFlag();

    // ---------------------- sensor resource handlers --------------------//

    int getDepthWidth();
    int getDepthHeight();
    void setDepthWidth(const int& width);
    void setDepthHeight(const int& height);

    void setSensorPCloudData(int16_t* ptr_pclData);
    std::shared_ptr<int16_t*> getSensorPCloudData();

    void setSensorTableData(k4a_float2_t* ptr_table);
    k4a_float2_t* getSensorTableData();

    void setSensorImgData(uint8_t* ptr_imgData);
    std::shared_ptr<uint8_t*> getSensorImgData();

    void setSensorDepthData(uint16_t* ptr_depth);
    std::shared_ptr<uint16_t*> getSensorDepthData();

    void setPCloud2x2Bin(const std::vector<Point>& points);
    std::shared_ptr<std::vector<Point>> getPCloud2x2Bin();

    // ----------------- preprocessed resource handlers -------------------//

    void setImgFrame_CV(const std::vector<uint8_t>& frame);
    std::shared_ptr<std::vector<uint8_t>> getImgFrame_CV();

    void setPCloudFrame(const std::vector<int16_t>& frame);
    std::shared_ptr<std::vector<int16_t>> getPCloudFrame();

    void setImgFrame_GL(const std::vector<uint8_t>& frame);
    std::shared_ptr<std::vector<uint8_t>> getImgFrame_GL();

    void setI3dPCloudSegFrame(const std::vector<int16_t>& frame);
    std::shared_ptr<std::vector<int16_t>> getPCloudSegFrame();

    void setI3dImgSegFrame_GL(const std::vector<uint8_t>& frame);
    std::shared_ptr<std::vector<uint8_t>> getImgSegFrame_GL();

    void setI3dBoundary(std::pair<Point, Point>& boundary);
    std::pair<Point, Point> getBoundary();

    void setPCloud(const std::vector<Point>& points);
    std::shared_ptr<std::vector<Point>> getPCloud();

    // ------------------------- operation handlers -----------------------//

    void setClusters(const t_clusters& clusters);
    std::shared_ptr<t_clusters> getClusters();

    // void setChromaBkgdPcl(int16_t* ptr_pcl);
    // std::shared_ptr<int16_t*> getChromaBkgdPcl();

    // void setChromaBkgdPoints(const std::vector<Point>& points);
    // std::shared_ptr<std::vector<Point>> getChromaBkgdPoints();

    // void setChromaBkgdImg_GL(uint8_t* ptr_img);
    // std::shared_ptr<uint8_t*> getChromaBkgdImg_GL();

    // void setChromaBkgdImg_CV(uint8_t* ptr_img);
    // std::shared_ptr<uint8_t*> getChromaBkgdImg_CV();

    static void frame(std::shared_ptr<Intact>& sptr_i3d);

    static void segment(std::shared_ptr<Intact>& sptr_i3d);

    static void chromakey(std::shared_ptr<Intact>& sptr_intact);

    void setPCloudSeg(const std::vector<Point>& points);

    std::shared_ptr<std::vector<Point>> getPCloudSeg();

    void setPCloudSeg2x2Bin(const std::vector<Point>& points);

    // std::shared_ptr<std::vector<Point>> getPCloudSeg2x2Bin();
};
#endif /* INTACT_H */
