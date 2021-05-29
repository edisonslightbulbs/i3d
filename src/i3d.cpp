#include <chrono>
#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>
#include <thread>
#include <utility>

#include "dbscan.h"
#include "helpers.h"
#include "i3d.h"
#include "kinect.h"
#include "macros.hpp"
#include "region.h"
#include "viewer.h"
#include "yolov5.h"

i3d::i3d()
{
    Point i3dMaxBound(SHRT_MAX, SHRT_MAX, SHRT_MAX);
    Point i3dMinBound(SHRT_MIN, SHRT_MIN, SHRT_MIN);

    m_boundary = { i3dMaxBound, i3dMinBound };

    sptr_run = std::make_shared<bool>(false);
    sptr_stop = std::make_shared<bool>(false);
    sptr_clustered = std::make_shared<bool>(false);
    sptr_segmented = std::make_shared<bool>(false);
    sptr_proposalReady = std::make_shared<bool>(false);
    sptr_pCloudReady = std::make_shared<bool>(false);
    sptr_resourcesReady = std::make_shared<bool>(false);
}

bool i3d::isRun()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_run;
}

void i3d::raiseRunFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_run = true;
}

bool i3d::isSensorReady()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_resourcesReady;
}

void i3d::raiseSensorReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_resourcesReady = true;
}

bool i3d::isSegmented()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_segmented;
}

bool i3d::isProposalReady()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_proposalReady;
}

void i3d::raiseProposalReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_proposalReady = true;
}

void i3d::raiseSegmentedFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_segmented = true;
}

bool i3d::isPCloudReady()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_pCloudReady;
}

void i3d::raisePCloudReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_pCloudReady = true;
}

bool i3d::isClustered()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_clustered;
}

void i3d::raiseClusteredFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_clustered = true;
}

bool i3d::isStop()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_stop;
}

void i3d::raiseStopFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_stop = true;
}

void i3d::stop()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_run = false;
}

void i3d::setDepthHeight(const int& height)
{
    std::lock_guard<std::mutex> lck(m_depthDimensions);
    m_depthHeight = height;
}

void i3d::setDepthWidth(const int& width)
{
    std::lock_guard<std::mutex> lck(m_depthDimensions);
    m_depthWidth = width;
}

int i3d::getDepthWidth()
{
    std::lock_guard<std::mutex> lck(m_depthDimensions);
    return m_depthWidth;
}

int i3d::getDepthHeight()
{
    std::lock_guard<std::mutex> lck(m_depthDimensions);
    return m_depthHeight;
}

void i3d::setPCloud2x2Bin(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloud2x2BinMutex);
    sptr_pCloud2x2Bin = std::make_shared<std::vector<Point>>(points);
}

std::shared_ptr<std::vector<Point>> i3d::getPCloud2x2Bin()
{
    std::lock_guard<std::mutex> lck(m_pCloud2x2BinMutex);
    return sptr_pCloud2x2Bin;
}

void i3d::setPCloudSeg2x2Bin(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloudSeg2x2BinMutex);
    sptr_pCloudSeg2x2Bin = std::make_shared<std::vector<Point>>(points);
}

__attribute__((unused)) std::shared_ptr<std::vector<Point>>
i3d::getPCloudSeg2x2Bin()
{
    std::lock_guard<std::mutex> lck(m_pCloudSeg2x2BinMutex);
    return sptr_pCloudSeg2x2Bin;
}

void i3d::setSensorImgData(uint8_t* ptr_imgData)
{
    std::lock_guard<std::mutex> lck(m_sensorImgDataMutex);
    sptr_sensorImgData = std::make_shared<uint8_t*>(ptr_imgData);
}

std::shared_ptr<uint8_t*> i3d::getSensorImgData()
{
    std::lock_guard<std::mutex> lck(m_sensorImgDataMutex);
    return sptr_sensorImgData;
}
void i3d::setSensorPCloudData(int16_t* ptr_pclData)
{
    std::lock_guard<std::mutex> lck(m_sensorPCloudDataMutex);
    sptr_sensorPCloudData = std::make_shared<int16_t*>(ptr_pclData);
}

std::shared_ptr<int16_t*> i3d::getSensorPCloudData()
{
    std::lock_guard<std::mutex> lck(m_sensorPCloudDataMutex);
    return sptr_sensorPCloudData;
}

void i3d::setSensorDepthData(uint16_t* ptr_depth)
{
    std::lock_guard<std::mutex> lck(m_sensorDepthDataMutex);
    sptr_sensorDepthData = std::make_shared<uint16_t*>(ptr_depth);
}

std::shared_ptr<uint16_t*> i3d::getSensorDepthData()
{
    std::lock_guard<std::mutex> lck(m_sensorDepthDataMutex);
    return sptr_sensorDepthData;
}

void i3d::setSensorTableData(k4a_float2_t* ptr_table)
{
    std::lock_guard<std::mutex> lck(m_sensorTableDataMutex);
    ptr_sensorTableData = ptr_table;
}

k4a_float2_t* i3d::getSensorTableData()
{
    std::lock_guard<std::mutex> lck(m_sensorTableDataMutex);
    return ptr_sensorTableData;
}

void i3d::setPCloud(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloudMutex);
    sptr_pCloud = std::make_shared<std::vector<Point>>(points);
}

__attribute__((unused)) std::shared_ptr<std::vector<Point>> i3d::getPCloud()
{
    std::lock_guard<std::mutex> lck(m_pCloudMutex);
    return sptr_pCloud;
}

void i3d::setPCloudSeg(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloudSegMutex);
    sptr_pCloudSeg = std::make_shared<std::vector<Point>>(points);
}

std::shared_ptr<std::vector<Point>> i3d::getPCloudSeg()
{
    std::lock_guard<std::mutex> lck(m_pCloudSegMutex);
    return sptr_pCloudSeg;
}

void i3d::setImgFrame_GL(const std::vector<uint8_t>& frame)
{
    std::lock_guard<std::mutex> lck(m_imgFrameMutex_GL);
    sptr_imgFrame_GL = std::make_shared<std::vector<uint8_t>>(frame);
}

std::shared_ptr<std::vector<uint8_t>> i3d::getImgFrame_GL()
{
    std::lock_guard<std::mutex> lck(m_imgFrameMutex_GL);
    return sptr_imgFrame_GL;
}

void i3d::setI3dBoundary(std::pair<Point, Point>& boundary)
{
    std::lock_guard<std::mutex> lck(m_boundaryMutex);
    m_boundary = boundary;
}

std::pair<Point, Point> i3d::getBoundary()
{
    std::lock_guard<std::mutex> lck(m_boundaryMutex);
    return m_boundary;
}

void i3d::setPCloudFrame(const std::vector<int16_t>& frame)
{
    std::lock_guard<std::mutex> lck(m_pCloudFrameMutex);
    sptr_pCloudFrame = std::make_shared<std::vector<int16_t>>(frame);
}

std::shared_ptr<std::vector<int16_t>> i3d::getPCloudFrame()
{
    std::lock_guard<std::mutex> lck(m_pCloudFrameMutex);
    return sptr_pCloudFrame;
}

void i3d::setPCloudSegFrame(const std::vector<int16_t>& frame)
{
    std::lock_guard<std::mutex> lck(m_pCloudSegFrameMutex);
    sptr_pCloudSegFrame = std::make_shared<std::vector<int16_t>>(frame);
}

std::shared_ptr<std::vector<int16_t>> i3d::getPCloudSegFrame()
{
    std::lock_guard<std::mutex> lck(m_pCloudSegFrameMutex);
    return sptr_pCloudSegFrame;
}

void i3d::setImgSegFrame_GL(const std::vector<uint8_t>& frame)
{
    std::lock_guard<std::mutex> lck(m_imgSegFrameMutex_GL);
    sptr_imgFrameSeg_GL = std::make_shared<std::vector<uint8_t>>(frame);
}

std::shared_ptr<std::vector<uint8_t>> i3d::getImgSegFrame_GL()
{
    std::lock_guard<std::mutex> lck(m_imgSegFrameMutex_GL);
    return sptr_imgFrameSeg_GL;
}

void i3d::setImgFrame_CV(const std::vector<uint8_t>& frame)
{
    std::lock_guard<std::mutex> lck(m_imgSegFrameMutex_CV);
    sptr_imgFrame_CV = std::make_shared<std::vector<uint8_t>>(frame);
}

std::shared_ptr<std::vector<uint8_t>> i3d::getImgFrame_CV()
{
    std::lock_guard<std::mutex> lck(m_imgSegFrameMutex_CV);
    return sptr_imgFrame_CV;
}

void i3d::setPCloudClusters(const t_clusters& clusters)
{
    std::lock_guard<std::mutex> lck(m_pCloudClusterMutex);
    sptr_pCloudClusters = std::make_shared<t_clusters>(clusters);
}

__attribute__((unused)) std::shared_ptr<i3d::t_clusters>
i3d::getPCloudClusters()
{
    std::lock_guard<std::mutex> lck(m_pCloudClusterMutex);
    return sptr_pCloudClusters;
}

void i3d::buildPCloud(std::shared_ptr<i3d>& sptr_i3d)
{
#if BUILD_POINTCLOUD == 1
    SLEEP_UNTIL_SENSOR_DATA_READY
    START
    int w = sptr_i3d->getDepthWidth();
    int h = sptr_i3d->getDepthHeight();

    uint16_t* ptr_depthData;
    k4a_float2_t* ptr_tableData;

    std::vector<Point> pCloud(w * h);
    while (sptr_i3d->isRun()) {

        START_TIMER
        ptr_depthData = *sptr_i3d->getSensorDepthData();
        ptr_tableData = sptr_i3d->getSensorTableData();

        int index = 0;
        for (int i = 0; i < w * h; i++) {
            if (utils::invalid(i, ptr_tableData, ptr_depthData)) {
                continue;
            }
            int16_t x = ptr_tableData[i].xy.x * (float)ptr_depthData[i];
            int16_t y = ptr_tableData[i].xy.y * (float)ptr_depthData[i];
            int16_t z = (float)ptr_depthData[i];
            Point point(x, y, z);
            pCloud[index] = point;
            index++;
        }
        std::vector<Point> optimizedPCloud(
            pCloud.begin(), pCloud.begin() + index);

        sptr_i3d->setPCloud2x2Bin(optimizedPCloud);
        RAISE_POINTCLOUD_READY_FLAG
        STOP_TIMER(" build point cloud thread: runtime @ ")
    }
#endif
}

void i3d::proposeRegion(std::shared_ptr<i3d>& sptr_i3d)
{
#if PROPOSE_REGION == 1
    SLEEP_UNTIL_POINTCLOUD_READY
    START
    while (sptr_i3d->isRun()) {
        START_TIMER
        std::vector<Point> pCloud = *sptr_i3d->getPCloud2x2Bin();
        std::vector<Point> pCloudSeg = region::segment(pCloud);
        std::pair<Point, Point> boundary = utils::queryBoundary(pCloudSeg);
        sptr_i3d->setPCloudSeg2x2Bin(pCloudSeg);
        sptr_i3d->setI3dBoundary(boundary);
        RAISE_PROPOSAL_READY_FLAG
        STOP_TIMER(" propose region thread: runtime @ ")
    }
#endif
}

void i3d::segmentRegion(std::shared_ptr<i3d>& sptr_i3d)
{
#if SEGMENT_REGION == 1
    SLEEP_UNTIL_PROPOSAL_READY_FLAG
    START
    int w = sptr_i3d->getDepthWidth();
    int h = sptr_i3d->getDepthHeight();

    std::vector<Point> pCloud(w * h);
    std::vector<Point> pCloudSeg(w * h);

    std::vector<int16_t> pCloudFrame(w * h * 3);
    std::vector<uint8_t> imgFrame_GL(w * h * 4);
    std::vector<uint8_t> imgFrame_CV(w * h * 4);

    std::vector<int16_t> pCloudSegFrame(w * h * 3);
    std::vector<uint8_t> imgSegFrame_GL(w * h * 4);
    std::vector<uint8_t> imgSegFrame_CV(w * h * 4);

    uint8_t* ptr_sensorImgData;
    int16_t* ptr_sensorPCloudData;

    while (sptr_i3d->isRun()) {
        START_TIMER
        ptr_sensorPCloudData = *sptr_i3d->getSensorPCloudData();
        ptr_sensorImgData = *sptr_i3d->getSensorImgData();

        int index = 0;
        for (int i = 0; i < w * h; i++) {
            Point point;

            // create unsegmented assets
            if (utils::invalid(i, ptr_sensorPCloudData, ptr_sensorImgData)) {
                utils::addXYZ(i, pCloudFrame);
                utils::addPixel_GL(i, imgFrame_GL);
                utils::addPixel_CV(i, imgFrame_CV);
            } else {
                utils::addXYZ(i, pCloudFrame, ptr_sensorPCloudData);
                utils::addPixel_GL(i, imgFrame_GL, ptr_sensorImgData);
                utils::addPixel_CV(i, imgFrame_CV, ptr_sensorImgData);
            }
            utils::adapt(i, point, pCloudFrame, imgFrame_CV);
            pCloud[i] = point;

            // create segmented assets
            if (utils::inSegment(i, pCloudFrame, sptr_i3d->getBoundary().first,
                    sptr_i3d->getBoundary().second)) {
                utils::addXYZ(i, pCloudSegFrame, ptr_sensorPCloudData);
                utils::addPixel_GL(i, imgSegFrame_GL, ptr_sensorImgData);
                utils::addPixel_CV(i, imgSegFrame_CV, ptr_sensorImgData);
                pCloudSeg[index] = point;
                index++;
            } else {
                utils::addXYZ(i, pCloudSegFrame);
                utils::addPixel_GL(i, imgSegFrame_GL);
            }
        }
        std::vector<Point> optimizedPCloudSeg(
            pCloudSeg.begin(), pCloudSeg.begin() + index);

        sptr_i3d->setPCloud(pCloud);
        sptr_i3d->setPCloudFrame(pCloudFrame);
        sptr_i3d->setImgFrame_GL(imgFrame_GL);
        sptr_i3d->setImgFrame_CV(imgFrame_CV);
        sptr_i3d->setPCloudSeg(optimizedPCloudSeg);
        sptr_i3d->setPCloudSegFrame(pCloudSegFrame);
        sptr_i3d->setImgSegFrame_GL(imgSegFrame_GL);
        RAISE_SEGMENT_READY_FLAG
        STOP_TIMER(" frame region thread: runtime @ ")
    }
#endif
}

void i3d::clusterRegion(
    const float& epsilon, const int& minPoints, std::shared_ptr<i3d>& sptr_i3d)
{
#if CLUSTER_REGION == 1
    SLEEP_UNTIL_SEGMENT_READY
    START
    while (sptr_i3d->isRun()) {
        START_TIMER
        std::vector<Point> points = *sptr_i3d->getPCloudSeg();
        auto clusters = dbscan::cluster(points, epsilon, minPoints);

        // sort clusters in descending order
        std::sort(clusters.begin(), clusters.end(),
            [](const std::vector<unsigned long>& a,
                const std::vector<unsigned long>& b) {
                return a.size() > b.size();
            });
        sptr_i3d->setPCloudClusters({ points, clusters });
        RAISE_CLUSTERS_READY_FLAG
        STOP_TIMER(" cluster region thread: runtime @ ")
    }
#endif
}

void i3d::renderRegion(std::shared_ptr<i3d>& sptr_i3d)
{
#if RENDER_REGION == 1
    SLEEP_UNTIL_SEGMENT_READY
    viewer::draw(sptr_i3d);
#endif
}

void i3d::findRegionObjects(std::vector<std::string>& classnames,
    torch::jit::script::Module& module, std::shared_ptr<i3d>& sptr_i3d)
{
#if DETECT_OBJECTS == 1
    // SLEEP_UNTIL_FRAMES_READY
    SLEEP_UNTIL_FRAMES_READY
    uint8_t* ptr_img;

    while (sptr_i3d->isRun()) {
        START_TIMER

        clock_t start = clock();
        int w = sptr_i3d->getDepthWidth();
        int h = sptr_i3d->getDepthHeight();

        // ptr_img = *sptr_i3d->getSensorImgData();
        ptr_img = sptr_i3d->getImgFrame_CV()->data();

        // create image frame
        cv::Mat img;
        cv::Mat frame
            = cv::Mat(h, w, CV_8UC4, (void*)ptr_img, cv::Mat::AUTO_STEP)
                  .clone();

        // format frame for tensor input
        cv::resize(frame, img, cv::Size(640, 384));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        torch::Tensor imgTensor = torch::from_blob(
            img.data, { img.rows, img.cols, 3 }, torch::kByte);
        imgTensor = imgTensor.permute({ 2, 0, 1 });
        imgTensor = imgTensor.toType(torch::kFloat);
        imgTensor = imgTensor.div(255);
        imgTensor = imgTensor.unsqueeze(0);

        torch::Tensor preds // preds: [?, 15120, 9]
            = module.forward({ imgTensor }).toTuple()->elements()[0].toTensor();
        std::vector<torch::Tensor> dets
            = yolo::non_max_suppression(preds, 0.4, 0.5);

        // show objects
        if (!dets.empty()) {
            for (size_t i = 0; i < dets[0].sizes()[0]; ++i) {
                float left
                    = dets[0][i][0].item().toFloat() * (float)frame.cols / 640;
                float top
                    = dets[0][i][1].item().toFloat() * (float)frame.rows / 384;
                float right
                    = dets[0][i][2].item().toFloat() * (float)frame.cols / 640;
                float bottom
                    = dets[0][i][3].item().toFloat() * (float)frame.rows / 384;
                float score = dets[0][i][4].item().toFloat();
                int classID = dets[0][i][5].item().toInt();

                cv::rectangle(frame,
                    cv::Rect(left, top, (right - left), (bottom - top)),
                    cv::Scalar(0, 255, 0), 2);
                cv::putText(frame,
                    classnames[classID] + ": " + cv::format("%.2f", score),
                    cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX,
                    (right - left) / 200, cv::Scalar(0, 255, 0), 2);
            }
        }
        utils::cvDisplay(frame, sptr_i3d, start);
        STOP_TIMER(" find region objects thread: runtime @ ")
    }
#endif
}
