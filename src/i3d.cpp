#include <chrono>
#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>
#include <thread>
#include <utility>

#include "dbscan.h"
#include "helpers.h"
#include "intact.h"
#include "kinect.h"
#include "macros.hpp"
#include "region.h"
#include "svd.h"
#include "viewer.h"
#include "yolov5.h"

Intact::Intact()
{
    Point i3dMaxBound(SHRT_MAX, SHRT_MAX, SHRT_MAX);
    Point i3dMinBound(SHRT_MIN, SHRT_MIN, SHRT_MIN);

    m_i3dBoundary = { i3dMaxBound, i3dMinBound };

    sptr_run = std::make_shared<bool>(false);
    sptr_stop = std::make_shared<bool>(false);
    sptr_clustered = std::make_shared<bool>(false);
    sptr_segmented = std::make_shared<bool>(false);
    sptr_boundarySet = std::make_shared<bool>(false);
    sptr_framesReady = std::make_shared<bool>(false);
    sptr_pCloudReady = std::make_shared<bool>(false);
    sptr_resourcesReady = std::make_shared<bool>(false);
    sptr_isBackgroundReady = std::make_shared<bool>(false);
}

// ---------------------- asynchronous semaphores ---------------------//

bool Intact::isRun()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_run;
}

void Intact::raiseRunFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_run = true;
}

bool Intact::isSensorReady()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_resourcesReady;
}

void Intact::raiseSensorReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_resourcesReady = true;
}

bool Intact::isSegmented()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_segmented;
}

bool Intact::isBoundarySet()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_boundarySet;
}

void Intact::raiseSegmentedFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_segmented = true;
}

void Intact::raiseBoundarySetFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_boundarySet = true;
}

bool Intact::isPCloudReady()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_pCloudReady;
}

void Intact::raisePCloudReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_pCloudReady = true;
}

bool Intact::framesReady()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_framesReady;
}

void Intact::raiseFramesReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_framesReady = true;
}

bool Intact::isClustered()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_clustered;
}

void Intact::raiseClusteredFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_clustered = true;
}

bool Intact::isBackgroundReady()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_isBackgroundReady;
}

void Intact::raiseBackgroundReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_isBackgroundReady = true;
}

bool Intact::isStop()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_stop;
}

void Intact::raiseStopFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_stop = true;
}

void Intact::stop()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_run = false;
}

// ---------------------- sensor resource handlers --------------------//

void Intact::setDepthHeight(const int& height)
{
    std::lock_guard<std::mutex> lck(m_depthDimensions);
    m_depthHeight = height;
}

void Intact::setDepthWidth(const int& width)
{
    std::lock_guard<std::mutex> lck(m_depthDimensions);
    m_depthWidth = width;
}

int Intact::getDepthWidth()
{
    std::lock_guard<std::mutex> lck(m_depthDimensions);
    return m_depthWidth;
}

int Intact::getDepthHeight()
{
    std::lock_guard<std::mutex> lck(m_depthDimensions);
    return m_depthHeight;
}

void Intact::setPCloud2x2Bin(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloud2x2BinMutex);
    sptr_pCloud2x2Bin = std::make_shared<std::vector<Point>>(points);
}

std::shared_ptr<std::vector<Point>> Intact::getPCloud2x2Bin()
{
    std::lock_guard<std::mutex> lck(m_pCloud2x2BinMutex);
    return sptr_pCloud2x2Bin;
}

void Intact::setPCloudSeg2x2Bin(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloudSeg2x2BinMutex);
    sptr_pCloudSeg2x2Bin = std::make_shared<std::vector<Point>>(points);
}

// std::shared_ptr<std::vector<Point>> Intact::getPCloudSeg2x2Bin()
// {
//     std::lock_guard<std::mutex> lck(m_pCloudSeg2x2BinMutex);
//     return sptr_pCloudSeg2x2Bin;
// }

void Intact::setSensorImgData(uint8_t* ptr_imgData)
{
    std::lock_guard<std::mutex> lck(m_sensorImgDataMutex);
    sptr_sensorImgData = std::make_shared<uint8_t*>(ptr_imgData);
}

std::shared_ptr<uint8_t*> Intact::getSensorImgData()
{
    std::lock_guard<std::mutex> lck(m_sensorImgDataMutex);
    return sptr_sensorImgData;
}
void Intact::setSensorPCloudData(int16_t* ptr_pclData)
{
    std::lock_guard<std::mutex> lck(m_sensorPCloudDataMutex);
    sptr_sensorPCloudData = std::make_shared<int16_t*>(ptr_pclData);
}

std::shared_ptr<int16_t*> Intact::getSensorPCloudData()
{
    std::lock_guard<std::mutex> lck(m_sensorPCloudDataMutex);
    return sptr_sensorPCloudData;
}

void Intact::setSensorDepthData(uint16_t* ptr_depth)
{
    std::lock_guard<std::mutex> lck(m_sensorDepthDataMutex);
    sptr_sensorDepthData = std::make_shared<uint16_t*>(ptr_depth);
}

std::shared_ptr<uint16_t*> Intact::getSensorDepthData()
{
    std::lock_guard<std::mutex> lck(m_sensorDepthDataMutex);
    return sptr_sensorDepthData;
}

void Intact::setSensorTableData(k4a_float2_t* ptr_table)
{
    std::lock_guard<std::mutex> lck(m_sensorTableDataMutex);
    ptr_sensorTableData = ptr_table;
}

k4a_float2_t* Intact::getSensorTableData()
{
    std::lock_guard<std::mutex> lck(m_sensorTableDataMutex);
    return ptr_sensorTableData;
}

// ----------------- preprocessed resource handlers -------------------//

void Intact::setPCloud(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloudMutex);
    sptr_pCloud = std::make_shared<std::vector<Point>>(points);
}

std::shared_ptr<std::vector<Point>> Intact::getPCloud()
{
    std::lock_guard<std::mutex> lck(m_pCloudMutex);
    return sptr_pCloud;
}

void Intact::setPCloudSeg(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloudSegMutex);
    sptr_pCloudSeg = std::make_shared<std::vector<Point>>(points);
}

std::shared_ptr<std::vector<Point>> Intact::getPCloudSeg()
{
    std::lock_guard<std::mutex> lck(m_pCloudSegMutex);
    return sptr_pCloudSeg;
}

void Intact::setImgFrame_GL(const std::vector<uint8_t>& frame)
{
    std::lock_guard<std::mutex> lck(m_imgFrameMutex_GL);
    sptr_imgFrame_GL = std::make_shared<std::vector<uint8_t>>(frame);
}

std::shared_ptr<std::vector<uint8_t>> Intact::getImgFrame_GL()
{
    std::lock_guard<std::mutex> lck(m_imgFrameMutex_GL);
    return sptr_imgFrame_GL;
}

void Intact::setI3dBoundary(std::pair<Point, Point>& boundary)
{
    std::lock_guard<std::mutex> lck(m_boundaryMutex);
    m_i3dBoundary = boundary;
}

std::pair<Point, Point> Intact::getBoundary()
{
    std::lock_guard<std::mutex> lck(m_boundaryMutex);
    return m_i3dBoundary;
}

void Intact::setPCloudFrame(const std::vector<int16_t>& frame)
{
    std::lock_guard<std::mutex> lck(m_pCloudFrameMutex);
    sptr_pCloudFrame = std::make_shared<std::vector<int16_t>>(frame);
}

std::shared_ptr<std::vector<int16_t>> Intact::getPCloudFrame()
{
    std::lock_guard<std::mutex> lck(m_pCloudFrameMutex);
    return sptr_pCloudFrame;
}

void Intact::setI3dPCloudSegFrame(const std::vector<int16_t>& frame)
{
    std::lock_guard<std::mutex> lck(m_pCloudSegFrameMutex);
    sptr_i3dPClSegFrame = std::make_shared<std::vector<int16_t>>(frame);
}

std::shared_ptr<std::vector<int16_t>> Intact::getPCloudSegFrame()
{
    std::lock_guard<std::mutex> lck(m_pCloudSegFrameMutex);
    return sptr_i3dPClSegFrame;
}

void Intact::setI3dImgSegFrame_GL(const std::vector<uint8_t>& frame)
{
    std::lock_guard<std::mutex> lck(m_imgSegFrameMutex_GL);
    sptr_i3dImgSegFrame_GL = std::make_shared<std::vector<uint8_t>>(frame);
}

std::shared_ptr<std::vector<uint8_t>> Intact::getImgSegFrame_GL()
{
    std::lock_guard<std::mutex> lck(m_imgSegFrameMutex_GL);
    return sptr_i3dImgSegFrame_GL;
}

void Intact::setImgFrame_CV(const std::vector<uint8_t>& frame)
{
    std::lock_guard<std::mutex> lck(m_imgSegFrameMutex_CV);
    sptr_imgFrame_CV = std::make_shared<std::vector<uint8_t>>(frame);
}

std::shared_ptr<std::vector<uint8_t>> Intact::getImgFrame_CV()
{
    std::lock_guard<std::mutex> lck(m_imgSegFrameMutex_CV);
    return sptr_imgFrame_CV;
}

// ------------------------- operation handlers -----------------------//

void Intact::setClusters(const t_clusters& clusters)
{
    std::lock_guard<std::mutex> lck(m_clusterMutex);
    sptr_clusters = std::make_shared<t_clusters>(clusters);
}

std::shared_ptr<Intact::t_clusters> Intact::getClusters()
{
    std::lock_guard<std::mutex> lck(m_clusterMutex);
    return sptr_clusters;
}

// void Intact::setChromaBkgdPoints(const std::vector<Point>& points)
// {
//     std::lock_guard<std::mutex> lck(m_bkgdMutex);
//     *sptr_chromaBkgdPoints = points;
// }
//
// std::shared_ptr<std::vector<Point>> Intact::getChromaBkgdPoints()
// {
//     std::lock_guard<std::mutex> lck(m_bkgdMutex);
//     return sptr_chromaBkgdPoints;
// }
//
// void Intact::setChromaBkgdPcl(int16_t* ptr_pcl)
// {
//     std::lock_guard<std::mutex> lck(m_bkgdMutex);
//     sptr_chromaBkgdPcl = std::make_shared<int16_t*>(ptr_pcl);
// }
//
// std::shared_ptr<int16_t*> Intact::getChromaBkgdPcl()
// {
//     std::lock_guard<std::mutex> lck(m_bkgdMutex);
//     return sptr_chromaBkgdPcl;
// }
//
// void Intact::setChromaBkgdImg_GL(uint8_t* ptr_img)
// {
//     std::lock_guard<std::mutex> lck(m_bkgdMutex);
//     sptr_chromaBkgdImg_GL = std::make_shared<uint8_t*>(ptr_img);
// }
//
// std::shared_ptr<uint8_t*> Intact::getChromaBkgdImg_GL()
// {
//     std::lock_guard<std::mutex> lck(m_bkgdMutex);
//     return sptr_chromaBkgdImg_GL;
// }
//
// void Intact::setChromaBkgdImg_CV(uint8_t* ptr_img)
// {
//     std::lock_guard<std::mutex> lck(m_bkgdMutex);
//     sptr_chromaBkgdImg_CV = std::make_shared<uint8_t*>(ptr_img);
// }
//
// std::shared_ptr<uint8_t*> Intact::getChromaBkgdImg_CV()
// {
//     std::lock_guard<std::mutex> lck(m_bkgdMutex);
//     return sptr_chromaBkgdImg_CV;
// }

void Intact::buildPCloud(std::shared_ptr<Intact>& sptr_i3d)
{
#if BUILD_POINTCLOUD == 1
    SLEEP_UNTIL_RESOURCES_READY
    START
    int w = sptr_i3d->getDepthWidth();
    int h = sptr_i3d->getDepthHeight();

    auto* ptr_depthData = (uint16_t*)malloc(sizeof(uint16_t) * w * h);
    auto* ptr_tableData = (k4a_float2_t*)malloc(sizeof(k4a_float2_t) * w * h);

    std::vector<Point> pCloud(w * h);
    while (sptr_i3d->isRun()) {
        START_TIMER
        // get depth and xy table data
        std::memcpy(ptr_depthData, *sptr_i3d->getSensorDepthData(),
            sizeof(int16_t) * w * h);
        std::memcpy(ptr_tableData, sptr_i3d->getSensorTableData(),
            sizeof(k4a_float2_t) * w * h);

        int index = 0;
        for (int i = 0; i < w * h; i++) {
            if (i3d::invalid(i, ptr_tableData, ptr_depthData)) {
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

        // PRINT(optimizedPCloud)
        // STOP
        sptr_i3d->setPCloud2x2Bin(optimizedPCloud);
        RAISE_POINTCLOUD_READY_FLAG
        STOP_TIMER(" point cloud builder thread: ")
    }
#endif
}

void Intact::frameRegion(std::shared_ptr<Intact>& sptr_i3d)
{
#if FRAMES == 1
    SLEEP_UNTIL_RESOURCES_READY
    START
    int w = sptr_i3d->getDepthWidth();
    int h = sptr_i3d->getDepthHeight();
    auto* ptr_sensorPCloudData = (int16_t*)malloc(sizeof(int16_t) * w * h * 3);
    auto* ptr_sensorImgData = (uint8_t*)malloc(sizeof(uint8_t) * w * h * 4);

    std::vector<Point> pCloud(w * h);
    std::vector<int16_t> pCloudFrame(w * h * 3);
    std::vector<uint8_t> imgFrame_GL(w * h * 4);
    std::vector<uint8_t> imgFrame_CV(w * h * 4);

    while (sptr_i3d->isRun()) {
        START_TIMER
        // get point cloud data and image
        std::memcpy(ptr_sensorPCloudData, *sptr_i3d->getSensorPCloudData(),
            sizeof(int16_t) * w * h * 3);
        std::memcpy(ptr_sensorImgData, *sptr_i3d->getSensorImgData(),
            sizeof(uint8_t) * w * h * 4);

        for (int i = 0; i < w * h; i++) {
            Point point;
            if (i3d::invalid(i, ptr_sensorPCloudData, ptr_sensorImgData)) {
                i3d::addXYZ(i, pCloudFrame);
                i3d::addPixel_GL(i, imgFrame_GL);
                i3d::addPixel_CV(i, imgFrame_CV);
            } else {
                i3d::addXYZ(i, pCloudFrame, ptr_sensorPCloudData);
                i3d::addPixel_GL(i, imgFrame_GL, ptr_sensorImgData);
                i3d::addPixel_CV(i, imgFrame_CV, ptr_sensorImgData);
            }
            i3d::adapt(i, point, pCloudFrame, imgFrame_CV);
            pCloud[i] = point;
        }
        sptr_i3d->setPCloud(pCloud);
        sptr_i3d->setPCloudFrame(pCloudFrame);
        sptr_i3d->setImgFrame_GL(imgFrame_GL);
        sptr_i3d->setImgFrame_CV(imgFrame_CV);
        RAISE_FRAMES_READY_FLAG
        STOP_TIMER(" frame builder thread: ")
    }
#endif
}

void Intact::proposeRegion(std::shared_ptr<Intact> &sptr_i3d) {
#if REGION == 1
    SLEEP_UNTIL_POINTCLOUD_READY
    START
    while (sptr_i3d->isRun()) {
        START_TIMER
        std::vector<Point> pCloud = *sptr_i3d->getPCloud2x2Bin();
        std::vector<Point> pCloudSeg = region::segment(pCloud);
        std::pair<Point, Point> boundary = i3d::queryBoundary(pCloudSeg);
        sptr_i3d->setPCloudSeg2x2Bin(pCloudSeg);
        sptr_i3d->setI3dBoundary(boundary);
        RAISE_BOUNDARY_SET_FLAG
        STOP_TIMER(" region finder thread: ")
    }
#endif
}

void Intact::segmentRegion(std::shared_ptr<Intact>& sptr_i3d)
{
#if SEGMENT == 1
    SLEEP_UNTIL_BOUNDARY_SET
    START
    int w = sptr_i3d->getDepthWidth();
    int h = sptr_i3d->getDepthHeight();

    std::vector<Point> pCloudSeg(w * h);

    while (sptr_i3d->isRun()) {
        START_TIMER
        std::vector<Point> pCloud = *sptr_i3d->getPCloud();
        std::vector<int16_t> pCloudFrame = *sptr_i3d->getPCloudFrame();
        std::vector<uint8_t> imgFrame_GL = *sptr_i3d->getImgFrame_GL();

        int index = 0;
        for (int i = 0; i < w * h; i++) {
            if (i3d::inSegment(i, pCloudFrame, sptr_i3d->getBoundary().first,
                    sptr_i3d->getBoundary().second)) {
                pCloudSeg[index] = pCloud[i];
                index++;
                continue;
            }
            i3d::addXYZ(i, pCloudFrame);
            i3d::addPixel_GL(i, imgFrame_GL);
        }
        std::vector<Point> optimizedPCloudSeg(
            pCloudSeg.begin(), pCloudSeg.begin() + index);

        sptr_i3d->setPCloudSeg(optimizedPCloudSeg);
        sptr_i3d->setI3dPCloudSegFrame(pCloudFrame);
        sptr_i3d->setI3dImgSegFrame_GL(imgFrame_GL);
        RAISE_SEGMENTATION_DONE_FLAG
        STOP_TIMER(" segmentation thread: ")
    }
#endif
}

void Intact::clusterRegion(const float& epsilon, const int& minPoints,
    std::shared_ptr<Intact>& sptr_intact)
{
#if CLUSTER == 1
    WHILE_INTACT_READY
    START
    while (sptr_intact->isRun()) {
        std::vector<Point> points = *sptr_intact->getSegmentPoints();

        // dbscan using kd-tree: returns clustered indexes
        auto clusters = dbscan::cluster(points, epsilon, minPoints);

        // sort clusters in descending order
        std::sort(clusters.begin(), clusters.end(),
            [](const std::vector<unsigned long>& a,
                const std::vector<unsigned long>& b) {
                return a.size() > b.size();
            });

        sptr_intact->setClusters({ points, clusters });
        CLUSTERS_READY
    }
#endif
}

void Intact::renderRegion(std::shared_ptr<Intact>& sptr_i3d)
{
#if RENDER == 1
    LOG(INFO) << "-- rendering";
    SLEEP_UNTIL_SEGMENT_READY
    viewer::draw(sptr_i3d);
#endif
}

void Intact::findRegionObjects(std::vector<std::string>& classnames,
    torch::jit::script::Module& module, std::shared_ptr<Intact>& sptr_i3d)
{
#if OR == 1
    SLEEP_UNTIL_FRAMES_READY
    while (sptr_i3d->isRun()) {

        // start frame rate clock
        clock_t start = clock();

        // get resources
        int w = sptr_i3d->getDepthWidth();
        int h = sptr_i3d->getDepthHeight();

        // todo: introduce key press control mirroring openGL implementation
        uint8_t* ptr_img = *sptr_i3d->getSensorImgData_CV();
        // uint8_t* ptr_img = *sptr_i3d->getI3dRawImgData_CV();
        // uint8_t* ptr_img = *sptr_intact->getChromaBkgdImg_CV();

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

        cv::putText(frame,
            "FPS: " + std::to_string(int(1e7 / (double)(clock() - start))),
            cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
            cv::Scalar(0, 255, 0), 2);

        cv::imshow("", frame);
        if (cv::waitKey(1) == 27) {
            sptr_i3d->raiseStopFlag();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
#endif
}

///////////////////////////////////////////////////////////////////////////////
//                       example/possible extensions
///////////////////////////////////////////////////////////////////////////////

void Intact::chromakey(std::shared_ptr<Intact>& sptr_intact)
{
#if CHROMAKEY == 1

    WHILE_CLUSTERS_READY
    START
    while (sptr_intact->isRun()) {

        // get un-edited image
        std::vector<Point> frame = *sptr_intact->getRefinedPoints();

        // get clusters
        auto clusters = sptr_intact->getClusters();
        auto indexClusters = clusters->second;
        std::vector<Point> points = clusters->first;

        // define chromakey color for tabletop surface
        uint8_t rgb[3] = { chromagreen[0], chromagreen[1], chromagreen[2] };
        uint8_t bgra[4] = { chromagreen[2], chromagreen[1], chromagreen[0], 0 };

        ///////////////////////////////////////////////////
        /////////////// beta testing //////////////////////
        ///////////////////////////////////////////////////
        // cast clusters of indexed to clusters of points
        std::vector<std::vector<Point>> pointClusters;
        for (const auto& cluster : indexClusters) {
            std::vector<Point> heap;
            for (const auto& index : cluster) {
                heap.emplace_back(points[index]);
            }
            pointClusters.emplace_back(heap);
        }

        // config flags for svd computation
        int flag = Eigen::ComputeThinU | Eigen::ComputeThinV;

        // heap norms for each cluster
        std::vector<Eigen::Vector3d> normals;
        for (const auto& cluster : pointClusters) {
            SVD usv(cluster, flag);
            normals.emplace_back(usv.getV3Normal());
        }

        const float ARGMIN = 0.4;
        // evaluate coplanarity between clusters
        int clusterIndex = 0;
        std::vector<Point> tabletop = pointClusters[0];
        for (const auto& normal : normals) {
            double a = normals[0].dot(normal);
            double b = normals[0].norm() * normal.norm();
            double solution = std::acos(a / b);

            if (!std::isnan(solution) && solution < ARGMIN && solution > -ARGMIN
                && pointClusters[clusterIndex].size() < 25) {
                tabletop.insert(tabletop.end(),
                    pointClusters[clusterIndex].begin(),
                    pointClusters[clusterIndex].end());
            }
            clusterIndex++;
        }

        // find the upper and lower z limits
        std::vector<float> zMeasures;
        for (const auto& point : pointClusters[0]) {
            zMeasures.emplace_back(point.m_xyz[2]);
        }
        int16_t maxH = *std::max_element(zMeasures.begin(), zMeasures.end());
        int16_t minH = *std::min_element(zMeasures.begin(), zMeasures.end());

        // use limits to refine tabletop points
        std::vector<Point> bkgd; //
        for (const auto& point : tabletop) {
            if (point.m_xyz[2] > maxH || point.m_xyz[2] < minH) {
                continue;
            }
            bkgd.emplace_back(point);
        }
        ///////////////////////////////////////////////////
        /////////////// beta testing //////////////////////
        ///////////////////////////////////////////////////

        // use clustered indexes to chromakey a selected cluster
        for (const auto& point : bkgd) {
            int id = point.m_id;
            frame[id].setPixel_GL(rgb);
            frame[id].setPixel_CV(bgra);
        }

        // sizes
        const int numPoints = sptr_intact->m_numPoints;
        const uint32_t pclsize = numPoints * 3;
        const uint32_t imgsize = numPoints * 4;

        int16_t pclBuf[pclsize];
        uint8_t imgBuf_GL[pclsize];
        uint8_t imgBuf_CV[imgsize];

        // stitch images from processed point cloud
        for (int i = 0; i < numPoints; i++) {
            i3d::stitch(i, frame[i], pclBuf, imgBuf_GL, imgBuf_CV);
        }
        sptr_intact->setChromaBkgdPoints(frame);
        sptr_intact->setChromaBkgdPcl(pclBuf);
        sptr_intact->setChromaBkgdImg_GL(imgBuf_GL);
        sptr_intact->setChromaBkgdImg_CV(imgBuf_CV);
        CHROMABACKGROUND_READY
    }
#endif
}
