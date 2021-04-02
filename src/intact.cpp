#include <Eigen/Dense>
#include <atomic>
#include <utility>

#include "grow.h"
#include "intact.h"
#include "io.h"
#include "kinect.h"
#include "outliers.h"
#include "svd.h"
#include "viewer.h"

std::atomic<bool> RUN(true);

std::vector<Point> propose(std::vector<Point>& points)
{
    /** filter out outliers */
    std::vector<Point> denoisedPcl = outliers::filter(points);

    /** compute svd */
    std::pair<Eigen::JacobiSVD<Eigen::MatrixXd>, Eigen::MatrixXd> USV;
    USV = svd::compute(denoisedPcl);

    /** coarse segment */
    std::vector<Point> coarseSeg = grow::propose(USV, denoisedPcl);

    /** final segment */
    std::vector<Point> finalSeg = outliers::filter(coarseSeg);

    /** final interaction context segment */
    return finalSeg;
}

void intact::segment(Kinect& kinect)
{
    kinect.getPclImage();
    while (RUN) {

        std::vector<float> pcl;
        pcl = kinect.getPcl();

        /** parse point cloud data into Point type definitions */
        std::vector<Point> pclPoints;
        for (int i = 0; i < kinect.getNumPoints(); i++) { // resource race
            if (pcl[3 * i + 0] == 0 || pcl[3 * i + 1] == 0
                || pcl[3 * i + 2] == 0) {
                continue;
            }
            float x = pcl[3 * i + 0];
            float y = pcl[3 * i + 1];
            float z = pcl[3 * i + 2];
            Point point(x, y, z);
            pclPoints.push_back(point);
        }

        /** segment tabletop interaction context */
        std::vector<Point> context = propose(pclPoints);
        io::ply(context);

        /** query upper and lower constraints */
        std::vector<float> X;
        std::vector<float> Y;
        std::vector<float> Z;
        for (auto& point : context) {
            X.push_back(point.m_x);
            Y.push_back(point.m_y);
            Z.push_back(point.m_z);
        }
        float xMax = *std::max_element(X.begin(), X.end());
        float xMin = *std::min_element(X.begin(), X.end());
        float yMax = *std::max_element(Y.begin(), Y.end());
        float yMin = *std::min_element(Y.begin(), Y.end());
        float zMax = *std::max_element(Z.begin(), Z.end());
        float zMin = *std::min_element(Z.begin(), Z.end());

        Point min(xMin, yMin, zMin);
        Point max(xMax, yMax, zMax);

        /** update constraints of tabletop interaction context*/
        kinect.setContext({ min, max });

        /** update interaction context constraints every second */
        usleep(3000000);
    }
}

void intact::render(Kinect& kinect) { viewer::draw(kinect); }
