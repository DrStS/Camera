
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <map>
#include <set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

double computeMedianDouble(std::vector<double> &values)
{
    size_t size = values.size();
    if (size == 0)
    {
        return 0;
    }
    else
    {
        sort(values.begin(), values.end());
        if (size % 2 == 0)
        {
            return (values[size / 2 - 1] + values[size / 2]) / 2;
        }
        else
        {
            return values[size / 2];
        }
    }
}

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));
    //  cout << "Bouding boxes size: " << boundingBoxes.size() << endl;
    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    /*    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }


    // STUDENT TASK (replacement for meanDistRatio)
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    */
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1 / frameRate; // time between two measurements in seconds

    std::vector<double> lidarPointsPrevX;
    std::vector<double> lidarPointsCurrX;

    // find closest distance to Lidar points
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        minXPrev = minXPrev > it->x ? it->x : minXPrev;
        lidarPointsPrevX.push_back(it->x);
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        minXCurr = minXCurr > it->x ? it->x : minXCurr;
        lidarPointsCurrX.push_back(it->x);
    }
    double medXPrev = computeMedianDouble(lidarPointsPrevX);
    double medXCurr = computeMedianDouble(lidarPointsCurrX);
    // compute TTC from both measurements
    double TTCmin = minXCurr * dT / (minXPrev - minXCurr);
    cout << "= TTCmin: " << TTCmin << endl;
    TTC = medXCurr * dT / (medXPrev - medXCurr);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    double t = (double)cv::getTickCount();
    static double totalTime = 0;
    bool enableDebug = false;
    int currentBoxID;
    int previousBoxID;
    std::multimap<int, int> foundBBmatch; // <currentBoxID,previousBoxID>

    double max_dist = 0;
    double min_dist = 100;
    ///Optional
    //Compute  max and min distances between keypoints
    for (int i = 0; i < matches.size(); i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }
#ifdef DEBUG
    cout << "Max dist :: " << max_dist << endl;
    cout << "Min dist :: " << min_dist << endl;
#endif // DEBUG
    //Use only "good" matches (i.e. whose distance is less than 10*min_dist ) : deactiavted 1e6
    std::vector<cv::DMatch> good_matches;

    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < 10 * min_dist)
        {
            good_matches.push_back(matches[i]);
        }
    }
    ///End Optional: this reduces accumulated time for all images for this function from 80ms to 14ms
#ifdef DEBUG
    cout << "Total number of matches: " << matches.size() << endl;
    cout << "Total number of good matches: " << good_matches.size() << endl;
#endif // DEBUG
    for (size_t i = 0; i < good_matches.size(); ++i)
    {
        //look for a matching bounding box on current frame
        for (size_t k = 0; k < currFrame.boundingBoxes.size(); ++k)
        {

            if (currFrame.boundingBoxes[k].roi.contains(currFrame.keypoints[good_matches[i].trainIdx].pt)) //found a bounding box in current frame
            {
                currentBoxID = currFrame.boundingBoxes[k].boxID;
                //look for a matching bounding box on previous frame
                for (size_t l = 0; l < prevFrame.boundingBoxes.size(); ++l)
                {

                    if (prevFrame.boundingBoxes[l].roi.contains(prevFrame.keypoints[good_matches[i].queryIdx].pt)) //found a bounding box in previous frame -> bounding box match
                    {
                        previousBoxID = prevFrame.boundingBoxes[l].boxID;
                        foundBBmatch.insert({currentBoxID, previousBoxID});
                    }
                }
            }
        }
    }

#ifdef DEBUG
    cout << "Total number of matches found: " << foundBBmatch.size() << endl;

    {
        for (size_t k = 0; k < currFrame.boundingBoxes.size(); ++k)
        {
            if ((currFrame.boundingBoxes[k].lidarPoints.size() != 0))
                std::cout << " Lidar points for bounding box (current frame): " << currFrame.boundingBoxes[k].lidarPoints.size() << " for BoundingBox ID:  " << currFrame.boundingBoxes[k].boxID << std::endl;
        }

        for (size_t l = 0; l < prevFrame.boundingBoxes.size(); ++l)
        {
            if ((prevFrame.boundingBoxes[l].lidarPoints.size() != 0))
                std::cout << " Lidar points for bounding box (previous frame): " << prevFrame.boundingBoxes[l].lidarPoints.size() << " for BoundingBox ID:  " << prevFrame.boundingBoxes[l].boxID << std::endl;
        }
#endif // DEBUG

        for (size_t k = 0; k < currFrame.boundingBoxes.size(); ++k)
        {

#ifdef DEBUG
            std::cout << "=For bounding box (current frame) " << currFrame.boundingBoxes[k].boxID << " there are  " << foundBBmatch.count(currFrame.boundingBoxes[k].boxID) << " matched keypoints." << std::endl;
#endif // DEBUG \
    // Count matches in the previous frame
            std::map<int, unsigned int> helperMap;

            for (auto it = foundBBmatch.equal_range(currFrame.boundingBoxes[k].boxID).first; it != foundBBmatch.equal_range(currFrame.boundingBoxes[k].boxID).second; ++it)
            {
                helperMap[(*it).second]++;
            }
            if (helperMap.size() != 0)
            {
                std::vector<std::pair<int, unsigned int>> helperPairVec;
                for (auto it = helperMap.begin(); it != helperMap.end(); ++it)
                {
#ifdef DEBUG
                    std::cout << "For bounding box (previous frame) " << it->first << " has count " << it->second << std::endl;
#endif // DEBUG
                    helperPairVec.push_back(*it);
                }
                //sort map by value using lambda code
                sort(helperPairVec.begin(), helperPairVec.end(), [=](std::pair<int, unsigned int> &a, std::pair<int, unsigned int> &b) {
                    return a.second < b.second;
                });
#ifdef DEBUG
                std::cout << "First :" << helperPairVec.back().first << " second :" << helperPairVec.back().second << std::endl;
                std::cout << "Bounding box (current frame): " << currFrame.boundingBoxes[k].boxID << " is matched with bounding box (previous frame) " << helperPairVec.back().first << std::endl;
#endif // DEBUG
                if (foundBBmatch.count(currFrame.boundingBoxes[k].boxID) > 0)
                    bbBestMatches.insert(std::pair<int, int>(helperPairVec.back().first, currFrame.boundingBoxes[k].boxID));
            }
        }
#ifdef DEBUG

        for (auto it = bbBestMatches.cbegin(); it != bbBestMatches.cend(); ++it)
        {
            std::cout << it->first << "---" << it->second << endl;
        }
#endif // DEBUG
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        totalTime = totalTime + t;
        cout << "Matching bounding boxes took " << 1000 * t / 1.0 << " ms" << endl;
        cout << "Matching bounding boxes accu time " << 1000 * totalTime / 1.0 << " ms" << endl;
    }
