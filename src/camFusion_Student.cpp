#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include <boost/accumulators/statistics/variance.hpp>
#include <math.h>
using namespace std;


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
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

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

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
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

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{   
    std::vector<double> distances;
    // Loop through all the matches
    for (auto match = kptMatches.begin(); match != kptMatches.end(); ++match)
    {
        // Check if point corresponding to the match in current Image is within ROI
        if(boundingBox.roi.contains(kptsCurr[match->trainIdx].pt))
        {
            // Find points in the previous and current image corresponding to the match
            cv::Point prevKpt = kptsPrev[match->queryIdx].pt;
            cv::Point currKpt = kptsCurr[match->trainIdx].pt;
            // Compute Euclidian distance between the matches
            double dist = cv::norm(currKpt - prevKpt);
            distances.push_back(dist);
            // Filter match based on Euclidian distance threshold of 10
            if (dist < 20.0) // The difference between keypoint locations in the two frames should not be vey high
                // Append keypoints and matches to bounding Box
                boundingBox.kptMatches.push_back((*match));
        }
    }
    // Compute mean of distance measures
    //double sum = std::accumulate(distances.begin(), distances.end(), 0.0);
    //double mean = sum / distances.size();
    //cout<< endl<<"mean distance :" << mean;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{

    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

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

    // compute camera-based TTC from distance ratios
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    // TODO: STUDENT TASK (replacement for meanDistRatio)
    double medianDistRatio = 0.0;
    std::sort(distRatios.begin(), distRatios.end());
    if (distRatios.size()%2)
        medianDistRatio = distRatios[distRatios.size()/2];
    else
        medianDistRatio = (distRatios[distRatios.size()/2] + distRatios[(distRatios.size()/2) - 1])/2;

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);

}

// Helper function for calculating key statistical values values
void computeStats(std::vector<float> scanPts, double& mean, double& stdev, double& median, double& min)
{
    double sum = std::accumulate(scanPts.begin(), scanPts.end(), 0.0); // Sum of all values
    mean = sum / scanPts.size(); // Mean
    double sq_sum = std::inner_product(scanPts.begin(), scanPts.end(), scanPts.begin(), 0.0); 
    stdev = sqrt(sq_sum / scanPts.size() - mean * mean); // Standard deviation
    std::sort(scanPts.begin(), scanPts.end()); // Sort 
    int idx = scanPts.size()/2;
    median = scanPts[idx]; // Median
    float stdevs= 2.0; // Standard deviation for outlier adjustment
    // Adjust min for outliers
    for(auto X : scanPts)
    {
        if((mean - X)/stdev <stdevs)
        {
            min = X;
            break;
        }
    }
}

// Compute time-to-collision (TTC) based on LiDAR readings in successive time steps 
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{   
    double minXPrev = 1e9, minXCurr = 1e9;
    // Extract X coordinates of Lidar scan
    std::vector<float> prevX;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        prevX.push_back(it->x);
    }
    std::vector<float> currX;
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {   
        currX.push_back(it->x);
    }

    // Calculate stistical key values (Mean, median and standard deviations) for previous frame
    double meanPrev, stdevPrev, medPrev;
    computeStats(prevX, meanPrev, stdevPrev, medPrev, minXPrev);
    
    // Calculate stistical key values (Mean, median and standard deviations) for current frame
    double meanCurr, stdevCurr, medCurr;
    computeStats(currX, meanCurr, stdevCurr, medCurr, minXCurr);

    // Manual selection of robust outlier adjustment method
    string adjustMethod = "median"; // Select between "median", "mean", "minimum"

    // compute TTC from both measurements
    if (adjustMethod.compare("minimum") == 0)
        TTC = minXCurr / ((minXPrev - minXCurr)*frameRate); // TTC based on min value
    else if (adjustMethod.compare("mean") == 0)
        TTC = meanCurr / ((meanPrev - meanCurr)*frameRate); // TTC based on mean 
    else if (adjustMethod.compare("median") == 0)
        TTC = medCurr / ((medPrev - medCurr)*frameRate); // TTC based on median
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{   
    // Multimap stores the bounding box pairs in the two frames corresponding to the matches
    std::multimap<int, int> Multimap;
    for(auto match = matches.begin(); match != matches.end(); ++match)
    {   
        // Associate point from previous image to Bounding boxes
        vector<vector<BoundingBox>::iterator> enclosingBoxes_prev;
        for(auto box = prevFrame.boundingBoxes.begin(); box != prevFrame.boundingBoxes.end(); ++box)
        {   
            if(box->roi.contains(prevFrame.keypoints[match->queryIdx].pt))
                enclosingBoxes_prev.push_back(box);
        }
        // Associate point from current image to Bounding boxes
        vector<vector<BoundingBox>::iterator> enclosingBoxes_curr;
        for(auto box = currFrame.boundingBoxes.begin(); box != currFrame.boundingBoxes.end(); ++box)
        {
            if(box->roi.contains(currFrame.keypoints[match->trainIdx].pt))
                enclosingBoxes_curr.push_back(box);
        }
        // add matching BoundingBoxIds of previous and current point to Multimap
        if(enclosingBoxes_prev.size() == 1 && enclosingBoxes_curr.size() ==1)
            Multimap.insert(make_pair(enclosingBoxes_prev[0]->boxID, enclosingBoxes_curr[0]->boxID)); 
    }  

    // Find the value with the maximum count for a given key and associate it with the key
    // Loop over all the keys of MultiMap
    for (auto it = Multimap.begin(); it != Multimap.end(); it = Multimap.upper_bound(it->first))
    {
        auto range = Multimap.equal_range(it->first);
        std::map<int,int> Map;
        // Count the elements in range and store it in the Map
        for (auto i = range.first; i != range.second; ++i)
        {
            // If element already present, increment count by 1
            if (Map.count(i->second))
                Map[i->second] = Map[i->second] +1;
            // If element not present, initialize count to 1
            else
                Map.insert(make_pair(i->second,1));
            // std::cout<< i->first << ": " << i->second << '\n'; // Debugging purpose
        }
        int greatest = 0;
        int greatest_key = 0;
        for(auto &el: Map)
        {   
            if (el.second > greatest)
            {
                greatest = el.second;
                greatest_key = el.first;
            }
        }
        // Match the Bounding box id of the first image frame to the Bounding Box id of the Secound frame wih the maximum count
        bbBestMatches.insert(make_pair(it->first,greatest_key));
        // cout<<endl<<"Curent BB pair : "<<it->first<<" : "<< greatest_key<<endl; // Print Corresponding bounding boxes
    }
}
