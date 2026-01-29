#include "blocks.hpp"
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

// block 1 : video capture

getVideo::getVideo(int cam_idx){
    isOpen = false;
    capture.open(cam_idx , cv::CAP_V4L2);
    if(!(capture.isOpened())){
        cout << "Error cannot open the camera";
        return;
    }
    capture.set(cv::CAP_PROP_FRAME_HEIGHT,480);
    capture.set(cv::CAP_PROP_FRAME_WIDTH,720);
    isOpen = true;
}
getVideo::~getVideo(){
    capture.release();
    isOpen = false;
}
cv::Mat getVideo::getFrame(){
    cv::Mat frame;
    if(isOpen == false){
        return cv::Mat();
    }
    bool status = capture.read(frame);
    if (status == false || frame.empty()){
        return cv::Mat();
    }
    return frame;
}
bool getVideo::isOpened(){
    return isOpen;
}
void getVideo::release(){
    if(capture.isOpened()){
        capture.release();
    }
    isOpen = false;
}

// block 2: light detection

lightDetector::lightDetector(int min_thresh, int min_area, int max_area){
    minVal = min_thresh;
    minArea = min_area;
    maxArea = max_area;
    lowerBound = cv::Scalar(0,0,minVal);
    upperBound = cv::Scalar(180,255,255);
}
cv::Point lightDetector::get_bp(const cv::Mat &frame ){
    if(frame.empty()){
        return cv::Point(-1,-1);
    }
    cv::Mat hsv_frame;
    cv::cvtColor(frame , hsv_frame, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    cv::inRange(hsv_frame , lowerBound ,upperBound , mask);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE , cv::Size(5,5));
    cv::morphologyEx(mask,mask,cv::MORPH_OPEN , kernel);
    cv::morphologyEx(mask,mask,cv::MORPH_CLOSE , kernel);
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> heirarchy;
    cv::findContours(mask,contours,heirarchy,cv::RETR_EXTERNAL , cv::CHAIN_APPROX_SIMPLE);
    vector<cv::Point> largest;
    double largest_area = -1.0;
    for (const auto &contour : contours){
        double area = cv::contourArea(contour);
        if(area >= minArea && area <= maxArea){
            if(area > largest_area){
                largest_area = area;
                largest = contour;
            }
        }
    }
    //cout << "largest area = " << largest_area << emdl;
    if(!largest.empty()){
        cv::Moments m = cv::moments(largest);
        if(m.m00 != 0){
            int cx = static_cast<int>(m.m10/m.m00);
            int cy= static_cast<int>(m.m01/m.m00);
            return cv::Point(cx,cy);
        }
    } 
    return cv::Point(-1,-1);
}

//block 3 : image overlaying

cv::Mat imageOverlay::putImage(const cv::Mat &bg , const cv::Mat &fg , cv::Point pos , double scale ){
    cv::Mat result = bg.clone() , patronus = fg.clone();
    int fgh = fg.rows , fgw = fg.cols ;
    if(scale != 1.0){
        fgw = static_cast<int>(fgw * scale);
        fgh = static_cast<int>(fgh * scale);
        cv::resize(patronus , patronus , cv::Size(fgw,fgh) , 0 , 0, cv::INTER_AREA);
    }
    int cx = pos.x , cy = pos.y ;
    int x1 = cx-(fgw/2) , y1 = cy-(fgh/2);
    int x2= x1+fgw , y2 = y1+fgh;
    int bgh = bg.rows , bgw = bg.cols;
    int bx1=max(0,x1) , by1 = max(0,y1);
    int bx2 =min(x2,bgw) , by2=min(y2,bgh);
    int fx1=max(0,-x1),fy1=max(0,-y1);
    int fx2=fgw-max(0,x2-bgw) , fy2=fgh-max(0,y2-bgh);
    if(bx1 >= bx2 || by1>=by2){
        return result;
    }
    cv::Mat bg_roi = result(cv::Rect(bx1,by1,bx2-bx1,by2-by1));
    cv::Mat fg_roi = patronus(cv::Rect(fx1,fy1,fx2-fx1,fy2-fy1));
    if(patronus.channels() == 4){
        vector<cv::Mat> channels;
        cv::split(fg_roi,channels);
        cv::Mat rgb , alpha = channels[3];
        cv::merge(vector<cv::Mat>{channels[0],channels[1],channels[2]} , rgb);
        alpha.convertTo(alpha,CV_32F,1.0/255.0);
        cv::Mat alpha_F;
        cv::merge(vector<cv::Mat>{alpha,alpha,alpha},alpha_F);
        cv::Mat rgb_F, bg_roi_F;
        rgb.convertTo(rgb_F , CV_32F);
        bg_roi.convertTo(bg_roi_F , CV_32F);
        cv::Mat blend = alpha_F.mul(rgb_F) + (cv::Scalar(1.0,1.0,1.0)-alpha_F).mul(bg_roi_F);
        blend.convertTo(bg_roi,CV_8U);
    }
    else{
        fg_roi.copyTo(bg_roi);
    }
    return result;
}