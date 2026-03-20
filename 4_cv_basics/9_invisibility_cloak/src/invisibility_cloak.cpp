/*
MIT License

Copyright (c) 2026 Society of Robotics and Automation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "invisibility_cloak.hpp"
#include <algorithm>

using namespace std;

/*
    Function to draw text on a frame
 
    Purpose:
    --------
    Computes the x position such that the text is centred relative to frame width, then calls cv::putText
 
    Input Args:
    -----------
    cv::Mat &frame : image to draw on (modified in-place)
    const string &text : text to display
    int y : y coordinate of the text baseline
    int fontFace : OpenCV font (e.g. cv::FONT_HERSHEY_SIMPLEX)
    double fontScale : font scale factor
    cv::Scalar color : text colour in BGR
    int thickness : stroke thickness in pixels
*/
void putText(cv::Mat &frame, const string &text, int y, int fontFace, double fontScale, cv::Scalar color, int thickness){
    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, 0);
    int x = (frame.cols - textSize.width) / 2;
    cv::putText(frame, text, cv::Point(x, y), fontFace, fontScale, color, thickness);
}

/*
    Function to draw the calibration rectangle on the frame

    Purpose:
    --------
    Draws a rectangle and a label so the user knows where to place the cloth before calibration.

    Input Args:
    -----------
    cv::Mat &frame : image to draw on
    cv::Rect rect  : position and size of the calibration box
*/
void drawCalibrationRect(cv::Mat &frame, cv::Rect rect){
    cv::rectangle(frame, rect, cv::Scalar(0,215,255), 3);
    string text = "Place cloth here - Press SPACE";
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, 0);
    int xPos = frame.cols / 2 - textSize.width / 2;
    cv::putText(frame, text, cv::Point(xPos, rect.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,215,255), 2);
}

/*
    Function to calibrate the cloth colour from a selected ROI

    Purpose:
    --------
    Computes mean HSV values inside rect and derives lower/upper bounds with fixed margins so the mask is robust to minor lighting changes:
        Hue -> 15
        Sat -> 50
        Val -> 50

    Input Args:
    -----------
    const cv::Mat &frame : current BGR camera frame
    cv::Rect rect : calibration rectangle
    cv::Scalar &lowerBound : output — lower HSV 
    cv::Scalar &upperBound : output — upper HSV
*/
void calibrateCloth(const cv::Mat &frame, cv::Rect rect, cv::Scalar &lowerBound, cv::Scalar &upperBound){
    cv::Mat roi = frame(rect);
    cv::Mat hsvROI;
    cv::cvtColor(roi, hsvROI, cv::COLOR_BGR2HSV);

    cv::Scalar mean, stddev;
    cv::meanStdDev(hsvROI, mean, stddev);

    const int hueMargin = 15;
    const int satMargin = 50;
    const int valMargin = 50;

    lowerBound = cv::Scalar(max(0.0, mean[0] - hueMargin), max(0.0, mean[1] - satMargin), max(0.0, mean[2] - valMargin));
    upperBound = cv::Scalar(min(180.0, mean[0] + hueMargin), min(255.0, mean[1] + satMargin), min(255.0, mean[2] + valMargin));
}

/*
    Function to generate a binary mask of the cloth

    Purpose:
    --------
    Converts frame to HSV, thresholds it with the calibrated bounds, then cleans up noise with morphological opening and dilation.

    Input Args:
    -----------
    const cv::Mat &frame : current BGR camera frame
    cv::Scalar lowerBound : lower HSV threshold from calibrateCloth()
    cv::Scalar upperBound : upper HSV threshold from calibrateCloth()

    Returns:
    --------
    Binary mask— 255 where cloth is detected, 0 elsewhere.
*/
cv::Mat getClothMask(const cv::Mat &frame, cv::Scalar lowerBound, cv::Scalar upperBound){
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, lowerBound, upperBound, mask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,   kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel);

    return mask;
}