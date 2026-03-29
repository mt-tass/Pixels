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

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "invisibility_cloak.hpp"

using namespace std;

int main()
{
    cv::VideoCapture video(0);

    cout << "Invisibility Cloak\n";
    cout << "1. Place your cloth inside the rectangle\n";
    cout << "2. Press Space to calibrate the cloth colour\n";
    cout << "3. Step out of frame\n";
    cout << "4. Press 'B' to capture the background\n";
    cout << "5. Step back with the cloth and disappear!\n";
    cout << "Press 'R' to reset, 'Q' to quit.\n\n";

    cv::Mat frame;
    video.read(frame);

    // Calibration rectangle — centred in the first frame
    const int rectW = 250, rectH = 250;
    const int rectX = (frame.cols - rectW) / 2;
    const int rectY = (frame.rows - rectH) / 2;
    cv::Rect calibRect(rectX, rectY, rectW, rectH);

    cv::Scalar lowerBound, upperBound;
    bool calibrated = false;
    bool backgroundCaptured = false;
    cv::Mat background;

    while (video.isOpened()){

        video.read(frame);
        if (frame.empty())
            break;
        cv::flip(frame, frame, 1);

        cv::Mat display = frame.clone();

        if (backgroundCaptured && calibrated){
            cv::Mat mask = getClothMask(frame, lowerBound, upperBound);
            // Show binary mask of the cloak
            cv::namedWindow("Mask", cv::WINDOW_NORMAL);
            cv::resizeWindow("Mask", 640, 480);
            cv::imshow("Mask", mask);
            // Replace cloth pixels with the saved background
            background.copyTo(display, mask);
            putText(display, "Invisibility Active", 40, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 215, 255), 2);
        }
        else if (calibrated){
            putText(display, "Step OUT of frame, then press B", 40, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 215, 255), 2);
        }
        else{
            drawCalibrationRect(display, calibRect);
        }

        putText(display, "[SPACE] Calibrate  [B] Background  [R] Reset  [Q] Quit", display.rows-15, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 220, 230), 1);

        cv::namedWindow("Invisibility Cloak", cv::WINDOW_NORMAL);
        cv::resizeWindow("Invisibility Cloak", 640, 480);
        cv::imshow("Invisibility Cloak", display);

        int key = cv::waitKey(10);
        if (key == 'q' || key == 'Q'){
            break;
        }
        else if (key == 32 && !calibrated){                          //SPACE
            calibrateCloth(frame, calibRect, lowerBound, upperBound);
            calibrated = true;
            cout << "Cloth calibrated!\n";
        }
        else if ((key == 'b' || key == 'B') && calibrated){
            background = frame.clone();
            backgroundCaptured = true;
            cout << "Background captured!\n";
        }
        else if (key == 'r' || key == 'R'){
            calibrated = false;
            backgroundCaptured = false;
            cout << "Reset —> recalibrate cloth.\n";
        }
    }

    video.release();
    cv::destroyAllWindows();
    return 0;
}