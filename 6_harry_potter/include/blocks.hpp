#ifndef BLOCKS_HPP
#define BLOCKS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <ctime>
#include <cstdlib>

// VIDEO CAPTURE BLOCK

class getVideo{
private:
    cv::VideoCapture capture;
    bool isOpen;

public:
    getVideo(int cameraIdx = 0);
    ~getVideo();
    
    cv::Mat getFrame();
    void release();
    bool isOpened();
};

// LIGHT DETECTION BLOCK

class lightDetector {
private:
    int minVal;
    int minArea;
    int maxArea;
    cv::Scalar lowerBound;
    cv::Scalar upperBound;

public:
    lightDetector(int min_thresh = 225, int min_area = 10, int max_area = 10000);
    cv::Point getBrightestPoint(const cv::Mat& frame);
};

// IMAGE OVERLAY BLOCK 

class imageOverlay {
public:
    imageOverlay() = default;
    cv::Mat putImage(const cv::Mat& background, const cv::Mat& foreground,cv::Point position, double scale = 1.0, bool removeBlack = true);
};

// CALIBRATION RECTANGLE 

class calibrationRectangle {
private:
    cv::Rect rect;
    cv::Scalar lowerBound;
    cv::Scalar upperBound;
    bool calibrated;
public:
    calibrationRectangle(int x, int y, int width, int height);
    
    void draw(cv::Mat& frame);
    bool calibrate(const cv::Mat& frame);
    cv::Mat getMask(const cv::Mat& frame);
    bool isCalibrated(){
        return calibrated;
    }
    void reset(){
        calibrated = false;
    }
};

// MENU SYSTEM 

class menuSystem {
private:
    std::vector<std::string> items;
    int selectedIndex;
    cv::Scalar normalColor;
    cv::Scalar selectedColor;
    cv::Scalar bgColor;

public:
    menuSystem();
    
    void addItem(const std::string& item);
    void moveUp();
    void moveDown();
    int getSelected() const { return selectedIndex; }
    void draw(cv::Mat& frame, int startY);
    void reset() { selectedIndex = 0; }
};

// MAIN UI BLOCK

class MagicalUI {
public:
    static const cv::Scalar GOLD;
    static const cv::Scalar DARK_BG;
    static const cv::Scalar SELECTED;
    static const cv::Scalar CREAM;
    
    MagicalUI() = default;
    
    void drawDarkBackground(cv::Mat& frame);
    void drawTitle(cv::Mat& frame, const std::string& title , const std::string &subtitle ,const std::string &footer);
    void drawStatus(cv::Mat& frame, const std::string& status, cv::Scalar color);
    void drawScore(cv::Mat& frame, int score, const std::string& label = "Score");
};

// QUIDDITCH GOLDEN SNITCH

class GoldenSnitch {
public:
    cv::Point position;
    cv::Point velocity;
    int radius;
    bool active;
    
    GoldenSnitch();
    GoldenSnitch(int frameWidth, int y = 0);
    
    void update(int frameHeight);
    void draw(cv::Mat& frame);
    bool checkCollision(cv::Point ringPosition, int ringRadius);
    void reset(int frameWidth);
};

// QUIDDITCH GAME LOGIC

class QuidditchGame {
private:
    std::vector<GoldenSnitch> snitches;
    int score;
    int missed;
    int ringX;
    int ringRadius;
    int frameWidth;
    int frameHeight;
    int spawnTimer;
    int spawnInterval;
    
public:
    QuidditchGame(int width, int height);
    
    void updateRingPosition(cv::Point lightPosition);
    void update();
    void draw(cv::Mat& frame);
    int getScore() const { return score; }
    int getMissed() const { return missed; }
    void reset();
};

#endif 