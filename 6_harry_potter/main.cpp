#include <iostream>
#include <opencv2/opencv.hpp>
#include "blocks.hpp"

using namespace std;

void runPatronusMode();
void runInvisibilityMode();
void runQuidditchMode();

int main() {
    cout << "\nInitializing...\n";
    getVideo cam(0);
    if (!cam.isOpened()) {
        cout  << "could not access camera" << endl;
        return -1;
    }
    
    cout << "Camera initialized" << endl;
    
    menuSystem menu;
    menu.addItem("EXPECTO PATRONUM");
    menu.addItem("INVISIBILITY CLOAK");
    menu.addItem("QUIDDITCH TRAINING");
    menu.addItem("EXIT");
    
    MagicalUI ui;
    bool running = true;
    
    while (running) {
        cv::Mat frame = cam.getFrame();
        if (frame.empty()){
            break;
        }
        
        cv::Mat display = frame.clone();
        ui.drawDarkBackground(display);
        ui.drawTitle(display, "HOGWARTS LEGACY" , "< OpenCV Magic >" , "UP/DOWN: Navigate    ENTER: Select");
        menu.draw(display,display.rows/3);
        imshow("harry potter world", display);
        int key = cv::waitKey(17);
        if (key == 27) {
            running = false;
        }
        else if (key == 82 || key == 0){
            menu.moveUp();
        }
        else if (key == 84 || key == 1){
            menu.moveDown();
        }
        else if (key == 13 || key == 32){
            int selected = menu.getSelected();
            if (selected == 0){
                cam.release();
                runPatronusMode();
                cam = getVideo(0);
            }
            else if (selected== 1){
                cam.release();
                runInvisibilityMode();
                cam = getVideo(0);
            }
            else if (selected == 2){
                cam.release();
                runQuidditchMode();
                cam = getVideo(0);
            }
            else if (selected == 3){
                running = false;
            }
        }
    }
    cam.release();
    cv::destroyAllWindows();
    cout << "\nMischief Managed!" << endl;
    return 0;
}

void runPatronusMode() {
    cout << "EXPECTO PATRONUm" << endl;
    cout << "point a bright light at the camera" << endl;
    cout << "press esc to return" << endl;
    
    const int hsv_thresh = 250;
    const int min_area = 8000;
    const int max_area = 12000;
    const double scale = 0.65;
    getVideo cam(0);
    if (!cam.isOpened()){
        return;
    }
    lightDetector detector(hsv_thresh, min_area, max_area);
    imageOverlay overlay;
    MagicalUI ui;
    cv::Mat patronus = cv::imread("data/unicorn.png");
    cout << "Patronus ready!" << endl;
    while (true){
        cv::Mat frame = cam.getFrame();
        if (frame.empty()){
            break;
        }
        cv::Mat display = frame.clone();
        cv::Point pos = detector.getBrightestPoint(frame);
        if (pos.x != -1 && pos.y != -1){
            display = overlay.putImage(display, patronus, pos, scale, true);
            ui.drawStatus(display, "Patronus Active ", MagicalUI::GOLD);
        } 
        else{
            ui.drawStatus(display, "Point your wand ", MagicalUI::CREAM);
        }
        cv::putText(display, "[ESC] Return",cv::Point(20, display.rows - 20),cv::FONT_HERSHEY_SIMPLEX, 0.5, MagicalUI::CREAM, 1);
        cv::imshow("Expecto Patronum", display);
        if (cv::waitKey(1) == 27){
            break;
        }
    }
    cam.release();
    cv::destroyWindow("Expecto Patronum");
    cout << "Patronous dismissed\n\n";
}
void runInvisibilityMode() {
    cout << "INVISIBILITY CLOAK" << endl;
    cout << "1. place your cloth in the rectangle\n";
    cout << "2. Press space key to calibrate\n";
    cout << "3. Step out of frame to get desired background\n";
    cout << "4. Press b to capture background\n";
    cout << "5. Step back with cloth and disappear !\n";
    cout << "Press esc to return.\n\n";
    getVideo cam(0);
    if (!cam.isOpened()){
        return;
    }
    cv::Mat frame = cam.getFrame();
    int rectW = 250;
    int rectH = 250;
    int rectX = (frame.cols - rectW) / 2;
    int rectY = (frame.rows - rectH) / 2;
    
    calibrationRectangle calibRect(rectX, rectY, rectW, rectH);
    MagicalUI ui;
    
    cv::Mat background;
    bool backgroundCaptured = false;
    
    while (true) {
        frame = cam.getFrame();
        if (frame.empty()){
            break;
        }
        
        cv::Mat display = frame.clone();
        
        if (backgroundCaptured && calibRect.isCalibrated()){
            cv::Mat mask = calibRect.getMask(frame);
            if (!mask.empty()) {
                background.copyTo(display, mask);
            }
            ui.drawStatus(display, "Invisibility Active", MagicalUI::GOLD);
        }
        else if (calibRect.isCalibrated()){
            ui.drawStatus(display, "Press B to capture background", MagicalUI::CREAM);
            cv::putText(display, "Step OUT of frame first!", cv::Point(20, 70),cv::FONT_HERSHEY_SIMPLEX, 0.6,cv::Scalar(0, 255, 255), 2);
        }
        else{
            calibRect.draw(display);
            ui.drawStatus(display, "Place cloth - Press SPACE", MagicalUI::CREAM);
        }
        cv::putText(display, "[SPACE] Calibrate  [B] Background  [ESC] Return",cv::Point(20, display.rows - 20),cv::FONT_HERSHEY_SIMPLEX, 0.5, MagicalUI::CREAM, 1);
        cv::imshow("Invisibility Cloak", display);
        int key = cv::waitKey(1);
        if (key == 27) break;
        else if (key == 32 && !calibRect.isCalibrated()){
            if (calibRect.calibrate(frame)){
                cout << "Cloth calibrated!\n";
            }
        }
        else if (key == 'b' || key == 'B'){
            if (calibRect.isCalibrated()){
                background = frame.clone();
                backgroundCaptured = true;
                cout << "Background captured!\n";
            }
        }
        else if (key == 'r' || key == 'R'){
            calibRect.reset();
            backgroundCaptured = false;
            cout << "Reset - recalibrate cloth.\n";
        }
    }
    cam.release();
    cv::destroyWindow("Invisibility Cloak");
    cout << "Invisibility deactivated!\n\n";
}

void runQuidditchMode() {
    cout << "QUIDDITCH TRAINING\n";
    cout << "use flashlight to move the ring.\n";
    cout << "catch falling Golden Snitches\n";
    cout << "press esc to return.\n\n";
    
    const int hsv_thresh = 250;
    const int min_area = 100;
    const int max_area = 50000;
    
    getVideo cam(0);
    if (!cam.isOpened()){
        return;
    }
    
    lightDetector detector(hsv_thresh, min_area, max_area);
    MagicalUI ui;
    
    cv::Mat frame = cam.getFrame();
    if (frame.empty()){
        return;
    }
    QuidditchGame game(frame.cols, frame.rows);
    cout << "Game started!\n\n";
    int highScore = 0;
    while (true) {
        frame = cam.getFrame();
        if (frame.empty()){
            break;
        }
        
        cv::Mat display = frame.clone();
        cv::Point lightPos = detector.getBrightestPoint(frame);
        game.updateRingPosition(lightPos);
        
        game.update();
        game.draw(display);
        ui.drawScore(display, game.getScore(), "Score");
        
        if (game.getScore() > highScore){
            highScore = game.getScore();
        }
        
        string highScoreText = "High: " + to_string(highScore);
        cv::putText(display, highScoreText,cv::Point(display.cols - 150, display.rows - 20),cv::FONT_HERSHEY_SIMPLEX, 0.7, MagicalUI::GOLD, 2);
        
        if (lightPos.x != -1 && lightPos.y != -1){
            ui.drawStatus(display, "Catching Snitches!", MagicalUI::GOLD);
        }
        else{
            ui.drawStatus(display, "Point flashlight!", MagicalUI::CREAM);
        }
        cv::putText(display, "[R] Reset  [ESC] Return",cv::Point(display.cols / 2 - 100, display.rows - 20),cv::FONT_HERSHEY_SIMPLEX, 0.5, MagicalUI::CREAM, 1);
        
        cv::imshow("Quidditch Training", display);
        int key = cv::waitKey(17);
        if (key == 27) break;
        else if (key == 'r' || key == 'R') {
            game.reset();
            cout << "Game reset! Score: " << game.getScore() << endl;
        }
    }
    cam.release();
    cv::destroyWindow("Quidditch Training");
    cout << "Final Score: " << game.getScore() << endl;
    cout << "High Score: " << highScore << endl << endl;
}