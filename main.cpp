#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
using namespace std;
using namespace cv;
using namespace cv::ml;

// 处理帧图像，将其转换为HSV色彩空间，并根据指定的颜色范围创建红色和蓝色的掩码
void handel_frame(Mat &frame, Mat &hsv, Mat &mask_red, Mat &mask_blue)
{
    // 将输入的BGR图像转换为HSV色彩空间
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    
    // 定义红色和蓝色的HSV颜色范围
    Scalar lower_red1(8, 0, 200), upper_red1(23, 255, 255);
    Scalar lower_blue(85, 0, 180), upper_blue(105, 150, 255);
    
    // 根据红色和蓝色的HSV颜色范围创建掩码
    inRange(hsv, lower_red1, upper_red1, mask_red);
    inRange(hsv, lower_blue, upper_blue, mask_blue);
}

// 从掩码中选择矩形区域，并计算其相关属性
void select_rect(Mat &frame, Mat &mask, vector<RotatedRect> &minRects, vector<array<Point2f, 4>> &points, vector<Point2f> &center)
{
    // 存储找到的轮廓
    vector<vector<Point>> contours;

    // 将输入图像转换为二值图像
    Mat binaryMask;
    threshold(mask, binaryMask, 0, 255, THRESH_BINARY);

    // 查找轮廓
    findContours(binaryMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 遍历所有轮廓，计算最小面积矩形
    for (int i = 0; i < contours.size(); i++)
    {
        RotatedRect minRect = minAreaRect(contours[i]);
        minRects.push_back(minRect);

        // 获取最小面积矩形的四个顶点
        Point2f rect_points[4];
        minRect.points(rect_points);
        points.push_back({rect_points[0], rect_points[1], rect_points[2], rect_points[3]});

        // 存储矩形中心点
        center.push_back(minRect.center);
    }
}

// 计算所有最小面积矩形的宽高比
void aspectRatio(vector<RotatedRect> &minRects, vector<float> &aspectRatios)
{
    // 遍历所有最小面积矩形，计算并存储宽高比
    for (int i = 0; i < minRects.size(); i++)
    {   
        RotatedRect rect = minRects[i];
        float width = max(rect.size.width, rect.size.height);
        float height = min(rect.size.width, rect.size.height);
        
        aspectRatios.push_back(width / height);
    }
}
void calculate(RotatedRect &minRects1,RotatedRect& minRects2,Point2f &center1,Point2f &center2,float &calculateRatios)
{
    float distance = sqrt(pow(center1.x - center2.x, 2) + pow(center1.y - center2.y, 2));
    float width = max(minRects1.size.width, minRects1.size.height);
    calculateRatios = distance / width;
}
bool ifpair(float calculateRatios)
{
    if ((calculateRatios >=2.0&&calculateRatios<=2.6)||(calculateRatios >= 3.7&&calculateRatios <= 4.3))
    {
        return true;
    }
    else
    {
        return false;
    }
}


// 判断给定的宽高比是否满足目标条件
bool iftarget(float aspectRatio)
{
    // 检查宽高比是否大于等于3.2，以确定是否为目标对象
    if (aspectRatio >= 3.2)
    {
        return true;
    }
    else
    {
        return false;
    }
}

// 在图像帧上绘制矩形
void draw_rect(Mat &frame, Point2f point[4], Scalar color)
{
    // 绘制矩形的四个边
    for (int i = 0; i < 4; i++)
    {
        line(frame, point[i], point[(i + 1) % 4], color, 4);
    }
}

// 通过直方图反向投影进行图像模板匹配
/*void MatchTemplate(Mat &frame, Mat &img, Mat &hsv)
{
    Mat img_hsv;
    cvtColor(img, img_hsv, COLOR_BGR2HSV);
    int channels[] = {0, 1, 2};
    int histSize[] = {32, 32, 32};
    float hrange[] = {0, 180};
    float srange[] = {0, 256};
    float vrange[] = {0, 256};
    const float *ranges[] = {hrange, srange, vrange};
    Mat hist;
    calcHist(&img_hsv, 1, channels, Mat(), hist, 3, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
    Mat backproject;
    calcBackProject(&hsv, 1, channels, hist, backproject, ranges, 1, true);
    Mat binary;
    threshold(backproject, binary, 0, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
    {
        Rect selection = boundingRect(contours[i]);
        RotatedRect trackBox = CamShift(backproject, selection, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
        rectangle(frame, selection, Scalar(0, 255, 0), 3);
    }
}*/

Mat preprocessImage(const Mat &image, int inputSize) 
{
    Mat img, resizedImage;
    // 将图像调整为模型输入大小
    resize(image, img, Size(inputSize, inputSize));
    img.convertTo(resizedImage, CV_32F);
    Mat grayImage;
    // 转换为灰度图像
    cvtColor(resizedImage, grayImage, COLOR_BGR2GRAY);
    Mat normalizedImage = grayImage / 255.0;
    return normalizedImage;
}  
Mat preprocess(const Mat &image, int inputSize) 
{
    Mat img, resizedImage;
    // 将图像调整为模型输入大小
    resize(image, img, Size(inputSize, inputSize));
    img.convertTo(resizedImage, CV_32F);
    Mat normalizedImage = img / 255.0;
    return normalizedImage;
}   
void ifempty(Mat &img)
{
    if (img.empty())
    {
        cout << "img is empty" << endl;
    }
}
void draw_prediction(Mat &frame, int &prediction)
{
    putText(frame, "prediction: " + to_string(prediction), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
}

void getroi(Mat &img, vector<Mat> &img_roi,vector<Rect> &rects,vector<vector<Point>> &contours_roi)
{
    
    
    
    // 用于二值化的图像
    Mat gray,binary;
    cvtColor(img,gray,COLOR_BGR2GRAY);
    GaussianBlur(gray,gray,Size(13,13),4,4);
    threshold(gray,binary,170,255,THRESH_BINARY);
   
    vector<vector<Point>> contours;
    findContours(binary,contours,RETR_TREE,CHAIN_APPROX_SIMPLE);
    
    for (const auto &contour : contours)
    {
        // 计算当前轮廓的边界矩形
        Rect bounding_rect = boundingRect(contour);
        if(contour.size()<=3)
        {
            
            continue;
        }
        if(bounding_rect.width==0)
        {
            
            continue;
        }
        // 检查矩形的高度与宽度之比，以确定是否提取该区域
        if (bounding_rect.height / bounding_rect.width > 0.5 && bounding_rect.height / bounding_rect.width < 2.5)
        {
            // 提取感兴趣区域
            Mat sample = img(bounding_rect);
            // 将样本添加到存储向量中
            img_roi.push_back(sample);
            // 将边界矩形添加到存储向量中
            rects.push_back(bounding_rect);
            contours_roi.push_back(contour);
        }
    }
}
void extract_features1(const vector<vector<Point>> &contour,vector<vector<double>>& train_features,vector<double>& train_labels)
{   
    for(auto &contours:contour)
    {
        vector<double> features;
        if(contour.size()>3)
        {
        
            continue;
                   
        }
    
        double perimeter = arcLength(contour, true);
        Rect bounding_rect = boundingRect(contour);
        double aspect_ratio = static_cast<double>(bounding_rect.width) / bounding_rect.height;
        double compactness = contourArea(contour) / (perimeter * perimeter);
        
        features.push_back(aspect_ratio);
        features.push_back(compactness);
        train_features.push_back(features);
        if ((features[0] > 0.5 && features[0] < 3.0) && (features[1] > 0.05 && features[1] < 0.5))
            {
                train_labels.push_back(1);
            }
            else
            {
                train_labels.push_back(0);
            }
    }

    
}
vector<double> extract_features2(const vector<Point> &contour)
{
    vector<double> features;
    
    
        double perimeter = arcLength(contour, true);
        Rect bounding_rect = boundingRect(contour);
        double aspect_ratio = static_cast<double>(bounding_rect.width) / bounding_rect.height;
        double compactness = contourArea(contour) / (perimeter * perimeter);
        
        features.push_back(aspect_ratio);
        features.push_back(compactness);
    
    return features;
}
/*dlib::decision_function<dlib::linear_kernel<dlib::matrix<double, 0, 1>>>
train_decision_tree(const std::vector<std::vector<double>>& features, const std::vector<double>& labels) {
using sample_type = dlib::matrix<double, 0, 1>;
using kernel_type = dlib::linear_kernel<sample_type>;
using decision_function_type = dlib::decision_function<kernel_type>;

// 准备训练数据
vector<sample_type> samples;
for (const auto& feature : features) 
{
    sample_type sample(feature.size());
    for (size_t i = 0; i < feature.size(); ++i)     
    {
        sample(i) = feature[i];
    }
    samples.push_back(sample);
}

// 创建并配置训练器
dlib::svm_c_trainer<kernel_type> trainer;
trainer.set_c(10); // 设置正则化参数C

// 训练模型
decision_function_type df = trainer.train(samples, labels);

return df;
}*/

// 主函数，处理视频流
int main()
{
    
    // 打开视频文件
    VideoCapture test2("/home/yuan/桌面/test/test2.mp4");
    if (!test2.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
    Mat img1 = imread("/home/yuan/桌面/test/1.png");
    Mat img2 = imread("/home/yuan/桌面/test/2.png");
    Mat img3 = imread("/home/yuan/桌面/test/3.png");
    Mat img4 = imread("/home/yuan/桌面/test/4.png");
    Mat img7 = imread("/home/yuan/桌面/test/7.png");
    ifempty(img1); ifempty(img2); ifempty(img3); ifempty(img4); ifempty(img7); 
    Ptr<RTrees> dtree = RTrees::create();
    dtree->setMaxDepth(10);
    dtree->setMinSampleCount(0);
    dtree->setRegressionAccuracy(0.01f);
    dtree->setUseSurrogates(false);
    dtree->setMaxCategories(5);
    dtree->setCVFolds(1);
    dtree->setUse1SERule(false);
    dtree->setTruncatePrunedTree(true);

    // 准备训练数据
    Mat samples, responses;
    vector<Mat> imgList = {img1, img2, img3, img4, img7};
    vector<int> responseList = {1, 2, 3, 4, 7};

    // 准备训练数据
    for(int i = 0; i < imgList.size(); i++)
    {
        Mat sample;
        sample = preprocessImage(imgList[i], 32);
        sample = sample.reshape(1, 1);
        samples.push_back(sample);
        
    
    
        Mat response = (Mat_<int>(1, 1) << responseList[i]);
        responses.push_back(response);
    }
    // 训练决策树
    dtree->train(samples, ROW_SAMPLE, responses);
    vector<vector<double>> train_features;
    vector<double> train_labels;
    /*for(int i=0;i<imgList.size();i++)
    {
        Mat gray,binary;
        cvtColor(imgList[i],gray,COLOR_BGR2GRAY);
        GaussianBlur(gray,gray,Size(13,13),4,4);
        threshold(gray,binary,170,255,THRESH_BINARY|THRESH_OTSU);
        vector<vector<Point>> contours;
        findContours(binary,contours,RETR_TREE,CHAIN_APPROX_SIMPLE);
        if(contours.size()==0)
        {
            cout<<"contours is empty"<<endl;
            continue;
        }
        extract_features1(contours,train_features,train_labels);  
        cout<<"train_features size:"<<train_features.size()<<endl;   
    }    
    auto df = train_decision_tree(train_features, train_labels);*/
    Mat frame;
    // 逐帧读取并处理视频
    while (test2.read(frame))
    {
        // 存储视频帧和处理结果的变量
        Mat hsv, mask_red, mask_blue;
        vector<RotatedRect> minRects_red, minRects_blue;
        vector<array<Point2f, 4>> points_red, points_blue;
        vector<Point2f> center_red, center_blue;
        vector<float> aspectRatios_red, aspectRatios_blue;

        // 预处理一帧图像以获取HSV图像和颜色掩码
        handel_frame(frame, hsv, mask_red, mask_blue);

        // 从红色和蓝色掩码中选择矩形区域
        select_rect(frame, mask_red, minRects_red, points_red, center_red);
        select_rect(frame, mask_blue, minRects_blue, points_blue, center_blue);

        // 计算所有检测到的矩形的宽高比
        aspectRatio(minRects_red, aspectRatios_red);
        aspectRatio(minRects_blue, aspectRatios_blue);

        // 遍历所有检测到的矩形，判断是否为目标对象
        for (int i = 0; i < minRects_red.size(); i++)
        {
            for(int j = 1; j < i; j++)
            {   // 获取当前矩形的宽高比
                float aspectRatio_red1 = aspectRatios_red[i];
                float aspectRatio_red2 = aspectRatios_red[j];
                // 判断是否为目标对象
                bool iftarget_red1= iftarget(aspectRatio_red1);
                bool iftarget_red2 = iftarget(aspectRatio_red2);
                float calculation;
                if (iftarget_red1&&iftarget_red2)
                {
                    calculate(minRects_red[i], minRects_red[j], center_red[i], center_red[j],calculation);
                    bool ifPair=ifpair(calculation);
                    if(ifPair)
                    {
                        
                        draw_rect(frame, points_red[i].data(), Scalar(255,0,0));
                        draw_rect(frame, points_red[j].data(), Scalar(255,0,0));
                        
                    }
                }
            }
        }
        for (int i = 0; i < minRects_blue.size(); i++)
        {
            for(int j = 1; j < i; j++)
            {   // 获取当前矩形的宽高比
                float aspectRatio_blue1 = aspectRatios_blue[i];
                float aspectRatio_blue2 = aspectRatios_blue[j];
                // 判断是否为目标对象
                bool iftarget_blue1= iftarget(aspectRatio_blue1);
                bool iftarget_blue2 = iftarget(aspectRatio_blue2);
                float calculation;
                if (iftarget_blue1&&iftarget_blue2)
                {
                    calculate(minRects_blue[i], minRects_blue[j], center_blue[i], center_blue[j],calculation);
                    bool ifPair=ifpair(calculation);
                    if(ifPair)
                    {
                        
                        draw_rect(frame, points_blue[i].data(), Scalar(0,0,255));
                        draw_rect(frame, points_blue[j].data(), Scalar(0,0,255));
                         
                    }
                }
            }
        }

        // 预处理当前帧图像
        vector<Mat> img_roi;
        vector<Rect> rects;
        vector<vector<Point>> contours;
        getroi(frame, img_roi,rects,contours);

        for(int i = 0; i < contours.size(); i++)
        {
            Mat sample1;
            sample1 = preprocessImage(img_roi[i], 32);
            sample1 = sample1.reshape(1, 1);
            
            
            for (auto& contour : contours)
            {
                
                
            
                vector<double> features;
                if(contour.size()<3)
                {
                
                    continue;
                }
    
                double perimeter = arcLength(contour, true);
            
                Rect bounding_rect = boundingRect(contour);
                 
                double aspect_ratio = static_cast<double>(bounding_rect.width) / bounding_rect.height;
                double compactness = contourArea(contour) / (perimeter * perimeter);
                
                features.push_back(aspect_ratio);
                features.push_back(compactness);
                /*dlib::matrix<double, 0, 1> sample2(features.size());
                for (size_t i = 0; i < features.size(); ++i)    
                {
                    sample2(i) = features[i];
                }*/
                double prediction_fea ;
                if ((features[0] > 0.5 && features[0] < 2.5) && (features[1] > 0.01 && features[1] < 0.1))
                {
                    prediction_fea = 1;
                }
                else
                {   
                    prediction_fea = 0;
                }

                if(prediction_fea>0)
                {
                    double prediction =dtree->predict(sample1) ;
                
            
                    rectangle(frame, rects[i], Scalar(0, 255, 0), 2);
                    //putText(frame, to_string(prediction), Point(rects[i].x, rects[i].y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
            
            
                }
            } 
        }
        

        
        // 显示处理后的帧
        imshow("frame", frame);
        if (waitKey(10) == 27)
        {
            break;
        }
    }
        
    return 0;
    
}