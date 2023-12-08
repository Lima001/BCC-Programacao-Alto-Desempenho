/*
    Código baseado e adaptado de: 
    https://docs.opencv.org/4.x/dc/ddf/tutorial_how_to_use_OpenCV_parallel_for_new.html 
*/

#include <iostream>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void convolutionSerial(cv::Mat src, cv::Mat &dst, cv::Mat kernel){
    int rows = src.rows, cols = src.cols;
    dst = cv::Mat(rows, cols, src.type());

    int sz = kernel.rows / 2;
    
    copyMakeBorder(src, src, sz, sz, sz, sz, cv::BORDER_CONSTANT);

    for (int i = 0; i < rows; i++){
        uchar* dptr = dst.ptr<uchar>(i);
        
        for (int j = 0; j < cols; j++){

            float value = 0.0;

            for (int k = -sz; k <= sz; k++){
                
                uchar* sptr = src.ptr<uchar>(i + sz + k);
                for (int l = -sz; l <= sz; l++){
                    value += kernel.ptr<float>(k + sz)[l + sz] * (float)sptr[j + sz + l];
                }
            }
            dptr[j] = cv::saturate_cast<uchar>(value);
        }
    }
}

// Função abaixo utiliza o recurso de funções lambdas - necessário C++ >= 11 
void convolutionParallel(cv::Mat src, cv::Mat &dst, cv::Mat kernel){
    int rows = src.rows, cols = src.cols;
    dst = cv::Mat(rows, cols, src.type());
    int sz = kernel.rows / 2;
    
    copyMakeBorder(src, src, sz, sz, sz, sz, cv::BORDER_CONSTANT);

    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range &range){
        for (int i = range.start; i < range.end; i++){

            uchar *dptr = dst.ptr<uchar>(i);
            for (int j = 0; j < cols; j++){

                float value = 0.0;
                for (int k = -sz; k <= sz; k++){

                    uchar *sptr = src.ptr<uchar>(i + sz + k);
                    for (int l = -sz; l <= sz; l++){
                        value += kernel.ptr<float>(k + sz)[l + sz] * (float)sptr[j + sz + l];
                    }
                }
                dptr[j] = cv::saturate_cast<uchar>(value);
            }
        }
    });
}

cv::Mat getSampleMat(int height, int width){
    cv::Mat kernel = cv::Mat(height,width,CV_32FC1);
    
    for (int i=0; i<height; i++){
        for (int j=0; j<width; j++){
            kernel.at<float>(i,j) = std::rand()%256;
        }
    }

    return kernel;
}

int main(int argc, char *argv[]){
    
    if (argc != 4)
        return 1;

    int img_height = std::atoi(argv[1]);
    int img_width = std::atoi(argv[2]);
    int kernel_dim = std::atoi(argv[3]);

    cv::Mat img = getSampleMat(img_height, img_width);
    cv::Mat kernel = getSampleMat(kernel_dim, kernel_dim);
    cv::Mat output;

    double t = (double)cv::getTickCount();
    convolutionSerial(img, output, kernel);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << " Sequential Implementation: " << t << " s" << std::endl;

    t = (double)cv::getTickCount();
    convolutionParallel(img, output, kernel);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << " Parallel Implementation: " << t << " s" << std::endl;

    return 0;
}
