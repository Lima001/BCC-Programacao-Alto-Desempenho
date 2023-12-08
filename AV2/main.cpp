/*
    Código baseado e adaptado de: 
    https://docs.opencv.org/4.x/dc/ddf/tutorial_how_to_use_OpenCV_parallel_for_new.html 
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void convolutionSerial(cv::Mat src, cv::Mat &dst, cv::Mat kernel){
    // Altura e largura da imagem de entrada - dimensões da matriz
    int rows = src.rows, cols = src.cols;
    
    // Inicialização da matriz (imagem) de mesma dimensão e tipo da matriz de entrada
    dst = cv::Mat(rows, cols, src.type());

    // Centro do Kernel
    int sz = kernel.rows / 2;
    
    // Extender as bordas na matriz de entrada
    copyMakeBorder(src, src, sz, sz, sz, sz, cv::BORDER_REPLICATE);

    for (int i = 0; i < rows; i++){
        /*
            Ponteiro para vetor (pixel) na linha "i".
            Segundo documentação do OpenCV, matrizes são armazenadas de forma contínua
            na memória. Sendo asssim, é mais eficiente acessar posições via aritmética
            de ponteiros, do que utilizar outras funções/métodos - como at().
        */
        cv::Vec3b* dptr = dst.ptr<cv::Vec3b>(i);
        for (int j = 0; j < cols; j++){
            // Vetor contendo o somatório da convolução
            cv::Vec3f value(0,0,0);

            // Aplicação da convolução
            for (int k = -sz; k <= sz; k++){
                
                cv::Vec3b* sptr = src.ptr<cv::Vec3b>(i + sz + k);
                for (int l = -sz; l <= sz; l++){
                    // Convolução para os três canais da imagem (BGR)
                    value[0] += kernel.ptr<float>(k + sz)[l + sz] * (float)sptr[j + sz + l][0];
                    value[1] += kernel.ptr<float>(k + sz)[l + sz] * (float)sptr[j + sz + l][1];
                    value[2] += kernel.ptr<float>(k + sz)[l + sz] * (float)sptr[j + sz + l][2];
                }
            }
            // Garantir que os valores dos pixels sejam discretos e estejam entre 0 e 255
            dptr[j][0] = cv::saturate_cast<uchar>(value[0]);
            dptr[j][1] = cv::saturate_cast<uchar>(value[1]);
            dptr[j][2] = cv::saturate_cast<uchar>(value[2]);
        }
    }
}

// Função abaixo utiliza o recurso de funções lambdas - necessário C++ >= 11 
void convolutionParallel(cv::Mat src, cv::Mat &dst, cv::Mat kernel){
    // Altura e largura da imagem de entrada - dimensões da matriz
    int rows = src.rows, cols = src.cols;
    
    // Inicialização da matriz (imagem) de mesma dimensão e tipo da matriz de entrada
    dst = cv::Mat(rows, cols, src.type());

    // Centro do Kernel
    int sz = kernel.rows / 2;
    
    // Extender as bordas na matriz de entrada
    copyMakeBorder(src, src, sz, sz, sz, sz, cv::BORDER_REPLICATE);

    /*
        Paralelismo utilizando função parallel_for_. Essa função permite definir
        um intervalo de dados que será dividido. Esses subintervalos são então
        processados (usando a função lambda) em diferentes threads usando algum dos
        frameworks suportados pelo OpenCV.

        Para mais detalhes, verificar a referência mencionado no início do arquivo.

        A título de especificação, a divisão do intervalo de dados adotada consiste
        em separar um conjunto de linhas da matriz. Dessa forma, diferentes threads
        processarão linhas diferentes.
    */
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range &range){
        for (int i = range.start; i < range.end; i++){

            cv::Vec3b *dptr = dst.ptr<cv::Vec3b>(i);
            for (int j = 0; j < cols; j++){

                cv::Vec3f value(0,0,0);
                for (int k = -sz; k <= sz; k++){

                    cv::Vec3b *sptr = src.ptr<cv::Vec3b>(i + sz + k);
                    for (int l = -sz; l <= sz; l++){
                        value[0] += kernel.ptr<float>(k + sz)[l + sz] * (float)sptr[j + sz + l][0];
                        value[1] += kernel.ptr<float>(k + sz)[l + sz] * (float)sptr[j + sz + l][1];
                        value[2] += kernel.ptr<float>(k + sz)[l + sz] * (float)sptr[j + sz + l][2];
                    }
                }
                dptr[j][0] = cv::saturate_cast<uchar>(value[0]);
                dptr[j][1] = cv::saturate_cast<uchar>(value[1]);
                dptr[j][2] = cv::saturate_cast<uchar>(value[2]);
            }
        }
    });
}

// Gera um kernel - para detecção de arestas - de exemplo 
cv::Mat getSampleKernel(){
    cv::Mat kernel = cv::Mat(3,3,CV_32FC1);
    
    kernel.at<float>(0,0) = -1;
    kernel.at<float>(0,1) = -1;
    kernel.at<float>(0,2) = -1;
    kernel.at<float>(1,0) = -1;
    kernel.at<float>(1,1) = 8;
    kernel.at<float>(1,2) = -1; 
    kernel.at<float>(2,0) = -1;
    kernel.at<float>(2,1) = -1;
    kernel.at<float>(2,2) = -1;

    return kernel;
}

int main(int argc, char *argv[]){
    
    // Espera receber a imagem a ser processada via linha de comando
    // Caso contrário, utiliza uma imagem padrão disponibilizada
    const char *filepath = argc >= 2 ? argv[1] : "imgs/dog.jpg";

    // Inicialização das matrizes
    cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);
    cv::Mat kernel = getSampleKernel();
    cv::Mat output;

    // Escolha uma função e descomente o código se quiser processar e salvar a imagem
    //convolutionSerial(img,output,kernel);
    //convolutionParallel(img,output,kernel);
    //cv::imwrite("result.jpg",output);

    // Realizar processamento serial e exibir o tempo gasto
    double t = (double)cv::getTickCount();
    convolutionSerial(img, output, kernel);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << " Sequential Implementation: " << t << "s" << std::endl;
    // Descomente se quiser verificar o resultado do processamento
    //cv::imshow("OutSerial", output);
    //cv::waitKey(0);

    // Realizar processamento paralel e exibir o tempo gasto
    t = (double)cv::getTickCount();
    convolutionParallel(img, output, kernel);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << " Parallel Implementation: " << t << "s" << std::endl;
    // Descomente se quiser verificar o resultado do processamento
    //cv::imshow("OutParallel", output);
    //cv::waitKey(0);

    return 0;
}
