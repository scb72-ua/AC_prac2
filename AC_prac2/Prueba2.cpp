#include <iostream>
#include <chrono>
#include "opencv2/opencv.hpp"

int height;
int width;

const int BLUR_ITERATIONS = 40;
const int SINGLE_ITERATION = 1;
const int SMOOTHING_ITERATIONS = 20;
const unsigned KERNEL_MASK_SSE[4] = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0 };

int kernelData1[3][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} }; // BLUR
int kernelData2[3][3] = { {1, 1, 1}, {1, 2, 1}, {1, 1, 1} }; // SMOOTHING
int kernelData3[3][3] = { {0, 1, 0}, {1, -4, 1}, {0, 1, 0} }; // EDGE DETECTION
int kernelData4[3][3] = { {-2, -1, 0}, {-1, 1, 1}, {0, 1, 2} }; // EMBOSSING

int** loadImage(const std::string& route) {
    cv::Mat image = cv::imread(route, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "ERROR: can't read image" << std::endl;
    }

    height = image.rows;
    width = image.cols;
    int** arrayImage = new int* [height];
    for (int i = 0; i < height; i++) {
        arrayImage[i] = new int[width];
        for (int j = 0; j < width; j++) {
            arrayImage[i][j] = static_cast<int>(image.at<uchar>(i, j));
        }
    }
    return arrayImage;
}

void applyFilter(int** originalImage, int** resultImage, int iterations, int kernelData[3][3]) {
    int** temp = new int* [height];

    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        temp[i] = new int[width];
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    // Apply the filter with the specified number of iterations
    for (int iteration = 0; iteration < iterations; iteration++) {

        for (int i = 1; i < height - 1; i++) {
            for (int j = 1; j < width - 1; j++) {
                int sum = 0;
                for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                        sum += resultImage[i + m - 1][j + n - 1] * kernelData[m][n];
                    }
                }
                temp[i][j] = (sum / 9);  // Normalize the result
            }
        }

        // Copy the temporary result to the final result
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }

    // Free temporary matrix memory
    for (int i = 0; i < height; i++) {
        delete[] temp[i];
    }
    delete[] temp;
}

void applyFilterx86(int** originalImage, int** resultImage, int iterations, int kernelData[3][3]) {
    int** temp = new int* [height];

    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        temp[i] = new int[width];
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    // Apply the filter with the specified number of iterations
    for (int iteration = 0; iteration < iterations; iteration++) {
        int row = 1, col = 1, k_row = 0, k_col = 0, sum = 0;

        __asm {
            mov row, 1
            r_loop:
            mov edx, height
                sub edx, 1
                cmp row, edx
                jnb end_r_loop

                mov col, 1
                c_loop:
            mov edx, width
                sub edx, 1
                cmp col, edx
                jnb end_c_loop

                mov sum, 0
                mov k_row, 0
                kr_loop:
            cmp k_row, 3
                jnb end_kr_loop

                mov k_col, 0
                kc_loop :
                cmp k_col, 3
                jnb end_kc_loop

                // Calculate 'resultImage' row offset
                mov edx, row    // offset = curr_row
                add edx, k_row  // offset += curr_kernelRow
                sub edx, 1      // offset -= 1
                // Get row address
                mov esi, resultImage
                mov ebx, [esi + 4 * edx]
                mov esi, ebx

                // Calculate 'resultImage' column offset
                mov edx, col    // offset = curr_column
                add edx, k_col  // offset += curr_kernelColumn
                sub edx, 1      // offset -= 1
                // Get resultImage[i + m -1][j + n -1]
                mov edx, [esi + 4 * edx]

                // Get kernel[m][n]
                mov ecx, k_row      // edx <- current_kernelRow
                imul ecx, 3         // multiply row number by the size of each row
                add ecx, k_col      // add column number
                lea esi, kernelData // get kernelData address
                mov ecx, [esi + 4 * ecx]    // retrieve element
                imul edx, ecx
                add sum, edx

                add k_col, 1
                jmp kc_loop
                end_kc_loop :

            add k_row, 1
                jmp kr_loop
                end_kr_loop :

            // Store calculated value in temporary result matrix
            mov edi, temp
                mov edx, row
                mov edi, [edi + 4 * edx]    // edi <- temp[i] (row address)

                mov eax, sum
                mov edx, 0
                mov ecx, 9
                cdq
                idiv ecx                    // eax <- sum/9

                mov edx, col
                mov[edi + 4 * edx], eax     // temp[i][j] <- sum/9

                add col, 1
                jmp c_loop
                end_c_loop :

            add row, 1
                jmp r_loop
                end_r_loop :
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }

    // Free temporary matrix memory
    for (int i = 0; i < height; i++) {
        delete[] temp[i];
    }
    delete[] temp;
}

void applyFilterSSE(int** originalImage, int** resultImage, int iterations, int kernelData[3][3]) {
    int** temp = new int* [height];

    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        temp[i] = new int[width];
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    int iterableHeight = height - 1, iterableWidth = width - 1;

    for (int iteration = 0; iteration < iterations; iteration++) {
        int sum = 0, kernelRow = 0, row = 1, col = 1;

        __asm {
            mov ecx, 9
            movups xmm2, KERNEL_MASK_SSE
            mov esi, temp

            row_loop :
            mov edx, iterableHeight
                cmp row, edx // compare if row < height - 1
                jnb end_row_loop

                mov col, 1
                col_loop :
                mov sum, 0 // clear the sum for each element
                mov edx, iterableWidth
                cmp col, edx // compare if col < width - 1
                jnb end_col_loop

                // save current row and column in the stack
                push row
                push col

                // move to the first element of the 3x3 submatrix
                sub row, 1
                sub col, 1

                mov kernelRow, 0
                // in each iteration a row of the 3x3 submatrix is multiplied by
                // the corresponding row of the kernel matrix
                kernel_loop:
            cmp kernelRow, 3
                jnb end_kernel_loop

                // access the image
                mov edx, row
                add edx, kernelRow
                mov edi, resultImage
                mov ebx, [edi + 4 * edx] // get row address
                mov edx, col
                movups xmm0, [ebx + 4 * edx] // move 4 integers to xmm0
                cvtdq2ps xmm0, xmm0 // convert to float (required for multiplication)

                // access the kernel matrix
                mov edx, kernelRow
                imul edx, 3 // multiply row number by the size of each row
                lea edi, kernelData
                movups xmm1, [edi + 4 * edx] // move 4 integers to xmm0
                cvtdq2ps xmm1, xmm1 // convert to float (required for multiplication)
                andps xmm1, xmm2 // apply the mask to set to 0 the bits 96:127 (we don't want that element)

                // multiply both rows
                mulps xmm0, xmm1

                // perform the addition of multiplication results
                haddps xmm0, xmm0
                haddps xmm0, xmm0
                cvtss2si eax, xmm0 // convert the sum to integer and store it in eax
                add sum, eax

                add kernelRow, 1
                jmp kernel_loop

                end_kernel_loop :
            mov eax, sum
                cdq // sign extension
                idiv ecx // eax <- sum / 9

                // retrieve row and col from the stack to store the result in the corresponding element
                pop col
                pop row
                mov edx, row
                mov ebx, [esi + 4 * edx]
                mov edx, col
                mov[ebx + 4 * edx], eax // temp[i][j] <- sum / 9

                add col, 1
                jmp col_loop
                end_col_loop :

            add row, 1
                jmp row_loop
                end_row_loop :
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }

    // Free temporary matrix memory
    for (int i = 0; i < height; i++) {
        delete[] temp[i];
    }
    delete[] temp;
}

void saveImage(const std::string& route, int** image) {
    // Crear la imagen desenfocada
    cv::Mat matImage(height, width, CV_8U);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            matImage.at<uchar>(i, j) = static_cast<uchar>(image[i][j]);
        }
    }

    cv::imwrite(route, matImage);
}

int main() {
        
    int** originalImage = loadImage("imagen.jpg");

    int** temp = new int* [height];
    int** resultImage = new int* [height];
    for (int i = 0; i < height; i++) {
        temp[i] = new int[width];
        resultImage[i] = new int[width];
    }
    
    
    // ---------- C BENCHMARK ----------

    std::cout << "C BENCHMARK" << std::endl;

    auto totalStartTime = std::chrono::high_resolution_clock::now();
    auto startTime = std::chrono::high_resolution_clock::now();
    applyFilter(originalImage, resultImage, BLUR_ITERATIONS, kernelData1);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;

    std::cout << "\tFirst filter time: " << elapsedTime.count() << " seconds" << std::endl;

    // Save the blurred image
    saveImage("imagen_desenfocada.jpg", resultImage);

    //-------------------------------------------

    startTime = std::chrono::high_resolution_clock::now();
    applyFilter(originalImage, resultImage, SINGLE_ITERATION, kernelData2);
    endTime = std::chrono::high_resolution_clock::now();
    elapsedTime = endTime - startTime;

    std::cout << "\tSecond filter time: " << elapsedTime.count() << " seconds" << std::endl;

    // Save the smoothed image
    saveImage("imagen_suavizado.jpg", resultImage);

    //----------------------------------------------

    startTime = std::chrono::high_resolution_clock::now();
    applyFilter(originalImage, resultImage, SINGLE_ITERATION, kernelData3);
    endTime = std::chrono::high_resolution_clock::now();
    elapsedTime = endTime - startTime;

    std::cout << "\tThird filter time: " << elapsedTime.count() << " seconds" << std::endl;

    // Save the image with edge detection
    saveImage("imagen_deteccionBordes.jpg", resultImage);

    //----------------------------------------------

    startTime = std::chrono::high_resolution_clock::now();
    applyFilter(originalImage, resultImage, SINGLE_ITERATION, kernelData4);
    endTime = std::chrono::high_resolution_clock::now();
    auto totalEndTime = std::chrono::high_resolution_clock::now();

    elapsedTime = endTime - startTime;
    std::cout << "\tFourth filter time: " << elapsedTime.count() << " seconds" << std::endl;

    std::chrono::duration<double> totalElapsedTime = totalEndTime - totalStartTime;
    std::cout << "\tTOTAL TIME: " << totalElapsedTime.count() << " seconds" << std::endl;

    // Save the embossed image
    saveImage("imagen_repujada.jpg", resultImage);


    // ---------- x86 BENCHMARK ----------

    std::cout << std::endl << "x86 BENCHMARK" << std::endl;

    totalStartTime = std::chrono::high_resolution_clock::now();
    startTime = std::chrono::high_resolution_clock::now();


    // FIRST FILTER


    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    // Apply the filter with the specified number of iterations
    for (int iteration = 0; iteration < BLUR_ITERATIONS; iteration++) {
        int row = 1, col = 1, k_row = 0, k_col = 0, sum = 0;

        __asm {
            mov row, 1
            r_loop_1:
            mov edx, height
                sub edx, 1
                cmp row, edx
                jnb end_r_loop_1

                mov col, 1
                c_loop_1:
            mov edx, width
                sub edx, 1
                cmp col, edx
                jnb end_c_loop_1

                mov sum, 0
                mov k_row, 0
                kr_loop_1:
            cmp k_row, 3
                jnb end_kr_loop_1

                mov k_col, 0
                kc_loop_1 :
                cmp k_col, 3
                jnb end_kc_loop_1

                // Calculate 'resultImage' row offset
                mov edx, row    // offset = curr_row
                add edx, k_row  // offset += curr_kernelRow
                sub edx, 1      // offset -= 1
                // Get row address
                mov esi, resultImage
                mov ebx, [esi + 4 * edx]
                mov esi, ebx

                // Calculate 'resultImage' column offset
                mov edx, col    // offset = curr_column
                add edx, k_col  // offset += curr_kernelColumn
                sub edx, 1      // offset -= 1
                // Get resultImage[i + m -1][j + n -1]
                mov edx, [esi + 4 * edx]

                // Get kernel[m][n]
                mov ecx, k_row      // edx <- current_kernelRow
                imul ecx, 3         // multiply row number by the size of each row
                add ecx, k_col      // add column number
                lea esi, kernelData1 // get kernelData address
                mov ecx, [esi + 4 * ecx]    // retrieve element
                imul edx, ecx
                add sum, edx

                add k_col, 1
                jmp kc_loop_1
                end_kc_loop_1 :

            add k_row, 1
                jmp kr_loop_1
                end_kr_loop_1 :

            // Store calculated value in temporary result matrix
            mov edi, temp
                mov edx, row
                mov edi, [edi + 4 * edx]    // edi <- temp[i] (row address)

                mov eax, sum
                mov edx, 0
                mov ecx, 9
                cdq
                idiv ecx                    // eax <- sum/9

                mov edx, col
                mov[edi + 4 * edx], eax     // temp[i][j] <- sum/9

                add col, 1
                jmp c_loop_1
                end_c_loop_1 :

            add row, 1
                jmp r_loop_1
                end_r_loop_1 :
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }


    endTime = std::chrono::high_resolution_clock::now();
    elapsedTime = endTime - startTime;

    std::cout << "\tFirst filter time: " << elapsedTime.count() << " seconds" << std::endl;

    // Save the blurred image
    saveImage("imagen_desenfocadax86.jpg", resultImage);

    //-------------------------------------------

    startTime = std::chrono::high_resolution_clock::now();

    // SECOND FILTER


    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    // Apply the filter with the specified number of iterations
    for (int iteration = 0; iteration < SMOOTHING_ITERATIONS; iteration++) {
        int row = 1, col = 1, k_row = 0, k_col = 0, sum = 0;

        __asm {
            mov row, 1
            r_loop_2:
            mov edx, height
                sub edx, 1
                cmp row, edx
                jnb end_r_loop_2

                mov col, 1
                c_loop_2:
            mov edx, width
                sub edx, 1
                cmp col, edx
                jnb end_c_loop_2

                mov sum, 0
                mov k_row, 0
                kr_loop_2:
            cmp k_row, 3
                jnb end_kr_loop_2

                mov k_col, 0
                kc_loop_2 :
                cmp k_col, 3
                jnb end_kc_loop_2

                // Calculate 'resultImage' row offset
                mov edx, row    // offset = curr_row
                add edx, k_row  // offset += curr_kernelRow
                sub edx, 1      // offset -= 1
                // Get row address
                mov esi, resultImage
                mov ebx, [esi + 4 * edx]
                mov esi, ebx

                // Calculate 'resultImage' column offset
                mov edx, col    // offset = curr_column
                add edx, k_col  // offset += curr_kernelColumn
                sub edx, 1      // offset -= 1
                // Get resultImage[i + m -1][j + n -1]
                mov edx, [esi + 4 * edx]

                // Get kernel[m][n]
                mov ecx, k_row      // edx <- current_kernelRow
                imul ecx, 3         // multiply row number by the size of each row
                add ecx, k_col      // add column number
                lea esi, kernelData2 // get kernelData address
                mov ecx, [esi + 4 * ecx]    // retrieve element
                imul edx, ecx
                add sum, edx

                add k_col, 1
                jmp kc_loop_2
                end_kc_loop_2 :

            add k_row, 1
                jmp kr_loop_2
                end_kr_loop_2 :

            // Store calculated value in temporary result matrix
            mov edi, temp
                mov edx, row
                mov edi, [edi + 4 * edx]    // edi <- temp[i] (row address)

                mov eax, sum
                mov edx, 0
                mov ecx, 9
                cdq
                idiv ecx                    // eax <- sum/9

                mov edx, col
                mov[edi + 4 * edx], eax     // temp[i][j] <- sum/9

                add col, 1
                jmp c_loop_2
                end_c_loop_2 :

            add row, 1
                jmp r_loop_2
                end_r_loop_2 :
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }



    endTime = std::chrono::high_resolution_clock::now();
    elapsedTime = endTime - startTime;

    std::cout << "\tSecond filter time: " << elapsedTime.count() << " seconds" << std::endl;

    // Save the smoothed image
    saveImage("imagen_suavizadox86.jpg", resultImage);

    //----------------------------------------------

    startTime = std::chrono::high_resolution_clock::now();

    
    // THIRD FILTER


    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    // Apply the filter with the specified number of iterations
    for (int iteration = 0; iteration < SINGLE_ITERATION; iteration++) {
        int row = 1, col = 1, k_row = 0, k_col = 0, sum = 0;

        __asm {
            mov row, 1
            r_loop_3:
            mov edx, height
                sub edx, 1
                cmp row, edx
                jnb end_r_loop_3

                mov col, 1
                c_loop_3:
            mov edx, width
                sub edx, 1
                cmp col, edx
                jnb end_c_loop_3

                mov sum, 0
                mov k_row, 0
                kr_loop_3:
            cmp k_row, 3
                jnb end_kr_loop_3

                mov k_col, 0
                kc_loop_3 :
                cmp k_col, 3
                jnb end_kc_loop_3

                // Calculate 'resultImage' row offset
                mov edx, row    // offset = curr_row
                add edx, k_row  // offset += curr_kernelRow
                sub edx, 1      // offset -= 1
                // Get row address
                mov esi, resultImage
                mov ebx, [esi + 4 * edx]
                mov esi, ebx

                // Calculate 'resultImage' column offset
                mov edx, col    // offset = curr_column
                add edx, k_col  // offset += curr_kernelColumn
                sub edx, 1      // offset -= 1
                // Get resultImage[i + m -1][j + n -1]
                mov edx, [esi + 4 * edx]

                // Get kernel[m][n]
                mov ecx, k_row      // edx <- current_kernelRow
                imul ecx, 3         // multiply row number by the size of each row
                add ecx, k_col      // add column number
                lea esi, kernelData3 // get kernelData address
                mov ecx, [esi + 4 * ecx]    // retrieve element
                imul edx, ecx
                add sum, edx

                add k_col, 1
                jmp kc_loop_3
                end_kc_loop_3 :

            add k_row, 1
                jmp kr_loop_3
                end_kr_loop_3 :

            // Store calculated value in temporary result matrix
            mov edi, temp
                mov edx, row
                mov edi, [edi + 4 * edx]    // edi <- temp[i] (row address)

                mov eax, sum
                mov edx, 0
                mov ecx, 9
                cdq
                idiv ecx                    // eax <- sum/9

                mov edx, col
                mov[edi + 4 * edx], eax     // temp[i][j] <- sum/9

                add col, 1
                jmp c_loop_3
                end_c_loop_3 :

            add row, 1
                jmp r_loop_3
                end_r_loop_3 :
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }



    
    endTime = std::chrono::high_resolution_clock::now();
    elapsedTime = endTime - startTime;

    std::cout << "\tThird filter time: " << elapsedTime.count() << " seconds" << std::endl;

    // Save the image with edge detection
    saveImage("imagen_deteccionBordesx86.jpg", resultImage);

    //----------------------------------------------

    startTime = std::chrono::high_resolution_clock::now();


    // FOURTH FILTER


    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    // Apply the filter with the specified number of iterations
    for (int iteration = 0; iteration < SINGLE_ITERATION; iteration++) {
        int row = 1, col = 1, k_row = 0, k_col = 0, sum = 0;

        __asm {
            mov row, 1
            r_loop_4:
            mov edx, height
                sub edx, 1
                cmp row, edx
                jnb end_r_loop_4

                mov col, 1
                c_loop_4:
            mov edx, width
                sub edx, 1
                cmp col, edx
                jnb end_c_loop_4

                mov sum, 0
                mov k_row, 0
                kr_loop_4:
            cmp k_row, 3
                jnb end_kr_loop_4

                mov k_col, 0
                kc_loop_4 :
                cmp k_col, 3
                jnb end_kc_loop_4

                // Calculate 'resultImage' row offset
                mov edx, row    // offset = curr_row
                add edx, k_row  // offset += curr_kernelRow
                sub edx, 1      // offset -= 1
                // Get row address
                mov esi, resultImage
                mov ebx, [esi + 4 * edx]
                mov esi, ebx

                // Calculate 'resultImage' column offset
                mov edx, col    // offset = curr_column
                add edx, k_col  // offset += curr_kernelColumn
                sub edx, 1      // offset -= 1
                // Get resultImage[i + m -1][j + n -1]
                mov edx, [esi + 4 * edx]

                // Get kernel[m][n]
                mov ecx, k_row      // edx <- current_kernelRow
                imul ecx, 3         // multiply row number by the size of each row
                add ecx, k_col      // add column number
                lea esi, kernelData4 // get kernelData address
                mov ecx, [esi + 4 * ecx]    // retrieve element
                imul edx, ecx
                add sum, edx

                add k_col, 1
                jmp kc_loop_4
                end_kc_loop_4 :

            add k_row, 1
                jmp kr_loop_4
                end_kr_loop_4 :

            // Store calculated value in temporary result matrix
            mov edi, temp
                mov edx, row
                mov edi, [edi + 4 * edx]    // edi <- temp[i] (row address)

                mov eax, sum
                mov edx, 0
                mov ecx, 9
                cdq
                idiv ecx                    // eax <- sum/9

                mov edx, col
                mov[edi + 4 * edx], eax     // temp[i][j] <- sum/9

                add col, 1
                jmp c_loop_4
                end_c_loop_4 :

            add row, 1
                jmp r_loop_4
                end_r_loop_4 :
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }



    endTime = std::chrono::high_resolution_clock::now();
    totalEndTime = std::chrono::high_resolution_clock::now();

    elapsedTime = endTime - startTime;
    std::cout << "\tFourth filter time: " << elapsedTime.count() << " seconds" << std::endl;

    totalElapsedTime = totalEndTime - totalStartTime;
    std::cout << "\tTOTAL TIME: " << totalElapsedTime.count() << " seconds" << std::endl;

    // Save the embossed image
    saveImage("imagen_repujadax86.jpg", resultImage);


    
    // ---------- SSE BENCHMARK ----------

    std::cout << std::endl << "SSE BENCHMARK" << std::endl;

    totalStartTime = std::chrono::high_resolution_clock::now();
    startTime = std::chrono::high_resolution_clock::now();


    // FIRST FILTER


    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    for (int iteration = 0; iteration < BLUR_ITERATIONS; iteration++) {
        int sum = 0, kernelRow = 0, row = 1, col = 1, iterableHeight = height - 1, iterableWidth = width - 1;

        __asm {
            mov ecx, 9
            movups xmm2, KERNEL_MASK_SSE
            mov esi, temp

            row_loop_1 :
            mov edx, iterableHeight
                cmp row, edx // compare if row < height - 1
                jnb end_row_loop_1

                mov col, 1
                col_loop_1 :
                mov sum, 0 // clear the sum for each element
                mov edx, iterableWidth
                cmp col, edx // compare if col < width - 1
                jnb end_col_loop_1

                // save current row and column in the stack
                push row
                push col

                // move to the first element of the 3x3 submatrix
                sub row, 1
                sub col, 1

                mov kernelRow, 0
                // in each iteration a row of the 3x3 submatrix is multiplied by
                // the corresponding row of the kernel matrix
                kernel_loop_1:
            cmp kernelRow, 3
                jnb end_kernel_loop_1

                // access the image
                mov edx, row
                add edx, kernelRow
                mov edi, resultImage
                mov ebx, [edi + 4 * edx] // get row address
                mov edx, col
                movups xmm0, [ebx + 4 * edx] // move 4 integers to xmm0
                cvtdq2ps xmm0, xmm0 // convert to float (required for multiplication)

                // access the kernel matrix
                mov edx, kernelRow
                imul edx, 3 // multiply row number by the size of each row
                lea edi, kernelData1
                movups xmm1, [edi + 4 * edx] // move 4 integers to xmm0
                cvtdq2ps xmm1, xmm1 // convert to float (required for multiplication)
                andps xmm1, xmm2 // apply the mask to set to 0 the bits 96:127 (we don't want that element)

                // multiply both rows
                mulps xmm0, xmm1

                // perform the addition of multiplication results
                haddps xmm0, xmm0
                haddps xmm0, xmm0
                cvtss2si eax, xmm0 // convert the sum to integer and store it in eax
                add sum, eax

                add kernelRow, 1
                jmp kernel_loop_1

                end_kernel_loop_1 :
            mov eax, sum
                cdq // sign extension
                idiv ecx // eax <- sum / 9

                // retrieve row and col from the stack to store the result in the corresponding element
                pop col
                pop row
                mov edx, row
                mov ebx, [esi + 4 * edx]
                mov edx, col
                mov[ebx + 4 * edx], eax // temp[i][j] <- sum / 9

                add col, 1
                jmp col_loop_1
                end_col_loop_1 :

            add row, 1
                jmp row_loop_1
                end_row_loop_1 :
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }


    endTime = std::chrono::high_resolution_clock::now();
    elapsedTime = endTime - startTime;

    std::cout << "\tFirst filter time: " << elapsedTime.count() << " seconds" << std::endl;

    // Save the blurred image
    saveImage("imagen_desenfocadaSSE.jpg", resultImage);

    //-------------------------------------------

    startTime = std::chrono::high_resolution_clock::now();

    // SECOND FILTER


    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    for (int iteration = 0; iteration < SMOOTHING_ITERATIONS; iteration++) {
        int sum = 0, kernelRow = 0, row = 1, col = 1, iterableHeight = height - 1, iterableWidth = width - 1;

        __asm {
            mov ecx, 9
            movups xmm2, KERNEL_MASK_SSE
            mov esi, temp

            row_loop_2 :
            mov edx, iterableHeight
                cmp row, edx // compare if row < height - 1
                jnb end_row_loop_2

                mov col, 1
                col_loop_2 :
                mov sum, 0 // clear the sum for each element
                mov edx, iterableWidth
                cmp col, edx // compare if col < width - 1
                jnb end_col_loop_2

                // save current row and column in the stack
                push row
                push col

                // move to the first element of the 3x3 submatrix
                sub row, 1
                sub col, 1

                mov kernelRow, 0
                // in each iteration a row of the 3x3 submatrix is multiplied by
                // the corresponding row of the kernel matrix
                kernel_loop_2:
            cmp kernelRow, 3
                jnb end_kernel_loop_2

                // access the image
                mov edx, row
                add edx, kernelRow
                mov edi, resultImage
                mov ebx, [edi + 4 * edx] // get row address
                mov edx, col
                movups xmm0, [ebx + 4 * edx] // move 4 integers to xmm0
                cvtdq2ps xmm0, xmm0 // convert to float (required for multiplication)

                // access the kernel matrix
                mov edx, kernelRow
                imul edx, 3 // multiply row number by the size of each row
                lea edi, kernelData2
                movups xmm1, [edi + 4 * edx] // move 4 integers to xmm0
                cvtdq2ps xmm1, xmm1 // convert to float (required for multiplication)
                andps xmm1, xmm2 // apply the mask to set to 0 the bits 96:127 (we don't want that element)

                // multiply both rows
                mulps xmm0, xmm1

                // perform the addition of multiplication results
                haddps xmm0, xmm0
                haddps xmm0, xmm0
                cvtss2si eax, xmm0 // convert the sum to integer and store it in eax
                add sum, eax

                add kernelRow, 1
                jmp kernel_loop_2

                end_kernel_loop_2 :
            mov eax, sum
                cdq // sign extension
                idiv ecx // eax <- sum / 9

                // retrieve row and col from the stack to store the result in the corresponding element
                pop col
                pop row
                mov edx, row
                mov ebx, [esi + 4 * edx]
                mov edx, col
                mov[ebx + 4 * edx], eax // temp[i][j] <- sum / 9

                add col, 1
                jmp col_loop_2
                end_col_loop_2 :

            add row, 1
                jmp row_loop_2
                end_row_loop_2 :
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }



    endTime = std::chrono::high_resolution_clock::now();
    elapsedTime = endTime - startTime;

    std::cout << "\tSecond filter time: " << elapsedTime.count() << " seconds" << std::endl;

    // Save the smoothed image
    saveImage("imagen_suavizadoSSE.jpg", resultImage);

    //----------------------------------------------

    startTime = std::chrono::high_resolution_clock::now();



    // THIRD FILTER


    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    for (int iteration = 0; iteration < SINGLE_ITERATION; iteration++) {
        int sum = 0, kernelRow = 0, row = 1, col = 1, iterableHeight = height - 1, iterableWidth = width - 1;

        __asm {
            mov ecx, 9
            movups xmm2, KERNEL_MASK_SSE
            mov esi, temp

            row_loop_3 :
            mov edx, iterableHeight
                cmp row, edx // compare if row < height - 1
                jnb end_row_loop_3

                mov col, 1
                col_loop_3 :
                mov sum, 0 // clear the sum for each element
                mov edx, iterableWidth
                cmp col, edx // compare if col < width - 1
                jnb end_col_loop_3

                // save current row and column in the stack
                push row
                push col

                // move to the first element of the 3x3 submatrix
                sub row, 1
                sub col, 1

                mov kernelRow, 0
                // in each iteration a row of the 3x3 submatrix is multiplied by
                // the corresponding row of the kernel matrix
                kernel_loop_3:
            cmp kernelRow, 3
                jnb end_kernel_loop_3

                // access the image
                mov edx, row
                add edx, kernelRow
                mov edi, resultImage
                mov ebx, [edi + 4 * edx] // get row address
                mov edx, col
                movups xmm0, [ebx + 4 * edx] // move 4 integers to xmm0
                cvtdq2ps xmm0, xmm0 // convert to float (required for multiplication)

                // access the kernel matrix
                mov edx, kernelRow
                imul edx, 3 // multiply row number by the size of each row
                lea edi, kernelData3
                movups xmm1, [edi + 4 * edx] // move 4 integers to xmm0
                cvtdq2ps xmm1, xmm1 // convert to float (required for multiplication)
                andps xmm1, xmm2 // apply the mask to set to 0 the bits 96:127 (we don't want that element)

                // multiply both rows
                mulps xmm0, xmm1

                // perform the addition of multiplication results
                haddps xmm0, xmm0
                haddps xmm0, xmm0
                cvtss2si eax, xmm0 // convert the sum to integer and store it in eax
                add sum, eax

                add kernelRow, 1
                jmp kernel_loop_3

                end_kernel_loop_3 :
            mov eax, sum
                cdq // sign extension
                idiv ecx // eax <- sum / 9

                // retrieve row and col from the stack to store the result in the corresponding element
                pop col
                pop row
                mov edx, row
                mov ebx, [esi + 4 * edx]
                mov edx, col
                mov[ebx + 4 * edx], eax // temp[i][j] <- sum / 9

                add col, 1
                jmp col_loop_3
                end_col_loop_3 :

            add row, 1
                jmp row_loop_3
                end_row_loop_3 :
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }



    endTime = std::chrono::high_resolution_clock::now();
    elapsedTime = endTime - startTime;

    std::cout << "\tThird filter time: " << elapsedTime.count() << " seconds" << std::endl;

    // Save the image with edge detection
    saveImage("imagen_deteccionBordesSSE.jpg", resultImage);

    //----------------------------------------------

    startTime = std::chrono::high_resolution_clock::now();


    // FOURTH FILTER


    // Copy the original image to the result
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            resultImage[i][j] = originalImage[i][j];
            temp[i][j] = originalImage[i][j];
        }
    }

    for (int iteration = 0; iteration < SINGLE_ITERATION; iteration++) {
        int sum = 0, kernelRow = 0, row = 1, col = 1, iterableHeight = height - 1, iterableWidth = width - 1;

        __asm {
            mov ecx, 9
            movups xmm2, KERNEL_MASK_SSE
            mov esi, temp

            row_loop_4 :
            mov edx, iterableHeight
                cmp row, edx // compare if row < height - 1
                jnb end_row_loop_4

                mov col, 1
                col_loop_4 :
                mov sum, 0 // clear the sum for each element
                mov edx, iterableWidth
                cmp col, edx // compare if col < width - 1
                jnb end_col_loop_4

                // save current row and column in the stack
                push row
                push col

                // move to the first element of the 3x3 submatrix
                sub row, 1
                sub col, 1

                mov kernelRow, 0
                // in each iteration a row of the 3x3 submatrix is multiplied by
                // the corresponding row of the kernel matrix
                kernel_loop_4:
            cmp kernelRow, 3
                jnb end_kernel_loop_4

                // access the image
                mov edx, row
                add edx, kernelRow
                mov edi, resultImage
                mov ebx, [edi + 4 * edx] // get row address
                mov edx, col
                movups xmm0, [ebx + 4 * edx] // move 4 integers to xmm0
                cvtdq2ps xmm0, xmm0 // convert to float (required for multiplication)

                // access the kernel matrix
                mov edx, kernelRow
                imul edx, 3 // multiply row number by the size of each row
                lea edi, kernelData4
                movups xmm1, [edi + 4 * edx] // move 4 integers to xmm0
                cvtdq2ps xmm1, xmm1 // convert to float (required for multiplication)
                andps xmm1, xmm2 // apply the mask to set to 0 the bits 96:127 (we don't want that element)

                // multiply both rows
                mulps xmm0, xmm1

                // perform the addition of multiplication results
                haddps xmm0, xmm0
                haddps xmm0, xmm0
                cvtss2si eax, xmm0 // convert the sum to integer and store it in eax
                add sum, eax

                add kernelRow, 1
                jmp kernel_loop_4

                end_kernel_loop_4 :
            mov eax, sum
                cdq // sign extension
                idiv ecx // eax <- sum / 9

                // retrieve row and col from the stack to store the result in the corresponding element
                pop col
                pop row
                mov edx, row
                mov ebx, [esi + 4 * edx]
                mov edx, col
                mov[ebx + 4 * edx], eax // temp[i][j] <- sum / 9

                add col, 1
                jmp col_loop_4
                end_col_loop_4 :

            add row, 1
                jmp row_loop_4
                end_row_loop_4 :
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                resultImage[i][j] = temp[i][j];
            }
        }
    }



    endTime = std::chrono::high_resolution_clock::now();
    totalEndTime = std::chrono::high_resolution_clock::now();

    elapsedTime = endTime - startTime;
    std::cout << "\tFourth filter time: " << elapsedTime.count() << " seconds" << std::endl;

    totalElapsedTime = totalEndTime - totalStartTime;
    std::cout << "\tTOTAL TIME: " << totalElapsedTime.count() << " seconds" << std::endl;

    // Save the embossed image
    saveImage("imagen_repujadaSSE.jpg", resultImage);

    

    
    // Free temporary matrix memory
    for (int i = 0; i < height; i++) {
        delete[] temp[i];
    }
    delete[] temp;

    // Free memory
    for (int i = 0; i < height; i++) {
        delete[] originalImage[i];
        delete[] resultImage[i];
    }
    delete[] originalImage;
    delete[] resultImage;
    

    return 0;
}