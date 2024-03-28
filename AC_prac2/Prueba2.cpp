#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

int altura;
int ancho;

int** cargarImagen(const std::string& ruta) {
    cv::Mat imagen = cv::imread(ruta, cv::IMREAD_GRAYSCALE);  // Convertir a blanco y negro
    if (imagen.empty()) {
        std::cerr << "No se pudo cargar la imagen" << std::endl;
    }

    altura = imagen.rows;
    ancho = imagen.cols;
    int** imagen_array = new int* [altura];
    for (int i = 0; i < altura; i++) {
        imagen_array[i] = new int[ancho];
        for (int j = 0; j < ancho; j++) {
            imagen_array[i][j] = static_cast<int>(imagen.at<uchar>(i, j));
        }
    }
    return imagen_array;
}

void aplicarFiltro(int** imagen_original, int** resultado, int iteraciones_totales, int kernelData[3][3]) {
    int** temp = new int* [altura];

    // Copiar la imagen original al resultado
    for (int i = 0; i < altura; i++) {
        temp[i] = new int[ancho];
        for (int j = 0; j < ancho; j++) {
            resultado[i][j] = imagen_original[i][j];
            temp[i][j] = imagen_original[i][j];
        }
    }

    // Aplicar el filtro con el número de iteraciones especificado
    for (int iteracion = 0; iteracion < iteraciones_totales; iteracion++) {


        for (int i = 1; i < altura - 1; i++) {
            for (int j = 1; j < ancho - 1; j++) {
                int suma = 0;
                for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                        suma += resultado[i + m - 1][j + n - 1] * kernelData[m][n];
                    }
                }
                temp[i][j] = (suma / 9);  // Normalizar el resultado
            }
        }

        // Copiar el resultado temporal al resultado final
        for (int i = 0; i < altura; i++) {
            for (int j = 0; j < ancho; j++) {
                resultado[i][j] = temp[i][j];
                // std::cout << resultado[i][j] << " ";
            }
            // std::cout << std::endl;
        }
    }

    // Liberar memoria de la matriz temporal
    for (int i = 0; i < altura; i++) {
        delete[] temp[i];
    }
    delete[] temp;
}

void aplicarFiltro_x86(int** imagen_original, int** resultado, int iteraciones_totales, int kernelData[3][3]) {
    int** temp = new int* [altura];

    // Copiar la imagen original al resultado
    for (int i = 0; i < altura; i++) {
        temp[i] = new int[ancho];
        for (int j = 0; j < ancho; j++) {
            resultado[i][j] = imagen_original[i][j];
            temp[i][j] = imagen_original[i][j];
        }
    }

    // Aplicar el filtro con el número de iteraciones especificado
    for (int iteracion = 0; iteracion < iteraciones_totales; iteracion++) {
        int row = 1, col = 1, k_row = 0, k_col = 0, sum = 0;

        __asm {
            mov row, 1
            r_loop:
            mov edx, altura
                sub edx, 1
                cmp row, edx
                jnb end_r_loop

                mov col, 1
                c_loop:
            mov edx, ancho
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

                // Calculate 'resultado' row offset
                mov edx, row    // offset = curr_row
                add edx, k_row  // offset += curr_kernelRow
                sub edx, 1      // offset -= 1
                // Get row address
                mov esi, resultado
                mov ebx, [esi + 4 * edx]
                mov esi, ebx

                // Calculate 'resultado' column offset
                mov edx, col    // offset = curr_column
                add edx, k_col  // offset += curr_kernelColumn
                sub edx, 1      // offset -= 1
                // Get resultado[i + m -1][j + n -1]
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

            // Store calculated value in temporal result matrix
            mov edi, temp
                mov edx, row
                mov edi, [edi + 4 * edx]    // edi <- temp[i] (row address)

                mov eax, sum
                mov edx, 0
                mov ecx, 9
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

        for (int i = 0; i < altura; i++) {
            for (int j = 0; j < ancho; j++) {
                resultado[i][j] = temp[i][j];
                // std::cout << resultado[i][j] << " ";
            }
            // std::cout << std::endl;
        }
    }

    // Liberar memoria de la matriz temporal
    for (int i = 0; i < altura; i++) {
        delete[] temp[i];
    }
    delete[] temp;
}

void guardarImagen(const std::string& ruta, int** imagen) {
    // Crear la imagen desenfocada
    cv::Mat imagen_mat(altura, ancho, CV_8U);
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < ancho; j++) {
            imagen_mat.at<uchar>(i, j) = static_cast<uchar>(imagen[i][j]);
        }
    }

    cv::imwrite(ruta, imagen_mat);
}


int main() {
    int** imagen_original = cargarImagen("imagen.jpg");

    int** imagen_resultado = new int*[altura];
    for (int i = 0; i < altura; i++) {
        imagen_resultado[i] = new int[ancho];
    }


    int iteraciones_totales = 40;
    int kernelData1[3][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };

    auto start_totalTime = std::chrono::high_resolution_clock::now();
    auto start_time = std::chrono::high_resolution_clock::now();
    
    /* -------------------------------------------------------------------------------

    int** temp = new int* [altura];

    // Copiar la imagen original al resultado
    for (int i = 0; i < altura; i++) {
        temp[i] = new int[ancho];
        for (int j = 0; j < ancho; j++) {
            imagen_resultado[i][j] = imagen_original[i][j];
            temp[i][j] = imagen_original[i][j];
        }
    }

    // Aplicar el filtro con el número de iteraciones especificado
    for (int iteracion = 0; iteracion < iteraciones_totales; iteracion++) {
        int row = 1, col = 1, k_row = 0, k_col = 0, sum = 0;

        __asm {
            mov row, 1
            r_loop:
            mov edx, altura
                sub edx, 1
                cmp row, edx
                jnb end_r_loop

                mov col, 1
                c_loop:
            mov edx, ancho
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

                // Calculate 'resultado' row offset
                mov edx, row    // offset = curr_row
                add edx, k_row  // offset += curr_kernelRow
                sub edx, 1      // offset -= 1
                // Get row address
                mov esi, imagen_resultado
                mov ebx, [esi + 4 * edx]
                mov esi, ebx

                // Calculate 'resultado' column offset
                mov edx, col    // offset = curr_column
                add edx, k_col  // offset += curr_kernelColumn
                sub edx, 1      // offset -= 1
                // Get resultado[i + m -1][j + n -1]
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
                jmp kc_loop
                end_kc_loop :

            add k_row, 1
                jmp kr_loop
                end_kr_loop :

            // Store calculated value in temporal result matrix
            mov edi, temp
                mov edx, row
                mov edi, [edi + 4 * edx]    // edi <- temp[i] (row address)

                mov eax, sum
                mov edx, 0
                mov ecx, 9
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

        for (int i = 0; i < altura; i++) {
            for (int j = 0; j < ancho; j++) {
                imagen_resultado[i][j] = temp[i][j];
            }
        }
    }

    // Liberar memoria de la matriz temporal
    for (int i = 0; i < altura; i++) {
        delete[] temp[i];
    }
    delete[] temp;

    // -------------------------------------------------------------------------------
    */

    aplicarFiltro(imagen_original, imagen_resultado, iteraciones_totales, kernelData1);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    std::cout << "Tiempo para el primer filtro: " << elapsed_time.count() << " segundos" << std::endl;

    // Guardar la imagen desenfocada
    guardarImagen("imagen_desenfocada.jpg", imagen_resultado);

    /*
    //-------------------------------------------

    iteraciones_totales = 1;
    int kernelData2[3][3] = { {1, 1, 1}, {1, 2, 1}, {1, 1, 1} };

    start_time = std::chrono::high_resolution_clock::now();
    aplicarFiltro_x86(imagen_original, imagen_resultado, iteraciones_totales, kernelData2);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = end_time - start_time;

    std::cout << "Tiempo para el segundo filtro: " << elapsed_time.count() << " segundos" << std::endl;

    // Guardar la imagen desenfocada
    guardarImagen("imagen_suavizado.jpg", imagen_resultado);

    //----------------------------------------------

    iteraciones_totales = 1;
    int kernelData3[3][3] = { {0, 1, 0}, {1, -4, 1}, {0, 1, 0} };

    start_time = std::chrono::high_resolution_clock::now();
    aplicarFiltro_x86(imagen_original, imagen_resultado, iteraciones_totales, kernelData3);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = end_time - start_time;

    std::cout << "Tiempo para el tercer filtro: " << elapsed_time.count() << " segundos" << std::endl;

    // Guardar la imagen desenfocada
    guardarImagen("imagen_deteccionBordes.jpg", imagen_resultado);

    //----------------------------------------------

    iteraciones_totales = 1;
    int kernelData4 [3][3] = { {-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}};

    start_time = std::chrono::high_resolution_clock::now();
    aplicarFiltro_x86(imagen_original, imagen_resultado, iteraciones_totales, kernelData4);
    end_time = std::chrono::high_resolution_clock::now();
    auto end_totalTime = std::chrono::high_resolution_clock::now();

    elapsed_time = end_time - start_time;
    std::cout << "Tiempo para el cuarto filtro: " << elapsed_time.count() << " segundos" << std::endl;

    std::chrono::duration<double> elapsed_totalTime = end_totalTime - start_totalTime;
    std::cout << "Tiempo total: " << elapsed_totalTime.count() << " segundos" << std::endl;

    // Guardar la imagen desenfocada
    guardarImagen("imagen_repujada.jpg", imagen_resultado);
    */



    // Liberar memoria
    for (int i = 0; i < altura; i++) {
        delete[] imagen_original[i];
        delete[] imagen_resultado[i];
    }
    delete[] imagen_original;
    delete[] imagen_resultado;

    return 0;
}