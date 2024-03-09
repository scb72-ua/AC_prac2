#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

cv::Mat cargarImagen(const std::string& ruta) {
    cv::Mat imagen = cv::imread(ruta, cv::IMREAD_GRAYSCALE);  // Convertir a blanco y negro
    if (imagen.empty()) {
        std::cerr << "No se pudo cargar la imagen" << std::endl;
    }
    return imagen;
}

void aplicarFiltroDesenfoque(int** imagen_original, int altura, int ancho, int iteraciones_totales, float kernelData[3][3], int** resultado) {
    // Copiar la imagen original al resultado
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < ancho; j++) {
            resultado[i][j] = imagen_original[i][j];
        }
    }

    int** temp = new int* [altura];
    for (int i = 0; i < altura; i++) {
        temp[i] = new int[ancho];
    }

    // Aplicar el filtro de desenfoque con el número de iteraciones especificado
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
            }
        }


    }
    // Liberar memoria de la matriz temporal
    for (int i = 0; i < altura; i++) {
        delete[] temp[i];
    }
    delete[] temp;
}


void guardarImagen(const std::string& ruta, const cv::Mat& imagen) {
    cv::imwrite(ruta, imagen);
}

int main() {
    // Cargar la imagen
    cv::Mat imagen = cargarImagen("imagen3.jpg");

    // Crear un array bidimensional para la imagen

        // Crear un array bidimensional para la imagen
    int altura = imagen.rows;
    int ancho = imagen.cols;
    int** imagen_array = new int* [altura];
    for (int i = 0; i < altura; i++) {
        imagen_array[i] = new int[ancho];
    }

    // Copiar la imagen a un array
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < ancho; j++) {
            imagen_array[i][j] = imagen.at<uchar>(i, j);
        }
    }

    int** resultado = new int* [altura];
    for (int i = 0; i < altura; i++) {
        resultado[i] = new int[ancho];
    }

    

    int iteraciones_totales = 40;
    float kernelData2[3][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };

    auto start_time1 = std::chrono::high_resolution_clock::now();
    aplicarFiltroDesenfoque(imagen_array, altura, ancho, iteraciones_totales, kernelData2, resultado);
    auto end_time1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time1 = end_time1 - start_time1;

    std::cout << "Tiempo para el primer filtro: " << elapsed_time1.count() << " segundos" << std::endl;


    // Crear la imagen desenfocada
    cv::Mat resultado_mat2(altura, ancho, CV_8U);
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < ancho; j++) {
            resultado_mat2.at<uchar>(i, j) = static_cast<uchar>(resultado[i][j]);
        }
    }

    // Guardar la imagen desenfocada
    guardarImagen("imagen_desenfocada.jpg", resultado_mat2);

    // Liberar memoria
    for (int i = 0; i < altura; i++) {
        delete[] imagen_array[i];
        delete[] resultado[i];
    }
    delete[] imagen_array;
    delete[] resultado;

    //-------------------------------------------

    altura = imagen.rows;
    ancho = imagen.cols;
    imagen_array = new int* [altura];
    for (int i = 0; i < altura; i++) {
        imagen_array[i] = new int[ancho];
    }

    // Copiar la imagen a un array
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < ancho; j++) {
            imagen_array[i][j] = imagen.at<uchar>(i, j);
        }
    }

    resultado = new int* [altura];
    for (int i = 0; i < altura; i++) {
        resultado[i] = new int[ancho];
    }

    iteraciones_totales = 1;
    float kernelData[3][3] = { {0, 1, 0}, {1, -4, 1}, {0, 1, 0} };

    start_time1 = std::chrono::high_resolution_clock::now();
    aplicarFiltroDesenfoque(imagen_array, altura, ancho, iteraciones_totales, kernelData, resultado);
    end_time1 = std::chrono::high_resolution_clock::now();
    elapsed_time1 = end_time1 - start_time1;

    std::cout << "Tiempo para el segundo filtro: " << elapsed_time1.count() << " segundos" << std::endl;


    // Crear la imagen desenfocada
    cv::Mat resultado_mat(altura, ancho, CV_8U);
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < ancho; j++) {
            resultado_mat.at<uchar>(i, j) = static_cast<uchar>(resultado[i][j]);
        }
    }

    // Guardar la imagen desenfocada
    guardarImagen("imagen_deteccionBordes.jpg", resultado_mat);

    // Liberar memoria
    for (int i = 0; i < altura; i++) {
        delete[] imagen_array[i];
        delete[] resultado[i];
    }
    delete[] imagen_array;
    delete[] resultado;

    //----------------------------------------------
        // Crear un array bidimensional para la imagen
    altura = imagen.rows;
    ancho = imagen.cols;
    imagen_array = new int* [altura];
    for (int i = 0; i < altura; i++) {
        imagen_array[i] = new int[ancho];
    }

    // Copiar la imagen a un array
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < ancho; j++) {
            imagen_array[i][j] = imagen.at<uchar>(i, j);
        }
    }

    resultado = new int* [altura];
    for (int i = 0; i < altura; i++) {
        resultado[i] = new int[ancho];
    }

    iteraciones_totales = 1;
    float kernelData3 [3][3] = { {-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}};

    start_time1 = std::chrono::high_resolution_clock::now();
    aplicarFiltroDesenfoque(imagen_array, altura, ancho, iteraciones_totales, kernelData3, resultado);
    end_time1 = std::chrono::high_resolution_clock::now();
    elapsed_time1 = end_time1 - start_time1;

    std::cout << "Tiempo para el tercer filtro: " << elapsed_time1.count() << " segundos" << std::endl;


    // Crear la imagen desenfocada
    cv::Mat resultado_mat3 (altura, ancho, CV_8U);
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < ancho; j++) {
            resultado_mat3.at<uchar>(i, j) = static_cast<uchar>(resultado[i][j]);
        }
    }

    // Guardar la imagen desenfocada
    guardarImagen("imagen_repujada.jpg", resultado_mat3);

    // Liberar memoria
    for (int i = 0; i < altura; i++) {
        delete[] imagen_array[i];
        delete[] resultado[i];
    }
    delete[] imagen_array;
    delete[] resultado;

    return 0;
}
