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
            }
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
    int** imagen_original = cargarImagen("imagen3.jpg");

    int** imagen_resultado = new int*[altura];
    for (int i = 0; i < altura; i++) {
        imagen_resultado[i] = new int[ancho];
    }

    // Primer fitro
    int iteraciones_totales = 40;
    int kernelData1[3][3] = { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} };

    auto start_time = std::chrono::high_resolution_clock::now();
    aplicarFiltro(imagen_original, imagen_resultado, iteraciones_totales, kernelData1);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    std::cout << "Tiempo para el primer filtro: " << elapsed_time.count() << " segundos" << std::endl;

    // Guardar la imagen desenfocada
    guardarImagen("imagen_desenfocada.jpg", imagen_resultado);

    //-------------------------------------------

    iteraciones_totales = 1;
    int kernelData2[3][3] = { {0, 1, 0}, {1, -4, 1}, {0, 1, 0} };

    start_time = std::chrono::high_resolution_clock::now();
    aplicarFiltro(imagen_original, imagen_resultado, iteraciones_totales, kernelData2);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = end_time - start_time;

    std::cout << "Tiempo para el segundo filtro: " << elapsed_time.count() << " segundos" << std::endl;

    // Guardar la imagen desenfocada
    guardarImagen("imagen_deteccionBordes.jpg", imagen_resultado);

    //----------------------------------------------

    iteraciones_totales = 1;
    int kernelData3 [3][3] = { {-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}};

    start_time = std::chrono::high_resolution_clock::now();
    aplicarFiltro(imagen_original, imagen_resultado, iteraciones_totales, kernelData3);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = end_time - start_time;

    std::cout << "Tiempo para el tercer filtro: " << elapsed_time.count() << " segundos" << std::endl;

    // Guardar la imagen desenfocada
    guardarImagen("imagen_repujada.jpg", imagen_resultado);

    // Liberar memoria
    for (int i = 0; i < altura; i++) {
        delete[] imagen_original[i];
        delete[] imagen_resultado[i];
    }
    delete[] imagen_original;
    delete[] imagen_resultado;

    return 0;
}