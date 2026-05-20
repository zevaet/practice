#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

// Для OpenBLAS в MSYS2:
// g++ -std=c++17 -O2 -fopenmp lab2.cpp -lopenblas -o lab2.exe
#include <openblas/cblas.h>

using namespace std;

const int DEFAULT_N = 2048;

int pos(int i, int j, int n) {
    return i * n + j;
}

void fillMatrix(vector<double>& matrix, int n, int seed) {
    mt19937 randomGenerator(seed);
    uniform_real_distribution<double> randomNumber(-1.0, 1.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[pos(i, j, n)] = randomNumber(randomGenerator);
        }
    }
}

void clearMatrix(vector<double>& matrix) {
    fill(matrix.begin(), matrix.end(), 0.0);
}

double getSeconds(chrono::high_resolution_clock::time_point start,
                  chrono::high_resolution_clock::time_point finish) {
    return chrono::duration<double>(finish - start).count();
}

double getMflops(long double operations, double seconds) {
    return (double)(operations / seconds * 0.000001L);
}

double sumAvx(__m256d value) {
    double temp[4];
    _mm256_storeu_pd(temp, value);
    return temp[0] + temp[1] + temp[2] + temp[3];
}

void multiplySimple(const vector<double>& a,
                    const vector<double>& b,
                    vector<double>& c,
                    int n) {
    clearMatrix(c);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            for (int k = 0; k < n; k++) {
                sum += a[pos(i, k, n)] * b[pos(k, j, n)];
            }

            c[pos(i, j, n)] = sum;
        }
    }
}

void multiplyBlas(const vector<double>& a,
                  const vector<double>& b,
                  vector<double>& c,
                  int n) {
    clearMatrix(c);

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                n,
                n,
                n,
                1.0,
                a.data(),
                n,
                b.data(),
                n,
                0.0,
                c.data(),
                n);
}

void multiplyAvxTransposed(const vector<double>& a,
                           const vector<double>& b,
                           vector<double>& c,
                           int n) {
    clearMatrix(c);

    vector<double> transposedB(n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            transposedB[pos(j, i, n)] = b[pos(i, j, n)];
        }
    }

    // После транспонирования элементы строки A и строки transposedB читаются подряд.
    // AVX обрабатывает сразу 4 числа типа double за одну операцию.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; i++) {
        const double* rowA = &a[pos(i, 0, n)];

        for (int j = 0; j < n; j += 8) {
            if (j + 7 < n) {
                const double* rowB0 = &transposedB[pos(j, 0, n)];
                const double* rowB1 = &transposedB[pos(j + 1, 0, n)];
                const double* rowB2 = &transposedB[pos(j + 2, 0, n)];
                const double* rowB3 = &transposedB[pos(j + 3, 0, n)];
                const double* rowB4 = &transposedB[pos(j + 4, 0, n)];
                const double* rowB5 = &transposedB[pos(j + 5, 0, n)];
                const double* rowB6 = &transposedB[pos(j + 6, 0, n)];
                const double* rowB7 = &transposedB[pos(j + 7, 0, n)];

                __m256d sum0 = _mm256_setzero_pd();
                __m256d sum1 = _mm256_setzero_pd();
                __m256d sum2 = _mm256_setzero_pd();
                __m256d sum3 = _mm256_setzero_pd();
                __m256d sum4 = _mm256_setzero_pd();
                __m256d sum5 = _mm256_setzero_pd();
                __m256d sum6 = _mm256_setzero_pd();
                __m256d sum7 = _mm256_setzero_pd();
                int k = 0;

                for (; k + 3 < n; k += 4) {
                    __m256d va = _mm256_loadu_pd(rowA + k);
                    sum0 = _mm256_fmadd_pd(va, _mm256_loadu_pd(rowB0 + k), sum0);
                    sum1 = _mm256_fmadd_pd(va, _mm256_loadu_pd(rowB1 + k), sum1);
                    sum2 = _mm256_fmadd_pd(va, _mm256_loadu_pd(rowB2 + k), sum2);
                    sum3 = _mm256_fmadd_pd(va, _mm256_loadu_pd(rowB3 + k), sum3);
                    sum4 = _mm256_fmadd_pd(va, _mm256_loadu_pd(rowB4 + k), sum4);
                    sum5 = _mm256_fmadd_pd(va, _mm256_loadu_pd(rowB5 + k), sum5);
                    sum6 = _mm256_fmadd_pd(va, _mm256_loadu_pd(rowB6 + k), sum6);
                    sum7 = _mm256_fmadd_pd(va, _mm256_loadu_pd(rowB7 + k), sum7);
                }

                double answer0 = sumAvx(sum0);
                double answer1 = sumAvx(sum1);
                double answer2 = sumAvx(sum2);
                double answer3 = sumAvx(sum3);
                double answer4 = sumAvx(sum4);
                double answer5 = sumAvx(sum5);
                double answer6 = sumAvx(sum6);
                double answer7 = sumAvx(sum7);

                for (; k < n; k++) {
                    answer0 += rowA[k] * rowB0[k];
                    answer1 += rowA[k] * rowB1[k];
                    answer2 += rowA[k] * rowB2[k];
                    answer3 += rowA[k] * rowB3[k];
                    answer4 += rowA[k] * rowB4[k];
                    answer5 += rowA[k] * rowB5[k];
                    answer6 += rowA[k] * rowB6[k];
                    answer7 += rowA[k] * rowB7[k];
                }

                c[pos(i, j, n)] = answer0;
                c[pos(i, j + 1, n)] = answer1;
                c[pos(i, j + 2, n)] = answer2;
                c[pos(i, j + 3, n)] = answer3;
                c[pos(i, j + 4, n)] = answer4;
                c[pos(i, j + 5, n)] = answer5;
                c[pos(i, j + 6, n)] = answer6;
                c[pos(i, j + 7, n)] = answer7;
            } else {
                for (int column = j; column < n; column++) {
                    const double* rowB = &transposedB[pos(column, 0, n)];
                    __m256d vectorSum = _mm256_setzero_pd();
                    int k = 0;

                    for (; k + 3 < n; k += 4) {
                        __m256d va = _mm256_loadu_pd(rowA + k);
                        __m256d vb = _mm256_loadu_pd(rowB + k);
                        vectorSum = _mm256_fmadd_pd(va, vb, vectorSum);
                    }

                    double sum = sumAvx(vectorSum);

                    for (; k < n; k++) {
                        sum += rowA[k] * rowB[k];
                    }

                    c[pos(i, column, n)] = sum;
                }
            }
        }
    }
}

void copyBlock(const vector<double>& source,
               vector<double>& block,
               int n,
               int row,
               int col,
               int blockSize) {
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < blockSize; j++) {
            block[pos(i, j, blockSize)] = source[pos(row + i, col + j, n)];
        }
    }
}

void addMatrix(const vector<double>& a,
               const vector<double>& b,
               vector<double>& result,
               int size,
               double sign) {
    for (int i = 0; i < size * size; i++) {
        result[i] = a[i] + sign * b[i];
    }
}

void multiplyStrassenOneLevel(const vector<double>& a,
                              const vector<double>& b,
                              vector<double>& c,
                              int n) {
    int m = n / 2;
    int smallSize = m * m;

    vector<double> a11(smallSize), a12(smallSize), a21(smallSize), a22(smallSize);
    vector<double> b11(smallSize), b12(smallSize), b21(smallSize), b22(smallSize);
    vector<double> m1(smallSize), m2(smallSize), m3(smallSize), m4(smallSize);
    vector<double> m5(smallSize), m6(smallSize), m7(smallSize);
    vector<double> tempA(smallSize), tempB(smallSize);

    copyBlock(a, a11, n, 0, 0, m);
    copyBlock(a, a12, n, 0, m, m);
    copyBlock(a, a21, n, m, 0, m);
    copyBlock(a, a22, n, m, m, m);

    copyBlock(b, b11, n, 0, 0, m);
    copyBlock(b, b12, n, 0, m, m);
    copyBlock(b, b21, n, m, 0, m);
    copyBlock(b, b22, n, m, m, m);

    addMatrix(a11, a22, tempA, m, 1.0);
    addMatrix(b11, b22, tempB, m, 1.0);
    multiplyAvxTransposed(tempA, tempB, m1, m);

    addMatrix(a21, a22, tempA, m, 1.0);
    multiplyAvxTransposed(tempA, b11, m2, m);

    addMatrix(b12, b22, tempB, m, -1.0);
    multiplyAvxTransposed(a11, tempB, m3, m);

    addMatrix(b21, b11, tempB, m, -1.0);
    multiplyAvxTransposed(a22, tempB, m4, m);

    addMatrix(a11, a12, tempA, m, 1.0);
    multiplyAvxTransposed(tempA, b22, m5, m);

    addMatrix(a21, a11, tempA, m, -1.0);
    addMatrix(b11, b12, tempB, m, 1.0);
    multiplyAvxTransposed(tempA, tempB, m6, m);

    addMatrix(a12, a22, tempA, m, -1.0);
    addMatrix(b21, b22, tempB, m, 1.0);
    multiplyAvxTransposed(tempA, tempB, m7, m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            int smallIndex = pos(i, j, m);

            c[pos(i, j, n)] = m1[smallIndex] + m4[smallIndex] - m5[smallIndex] + m7[smallIndex];
            c[pos(i, j + m, n)] = m3[smallIndex] + m5[smallIndex];
            c[pos(i + m, j, n)] = m2[smallIndex] + m4[smallIndex];
            c[pos(i + m, j + m, n)] = m1[smallIndex] - m2[smallIndex] + m3[smallIndex] + m6[smallIndex];
        }
    }
}

void multiplyOptimized(const vector<double>& a,
                       const vector<double>& b,
                       vector<double>& c,
                       int n) {
    clearMatrix(c);

    if (n % 2 == 0 && n >= 512) {
        multiplyStrassenOneLevel(a, b, c, n);
    } else {
        multiplyAvxTransposed(a, b, c, n);
    }
}

double maxDifference(const vector<double>& a, const vector<double>& b) {
    double answer = 0.0;

    for (int i = 0; i < (int)a.size(); i++) {
        answer = max(answer, abs(a[i] - b[i]));
    }

    return answer;
}

void printResult(const string& name, double seconds, long double operations) {
    cout << name << ": ";
    cout << "time = " << setw(10) << fixed << setprecision(4) << seconds << " sec, ";
    cout << "p = " << setw(12) << fixed << setprecision(2) << getMflops(operations, seconds);
    cout << " MFLOPS\n";
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    cout << "ФИО: Кононенко Г. А.\n";
    cout << "Группа: 090304-РПИа-о25\n\n";

    openblas_set_num_threads(1);

    int n = DEFAULT_N;

    if (argc > 1) {
        n = atoi(argv[1]);

        if (n <= 0) {
            n = DEFAULT_N;
        }
    }

    int matrixSize = n * n;
    long double operations = 2.0L * n * n * n;

    cout << "Размер матриц: " << n << " x " << n << "\n";
    cout << "Тип элементов: double\n";
    cout << "Сложность c = 2 * n^3 = " << fixed << setprecision(0) << operations << " операций\n";

    cout << "Вариант 2: функция cblas_dgemm из библиотеки BLAS.\n\n";

    vector<double> a(matrixSize);
    vector<double> b(matrixSize);
    vector<double> c1(matrixSize);
    vector<double> c2(matrixSize);
    vector<double> c3(matrixSize);

    fillMatrix(a, n, 1);
    fillMatrix(b, n, 2);

    cout << "1-й вариант считается...\n";
    auto start = chrono::high_resolution_clock::now();
    multiplySimple(a, b, c1, n);
    auto finish = chrono::high_resolution_clock::now();
    double time1 = getSeconds(start, finish);

    cout << "2-й вариант считается...\n";
    start = chrono::high_resolution_clock::now();
    multiplyBlas(a, b, c2, n);
    finish = chrono::high_resolution_clock::now();
    double time2 = getSeconds(start, finish);

    cout << "3-й вариант считается...\n\n";
    start = chrono::high_resolution_clock::now();
    multiplyOptimized(a, b, c3, n);
    finish = chrono::high_resolution_clock::now();
    double time3 = getSeconds(start, finish);

    printResult("1. Обычная формула", time1, operations);
    printResult("2. cblas_dgemm из BLAS", time2, operations);
    printResult("3. Алгоритм Штрассена и AVX", time3, operations);

    double error12 = maxDifference(c1, c2);
    double error13 = maxDifference(c1, c3);

    cout << "\nПроверка результатов:\n";
    cout << "Максимальная погрешность между 1-м и 2-м вариантом: " << scientific << error12 << "\n";
    cout << "Максимальная погрешность между 1-м и 3-м вариантом: " << scientific << error13 << "\n";

    double percent = getMflops(operations, time3) / getMflops(operations, time2) * 100.0;
    cout << fixed << setprecision(2);
    cout << "Производительность 3-го варианта от 2-го: " << percent << "%\n";

    cout << "Нажмите Enter для выхода...";

    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    cin.get();

    return 0;
}
