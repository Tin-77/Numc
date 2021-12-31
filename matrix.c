#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if (rows < 1 || cols < 1) return -1;
    matrix *m = (matrix *) malloc(sizeof(matrix));
    m->rows = rows;
    m->cols = cols;
    m->parent = NULL;
    m->ref_cnt = 1;
    if (rows == 1 || cols == 1) {
        m->is_1d = 1;
    } else {
        m->is_1d = 0;
    }
    // double **data = (double **) calloc(rows, sizeof(double*));
    // if (data == NULL) return -1;
    // for (int i = 0; i < rows; i++) {
    //     data[i] = (double *) calloc(cols, sizeof(double));
    //     if (data[i] == NULL) return -1;
    // }

    // have array of double * pointing to start of each row
    double **ptrs = (double **) calloc(rows,sizeof(double*));
    if (ptrs == NULL) return -1;
    ptrs[0] = (double *) calloc(rows*cols, sizeof(double));
    if (ptrs[0] == NULL) return -1;
    for (int i = 1; i < rows; i++) {
        ptrs[i] = ptrs[i - 1] + cols;
    }

    m->data = ptrs;
    *mat = m;
    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    if (rows < 1 || cols < 1) return -1;
    matrix *m = (matrix*) malloc(sizeof(matrix));
    m->rows = rows;
    m->cols = cols;
    matrix *temp = from;
    while (temp->parent != NULL) {
        temp = temp->parent;
    }
    m->parent = temp;
    temp->ref_cnt = 1 + temp->ref_cnt;
    m->ref_cnt = 1;
    if (rows == 1 || cols == 1) {
        m->is_1d = 1;
    } else {
        m->is_1d = 0;
    }
    // double **data = (double**) malloc(rows*sizeof(double**));
    // if (data == NULL) return -1;
    // for (int i = 0; i < rows; i++) {
    //     data[i] = (double*) malloc(cols*sizeof(double));
    //     if (data[i] == NULL) return -1;
    //     data[i] = from->data[row_offset+i]+col_offset;
    // }
    double **ptrs = (double **) malloc(rows*sizeof(double*));
    if (ptrs == NULL) return -1;
    for (int i = 0; i < rows; i++) {
        ptrs[i] = from->data[row_offset+i]+col_offset;
    }
    m->data = ptrs;
    *mat = m;
    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat->data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if (mat == NULL) return;
    if (mat->parent == NULL) {
        if (mat->ref_cnt == 1) {
            // for (int i = 0; i < mat->rows; i++) {
            //     free(mat->data[i]);
            // }
            // for (int i = 0; i < mat->rows*mat->cols; i++) {
            //     free(&mat->data[0][i]);
            // }
            free(mat->data[0]);
            free(mat->data);
            free(mat);
        }
    } else {
        mat->parent->ref_cnt -= 1;
        free(mat->data);
        free(mat);
    }
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    return mat->data[row][col];
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    mat->data[row][col] = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
    #pragma omp parallel for
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i][j] = val;
        }
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) return -1;
    #pragma omp parallel for
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols / 8 * 8; j+=8) {
            result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
            result->data[i][j+1] = mat1->data[i][j+1] + mat2->data[i][j+1];
            result->data[i][j+2] = mat1->data[i][j+2] + mat2->data[i][j+2];
            result->data[i][j+3] = mat1->data[i][j+3] + mat2->data[i][j+3];
            result->data[i][j+4] = mat1->data[i][j+4] + mat2->data[i][j+4];
            result->data[i][j+5] = mat1->data[i][j+5] + mat2->data[i][j+5];
            result->data[i][j+6] = mat1->data[i][j+6] + mat2->data[i][j+6];
            result->data[i][j+7] = mat1->data[i][j+7] + mat2->data[i][j+7];
        }
        for (int j = mat1->cols / 8 * 8; j < mat1->cols; j++) {
            result->data[i][j] = mat1->data[i][j] + mat2->data[i][j];
        }
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    if (mat1->rows != mat2->rows || mat1->cols != mat2->cols) return -1;
    #pragma omp parallel for
    for (int i = 0; i < mat1->rows; i++) {
        for (int j = 0; j < mat1->cols / 8 * 8; j+=8) {
            result->data[i][j] = mat1->data[i][j] - mat2->data[i][j];
            result->data[i][j+1] = mat1->data[i][j+1] - mat2->data[i][j+1];
            result->data[i][j+2] = mat1->data[i][j+2] - mat2->data[i][j+2];
            result->data[i][j+3] = mat1->data[i][j+3] - mat2->data[i][j+3];
            result->data[i][j+4] = mat1->data[i][j+4] - mat2->data[i][j+4];
            result->data[i][j+5] = mat1->data[i][j+5] - mat2->data[i][j+5];
            result->data[i][j+6] = mat1->data[i][j+6] - mat2->data[i][j+6];
            result->data[i][j+7] = mat1->data[i][j+7] - mat2->data[i][j+7];
        }
        for (int j = mat1->cols / 8 * 8; j < mat1->cols; j++) {
            result->data[i][j] = mat1->data[i][j] - mat2->data[i][j];
        }
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    if (mat1->cols != mat2->rows || result->rows != mat1->rows || result->cols != mat2->cols) return -1;
    if (mat1->rows*mat1->cols > 500) {
        //Transpose the second matrix
        matrix *mat2_transposed;
        allocate_matrix(&mat2_transposed, mat2->cols, mat2->rows);
        #pragma omp parallel 
        {
            #pragma omp for schedule(static, 1)
            for (int r = 0; r < mat2->rows; r++) {
                for (int c = 0; c < mat2->cols; c++) {
                    mat2_transposed->data[c][r] = mat2->data[r][c];
                }
            }
            #pragma omp for schedule(static, 1)
            for (int i = 0; i < mat1->rows; i++) {
                for (int j = 0; j < mat2->cols / 2 * 2; j += 2) {
                    __m256d add_vec1 = _mm256_setzero_pd();
                    __m256d add_vec2 = _mm256_setzero_pd();
                    for (int k = 0; k < mat1->cols / 8 * 8; k+=8) {
                        __m256d vector1 = _mm256_loadu_pd(&mat1->data[i][k]);
                        __m256d vector2 = _mm256_loadu_pd(&mat2_transposed->data[j][k]);
                        __m256d mul_vec = _mm256_mul_pd(vector1, vector2);
                        add_vec1 = _mm256_add_pd(add_vec1, mul_vec);

                        vector1 = _mm256_loadu_pd(&mat1->data[i][k+4]);
                        vector2 = _mm256_loadu_pd(&mat2_transposed->data[j][k+4]);
                        mul_vec = _mm256_mul_pd(vector1, vector2);
                        add_vec1 = _mm256_add_pd(add_vec1, mul_vec);

                        vector1 = _mm256_loadu_pd(&mat1->data[i][k]);
                        vector2 = _mm256_loadu_pd(&mat2_transposed->data[j + 1][k]);
                        mul_vec = _mm256_mul_pd(vector1, vector2);
                        add_vec2 = _mm256_add_pd(add_vec2, mul_vec);

                        vector1 = _mm256_loadu_pd(&mat1->data[i][k + 4]);
                        vector2 = _mm256_loadu_pd(&mat2_transposed->data[j + 1][k + 4]);
                        mul_vec = _mm256_mul_pd(vector1, vector2);
                        add_vec2 = _mm256_add_pd(add_vec2, mul_vec);

                    }
                    double temp1[4] = {0};
                    double temp2[4] = {0};
                    _mm256_store_pd(temp1, add_vec1);
                    _mm256_store_pd(temp2, add_vec2);
                    double product1 = 0;
                    double product2 = 0;
                    for (int index = 0; index < 4; index++) {
                        product1 += temp1[index];
                        product2 += temp2[index];
                    }
                    result->data[i][j] = product1;
                    result->data[i][j + 1] = product2;
                    for (int k = mat1->cols / 8 * 8; k < mat1->cols; k++) {
                        result->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
                        result->data[i][j+1] += mat1->data[i][k] * mat2->data[k][j+1];
                    }
                }
                for (int j = mat2->cols / 2 * 2; j < mat2->cols; j++) {
                    __m256d add_vec1 = _mm256_setzero_pd();
                    for (int k = 0; k < mat1->cols / 8 * 8; k+=8) {
                        __m256d vector1 = _mm256_loadu_pd(&mat1->data[i][k]);
                        __m256d vector2 = _mm256_loadu_pd(&mat2_transposed->data[j][k]);
                        __m256d mul_vec = _mm256_mul_pd(vector1, vector2);
                        add_vec1 = _mm256_add_pd(add_vec1, mul_vec);

                        vector1 = _mm256_loadu_pd(&mat1->data[i][k+4]);
                        vector2 = _mm256_loadu_pd(&mat2_transposed->data[j][k+4]);
                        mul_vec = _mm256_mul_pd(vector1, vector2);
                        add_vec1 = _mm256_add_pd(add_vec1, mul_vec);
                    }
                    double temp1[4] = {0};
                    _mm256_store_pd(temp1, add_vec1);
                    double product1 = 0;
                    for (int index = 0; index < 4; index++) {
                        product1 += temp1[index];
                    }
                    result->data[i][j] = product1;
                    for (int k = mat1->cols / 8 * 8; k < mat1->cols; k++) {
                        result->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
                    }
                }
            }
        }
        deallocate_matrix(mat2_transposed);
    } else {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols; j++) {
                for (int k = 0; k < mat1->cols; k++) {
                    result->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
                }
            }
        }
    }
    return 0;
}

int mul_matrix_helper(matrix *result, matrix *mat1, matrix *mat2, matrix *mat2_transposed, matrix *mat_copy) {
    /* TODO: YOUR CODE HERE */
    if (mat1->cols != mat2->rows || result->rows != mat1->rows || result->cols != mat2->cols) return -1;
    
    #pragma omp parallel 
    {
        #pragma omp for schedule(static, 1)
        for (int i = 0; i < mat1->rows; i++) {
            for (int k = 0; k < mat1->cols; k+=1) {
                mat_copy->data[i][k] = 0.0;
            }
        }
        #pragma omp for schedule(static, 1)
        for (int r = 0; r < mat2->rows; r++) {
            for (int c = 0; c < mat2->cols; c++) {
                mat2_transposed->data[c][r] = mat2->data[r][c];
            }
        }
        #pragma omp for schedule(static, 1)
        for (int i = 0; i < mat1->rows; i++) {
            for (int j = 0; j < mat2->cols / 2 * 2; j += 2) {
                __m256d add_vec1 = _mm256_setzero_pd();
                __m256d add_vec2 = _mm256_setzero_pd();
                for (int k = 0; k < mat1->cols / 8 * 8; k+=8) {
                    __m256d vector1 = _mm256_loadu_pd(&mat1->data[i][k]);
                    __m256d vector2 = _mm256_loadu_pd(&mat2_transposed->data[j][k]);
                    __m256d mul_vec = _mm256_mul_pd(vector1, vector2);
                    add_vec1 = _mm256_add_pd(add_vec1, mul_vec);

                    vector1 = _mm256_loadu_pd(&mat1->data[i][k+4]);
                    vector2 = _mm256_loadu_pd(&mat2_transposed->data[j][k+4]);
                    mul_vec = _mm256_mul_pd(vector1, vector2);
                    add_vec1 = _mm256_add_pd(add_vec1, mul_vec);

                    vector1 = _mm256_loadu_pd(&mat1->data[i][k]);
                    vector2 = _mm256_loadu_pd(&mat2_transposed->data[j + 1][k]);
                    mul_vec = _mm256_mul_pd(vector1, vector2);
                    add_vec2 = _mm256_add_pd(add_vec2, mul_vec);

                    vector1 = _mm256_loadu_pd(&mat1->data[i][k + 4]);
                    vector2 = _mm256_loadu_pd(&mat2_transposed->data[j + 1][k + 4]);
                    mul_vec = _mm256_mul_pd(vector1, vector2);
                    add_vec2 = _mm256_add_pd(add_vec2, mul_vec);

                }
                double temp1[4];
                double temp2[4];
                _mm256_storeu_pd(temp1, add_vec1);
                _mm256_storeu_pd(temp2, add_vec2);
                double product1 = 0;
                double product2 = 0;
                for (int index = 0; index < 4; index++) {
                    product1 += temp1[index];
                    product2 += temp2[index];
                }
                mat_copy->data[i][j] = product1;
                mat_copy->data[i][j + 1] = product2;
                for (int k = mat1->cols / 8 * 8; k < mat1->cols; k++) {
                    mat_copy->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
                    mat_copy->data[i][j+1] += mat1->data[i][k] * mat2->data[k][j+1];
                }
            }
            for (int j = mat2->cols / 2 * 2; j < mat2->cols; j++) {
                __m256d add_vec1 = _mm256_setzero_pd();
                for (int k = 0; k < mat1->cols / 8 * 8; k+=8) {
                    __m256d vector1 = _mm256_loadu_pd(&mat1->data[i][k]);
                    __m256d vector2 = _mm256_loadu_pd(&mat2_transposed->data[j][k]);
                    __m256d mul_vec = _mm256_mul_pd(vector1, vector2);
                    add_vec1 = _mm256_add_pd(add_vec1, mul_vec);

                    vector1 = _mm256_loadu_pd(&mat1->data[i][k+4]);
                    vector2 = _mm256_loadu_pd(&mat2_transposed->data[j][k+4]);
                    mul_vec = _mm256_mul_pd(vector1, vector2);
                    add_vec1 = _mm256_add_pd(add_vec1, mul_vec);
                }
                double temp1[4] = {0};
                _mm256_store_pd(temp1, add_vec1);
                double product1 = 0;
                for (int index = 0; index < 4; index++) {
                    product1 += temp1[index];
                }
                mat_copy->data[i][j] = product1;
                for (int k = mat1->cols / 8 * 8; k < mat1->cols; k++) {
                    mat_copy->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
                }
            }
        }
        #pragma omp for schedule(static, 1)
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                result->data[i][j] = mat_copy->data[i][j];
            }
        }
    }
    return 0;
}

int mul_matrix_helper2(matrix *result, matrix *mat1, matrix *mat2, matrix *mat_copy) {
    // if (mat1->cols <= 10) {
    //     omp_set_num_threads(4);
    // } else {
    //     omp_set_num_threads(8);
    // }
    #pragma omp parallel
    {
        #pragma omp for schedule(static, 1)
        for (int i = 0; i < mat1->rows; i++) {
            for (int k = 0; k < mat1->cols; k+=1) {
                mat_copy->data[i][k] = 0.0;
            }
        }
        #pragma omp for schedule(static, 1)
        for (int i = 0; i < mat1->rows; i++) {
            for (int k = 0; k < mat2->cols / 4 * 4; k+=4) {
                for (int j = 0; j < mat1->cols; j++) {
                    mat_copy->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
                    mat_copy->data[i][j] += mat1->data[i][k+1] * mat2->data[k+1][j];
                    mat_copy->data[i][j] += mat1->data[i][k+2] * mat2->data[k+2][j];
                    mat_copy->data[i][j] += mat1->data[i][k+3] * mat2->data[k+3][j];
                }
            }
            for (int k = mat1->rows / 4 * 4; k < mat1->rows; k += 1) {
                for (int j = 0; j < mat1->cols; j++) {
                    mat_copy->data[i][j] += mat1->data[i][k] * mat2->data[k][j];
                }
            }
        }
        #pragma omp parallel for schedule(static, 1)
        for (int i = 0; i < result->rows; i++) {
            for (int j = 0; j < result->cols; j++) {
                result->data[i][j] = mat_copy->data[i][j];
            }
        }
    }
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    if (mat->rows != mat->cols || pow < 0) return -1;
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            if (i == j) {
                result->data[i][j] = 1;
            } else {
                result->data[i][j] = 0;
            }
        }
    }
    matrix *mat_copy;
    allocate_matrix(&mat_copy, mat->rows, mat->cols);
    #pragma omp parallel for
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
        mat_copy->data[i][j] = mat->data[i][j];
        }
    }
    matrix *mat_copy2;
    allocate_matrix(&mat_copy2, mat->rows, mat->cols);
    if (mat->cols > 500) {
        matrix *mat2_transposed;
        allocate_matrix(&mat2_transposed, mat->cols, mat->rows);
        while (pow != 0) {
            if (pow%2 == 0) {
                mul_matrix_helper(mat_copy, mat_copy, mat_copy, mat2_transposed, mat_copy2);
                pow /= 2;
            } else {
                mul_matrix_helper(result, result, mat_copy, mat2_transposed, mat_copy2);
                pow -= 1;
            }
        }
        deallocate_matrix(mat2_transposed);
    } else {
        while (pow != 0) {
            if (pow%2 == 0) {
                mul_matrix_helper2(mat_copy, mat_copy, mat_copy, mat_copy2);
                pow /= 2;
            } else {
                mul_matrix_helper2(result, result, mat_copy, mat_copy2);
                pow -= 1;
            }
        }
    }
    deallocate_matrix(mat_copy);
    deallocate_matrix(mat_copy2);
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if (result->rows != mat->rows || result->cols != mat->cols) return -1;
    if (mat->rows*mat->cols > 100) {
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < mat->rows; i++) {
                for (int j = 0; j < mat->cols/8*8; j+=8) {
                    result->data[i][j] = -mat->data[i][j];
                    result->data[i][j+1] = -mat->data[i][j+1];
                    result->data[i][j+2] = -mat->data[i][j+2];
                    result->data[i][j+3] = -mat->data[i][j+3];
                    result->data[i][j+4] = -mat->data[i][j+4];
                    result->data[i][j+5] = -mat->data[i][j+5];
                    result->data[i][j+6] = -mat->data[i][j+6];
                    result->data[i][j+7] = -mat->data[i][j+7];
                }
            }
            #pragma omp for
            for (int i = 0; i < mat->rows; i++) {
                for (int j = mat->cols/8*8; j < mat->cols; j++) {
                    result->data[i][j] = -mat->data[i][j];
                }
            }
        }
    } else {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < mat->rows; i++) {
            for (int j = 0; j < mat->cols; j++) {
                result->data[i][j] = -mat->data[i][j];
            }
        }
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
    if (result->rows != mat->rows || result->cols != mat->cols) return -1;
    #pragma omp parallel for
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            result->data[i][j] = mat->data[i][j] > 0 ? mat->data[i][j] : -mat->data[i][j];
        }
    }
    return 0;
}
