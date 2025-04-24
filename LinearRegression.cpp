#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

// Matrix × Matrix → Matrix
Matrix matmul(const Matrix &A, const Matrix &B)
{
    int a_rows = A.size(), a_cols = A[0].size();
    int b_rows = B.size(), b_cols = B[0].size();

    if (a_cols != b_rows)
        throw std::invalid_argument("Số cột của A phải bằng số hàng của B");

    Matrix result(a_rows, Vector(b_cols, 0.0));
    for (int i = 0; i < a_rows; ++i)
        for (int j = 0; j < b_cols; ++j)
            for (int k = 0; k < a_cols; ++k)
                result[i][j] += A[i][k] * B[k][j];

    return result;
}

// Matrix × Vector → Vector
Vector matmul(const Matrix &A, const Vector &v)
{
    int rows = A.size(), cols = A[0].size();
    if (v.size() != cols)
        throw std::invalid_argument("Vector phải có cùng số phần tử với số cột của ma trận");

    Vector result(rows, 0.0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[i] += A[i][j] * v[j];

    return result;
}

// Vector × Matrix → Vector
Vector matmul(const Vector &v, const Matrix &A)
{
    int v_size = v.size(), A_rows = A.size(), A_cols = A[0].size();
    if (v_size != A_rows)
        throw std::invalid_argument("Vector phải có cùng số phần tử với số hàng của ma trận");

    Vector result(A_cols, 0.0);
    for (int j = 0; j < A_cols; ++j)
        for (int i = 0; i < v_size; ++i)
            result[j] += v[i] * A[i][j];

    return result;
}

// Ma trận chuyển vị (transpose)
Matrix transpose(const Matrix &A)
{
    int m = A.size(), n = A[0].size();
    Matrix AT(n, Vector(m));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            AT[j][i] = A[i][j];
    return AT;
}

// Chuyển vector thành ma trận cột
Matrix vector_to_matrix(const Vector &v)
{
    Matrix result(v.size(), Vector(1, 0.0));
    for (size_t i = 0; i < v.size(); ++i)
    {
        result[i][0] = v[i];
    }
    return result;
}

// Chuyển ma trận cột thành vector
Vector matrix_to_vector(const Matrix &m)
{
    Vector result(m.size());
    for (size_t i = 0; i < m.size(); ++i)
    {
        result[i] = m[i][0];
    }
    return result;
}

// Huấn luyện mô hình
void train(const Matrix &X, const Vector &y, Vector &W, double &b,
           int epochs = 1000, double lr = 0.01)
{
    int m = X.size();    // Số mẫu
    int n = X[0].size(); // Số đặc trưng

    Matrix XT = transpose(X); // X chuyển vị

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Tính dự đoán: y_hat = X·W + b
        Vector y_hat(m, b); // Khởi tạo y_hat với giá trị bias

        // Tính X·W và cộng vào y_hat
        Matrix W_matrix = vector_to_matrix(W);
        Matrix XW = matmul(X, W_matrix);

        for (int i = 0; i < m; ++i)
        {
            y_hat[i] += XW[i][0]; // y_hat = XW + b
        }

        // Tính sai số: error = y_hat - y
        Vector error(m);
        for (int i = 0; i < m; ++i)
        {
            error[i] = y_hat[i] - y[i];
        }

        // Tính gradient cho W: grad_W = X^T · error / m
        Matrix error_matrix = vector_to_matrix(error);
        Matrix grad_W_matrix = matmul(XT, error_matrix);
        Vector grad_W = matrix_to_vector(grad_W_matrix);

        for (int i = 0; i < n; ++i)
        {
            grad_W[i] /= m;
        }

        // Tính gradient cho bias: grad_b = sum(error) / m
        double grad_b = 0.0;
        for (int i = 0; i < m; ++i)
        {
            grad_b += error[i];
        }
        grad_b /= m;

        // Cập nhật trọng số W và bias b
        for (int i = 0; i < n; ++i)
        {
            W[i] -= lr * grad_W[i];
        }
        b -= lr * grad_b;

        // In loss mỗi 100 epoch
        if (epoch % 100 == 0)
        {
            double loss = 0.0;
            for (int i = 0; i < m; ++i)
            {
                loss += error[i] * error[i];
            }
            std::cout << "Epoch " << epoch << ", Loss: " << loss / (2 * m) << "\n";
        }
    }
}

int main()
{
    // Dữ liệu X (không có bias)
    Matrix X = {
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0},
        {5.0, 6.0}};

    // Nhãn đầu ra y
    Vector y = {8.0, 11.0, 14.0, 17.0};

    // Khởi tạo vector trọng số W và bias b
    Vector W(X[0].size(), 0.0); // W chỉ chứa trọng số cho các đặc trưng
    double b = 0.0;             // bias tách riêng

    // Huấn luyện mô hình
    train(X, y, W, b);

    // In kết quả trọng số và bias
    std::cout << "\nFinal weights and bias:\n";
    std::cout << "b (bias) = " << b << "\n";
    for (std::size_t i = 0; i < W.size(); ++i)
    {
        std::cout << "w" << i + 1 << " = " << W[i] << "\n";
    }

    // Kiểm tra dự đoán
    std::cout << "\nPredictions vs Actual:\n";
    for (size_t i = 0; i < X.size(); ++i)
    {
        double pred = b; // Bắt đầu với bias
        for (size_t j = 0; j < X[i].size(); ++j)
        {
            pred += X[i][j] * W[j]; // Cộng thêm w_j * x_j
        }
        std::cout << "Sample " << i + 1 << ": Predicted = " << pred
                  << ", Actual = " << y[i] << ", Error = " << (pred - y[i]) << "\n";
    }

    return 0;
}