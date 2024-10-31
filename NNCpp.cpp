//
// Created by Richard on 2022/8/5.
// in 1, hidden 2, output 1.

#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <iomanip>

#define InputShape 10
#define Layer1_OutShape 32
#define Layer2_OutShape 16
#define Layer3_OutShape 1

#define DataNum 100
#define Epcho   200

using namespace Eigen;
using namespace std;

MatrixXf Leak_ReLu(MatrixXf, float);

float MSE(MatrixXf, MatrixXf);

class NN {
public:
    NN(MatrixXf input, MatrixXf y_true, float alph);

    MatrixXf ForWard();

    float BackWard();

private:
    // 神经网络的输入值、输出的真实值、学习率等较为固定的值
    MatrixXf input;
    MatrixXf y_true;
    float alph;
    // 神经网络待学习参数
    MatrixXf W1_T;
    MatrixXf B1;
    MatrixXf W2_T;
    MatrixXf B2;
    MatrixXf W3_T;
    MatrixXf B3;
    // 神经网络中间参数
    MatrixXf Z1;
    MatrixXf A1;
    MatrixXf Z2;
    MatrixXf A2;
    MatrixXf Z3;
    MatrixXf A3;  // ==out
    // 反向传播所用参数
    MatrixXf dW3;
    MatrixXf dB3;
    MatrixXf dW2;
    MatrixXf dB2;
    MatrixXf dW1;
    MatrixXf dB1;
    // Layer3
    // j/w3 = j/a3 * a3/z3 * z3/w3
    // j/b3 = j/a3 * a3/z3
    MatrixXf dJ_dA3;
    MatrixXf dA3_dZ3;
    MatrixXf dZ3_dW3;
    // 对Layer2所需参数
    // j/w2 = j/a3 * a3/z3 * z3/a2 * a2/z2 * z2/w2
    // j/b2 = j/a3 * a3/z3 * z3/a2 * a2/z2
    MatrixXf dZ3_dA2;
    MatrixXf dA2_dZ2;
    MatrixXf dZ2_dW2;
    // 对Layer1所需参数
    // j/w1 = j/a3 * a3/z3 * z3/a2 * a2/z2 * z2/a1 * a1/z1 * z1/w1
    // j/b1 = j/a3 * a3/z3 * z3/a2 * a2/z2 * z2/a1 * a1/z1
    MatrixXf dZ2_dA1;
    MatrixXf dA1_Z1;
    MatrixXf dZ1_W1;
};

NN::NN(MatrixXf input, MatrixXf y_true, float alph)  // 初始化权值
{
    this->input = input;
    this->y_true = y_true;
    this->alph = alph;
    this->W1_T = MatrixXf::Random(Layer1_OutShape, InputShape);
    this->W2_T = MatrixXf::Random(Layer2_OutShape, Layer1_OutShape);
    this->W3_T = MatrixXf::Random(Layer3_OutShape, Layer2_OutShape);
    this->B1 = MatrixXf::Zero(Layer1_OutShape, this->input.cols());
    this->B2 = MatrixXf::Zero(Layer2_OutShape, this->input.cols());
    this->B3 = MatrixXf::Zero(Layer3_OutShape, this->input.cols());
}

MatrixXf NN::ForWard() {
    this->Z1 = this->W1_T * this->input;
    this->A1 = Leak_ReLu(this->Z1, this->alph);
    this->Z2 = this->W2_T * this->A1;
    this->A2 = Leak_ReLu(this->Z2, this->alph);
    this->Z3 = this->W3_T * this->A2;
    this->A3 = Leak_ReLu(this->Z3, this->alph);
    return this->A3;
}

float NN::BackWard() {
    int rows_temp, cols_temp;  // 临时行列变量

    // Abount Layer3 work start!!
    this->dJ_dA3 = 2 * (this->A3 - this->y_true);
    this->dA3_dZ3 = MatrixXf::Ones(this->Z3.rows(), this->Z3.cols());
    for (rows_temp = 0; rows_temp < this->Z3.rows(); ++rows_temp) {
        for (cols_temp = 0; cols_temp < this->Z3.cols(); ++cols_temp) {
            this->dA3_dZ3(rows_temp, cols_temp) = this->Z3(rows_temp, cols_temp) >= 0 ? 1.0 : this->alph;
        }
    }
    this->dZ3_dW3 = this->A2.transpose();
    this->dW3 = this->dJ_dA3.cwiseProduct(this->dA3_dZ3) * this->dZ3_dW3 / DataNum;
    this->dB3 = this->dJ_dA3.cwiseProduct(this->dA3_dZ3) / DataNum;
    // Abount Layer3 work end!!

    // Abount Layer2 work start!!
    this->dZ3_dA2 = this->W3_T.transpose();
    this->dA2_dZ2 = MatrixXf::Ones(this->Z2.rows(), this->Z2.cols());
    for (rows_temp = 0; rows_temp < this->Z2.rows(); ++rows_temp) {
        for (cols_temp = 0; cols_temp < this->Z2.cols(); ++cols_temp) {
            this->dA2_dZ2(rows_temp, cols_temp) = this->Z2(rows_temp, cols_temp) >= 0 ? 1.0 : this->alph;
        }
    }

    this->dZ2_dW2 = this->A1.transpose();
    this->dW2 = this->dA2_dZ2.cwiseProduct(this->dZ3_dA2 * this->dJ_dA3.cwiseProduct(this->dA3_dZ3)) * this->dZ2_dW2 /
                DataNum;
    this->dB2 = this->dA2_dZ2.cwiseProduct(this->dZ3_dA2 * this->dJ_dA3.cwiseProduct(this->dA3_dZ3)) / DataNum;
    // Abount Layer2 work end!!

    // Abount Layer1 work start!!
    this->dZ2_dA1 = this->W2_T.transpose();
    this->dA1_Z1 = MatrixXf::Ones(this->Z1.rows(), this->Z1.cols());
    for (rows_temp = 0; rows_temp < this->Z1.rows(); ++rows_temp) {
        for (cols_temp = 0; cols_temp < this->Z1.cols(); ++cols_temp) {
            this->dA1_Z1(rows_temp, cols_temp) = this->Z1(rows_temp, cols_temp) >= 0 ? 1.0 : this->alph;
        }
    }
    this->dZ1_W1 = this->input.transpose();
    this->dW1 = this->dA1_Z1.cwiseProduct(
            this->dZ2_dA1 * this->dA2_dZ2.cwiseProduct(this->dZ3_dA2 * this->dJ_dA3.cwiseProduct(this->dA3_dZ3))) *
                this->dZ1_W1 / DataNum;;
    this->dB1 = this->dA1_Z1.cwiseProduct(
            this->dZ2_dA1 * this->dA2_dZ2.cwiseProduct(this->dZ3_dA2 * this->dJ_dA3.cwiseProduct(this->dA3_dZ3))) /
                DataNum;

    // Abount Layer1 work end!!

    // 调整学习参数
    this->W3_T = this->W3_T - this->alph * this->dW3;
    this->W2_T = this->W2_T - this->alph * this->dW2;
    this->W1_T = this->W1_T - this->alph * this->dW1;
    this->B3 = this->B3 - this->alph * this->dB3;
    this->B2 = this->B2 - this->alph * this->dB2;
    this->B1 = this->B1 - this->alph * this->dB1;
    return MSE(this->y_true, this->ForWard());
}

MatrixXf Leak_ReLu(MatrixXf Z, float a) {
    int Z_rows = Z.rows();
    int Z_cols = Z.cols();
    int i, j;
    // cout << Z_rows <<" " <<Z_cols << endl;
    MatrixXf A(Z_rows, Z_cols);
    // cout << Z_rows << " " << Z_cols << endl;
    for (i = 0; i < Z_rows; ++i) {
        for (j = 0; j < Z_cols; ++j) {
            // cout << "i=" << i << "," << "j=" << j << endl;
            A(i, j) = Z(i, j) >= 0 ? Z(i, j) : a * Z(i, j);
        }
    }

    return A;
}

float MSE(MatrixXf y_true, MatrixXf y_pred)  // 损失函数
{
    MatrixXf true_pred = y_true - y_pred;
    return true_pred.array().square().sum() / true_pred.cols();
}

int main() {
    MatrixXf input_data, y_true, y_pred;
    int rows_temp, cols_temp;  // 行列数临时变量
    int epch;  // 训练轮数临时变量
    int count_right = 0, count_error = 0;
    input_data = MatrixXf::Random(InputShape, DataNum);
    y_true = MatrixXf::Zero(Layer3_OutShape, DataNum);
    for (cols_temp = 0; cols_temp < DataNum; ++cols_temp) {
        y_true(0, cols_temp) = input_data.col(cols_temp).sum() > 0 ? 1.0 : 0.0;
    }

    NN MyNN = NN(input_data, y_true, 0.1);
    for (epch = 0; epch < Epcho; ++epch) {
        MyNN.ForWard();
        cout << MyNN.BackWard() << endl;
    }
    y_pred = MyNN.ForWard();
    cout << setw(15) << "Predicted Value" << setw(20) << "True Value" << endl;
    for (cols_temp = 0; cols_temp < DataNum; ++cols_temp) {
        cout << setw(15) << y_pred(0, cols_temp) << setw(5) << "for" << setw(10) << y_true(0, cols_temp) << setw(10)
             << " is ";
        if ((y_true(0, cols_temp) > 0.5 && y_pred(0, cols_temp) > 0.5) ||
            (y_true(0, cols_temp) < 0.5 && y_pred(0, cols_temp) < 0.5)) {
            cout << "True" << endl;
            count_right++;
        } else {
            cout << "False" << endl;
            count_error++;
        }
    }
    cout << "预测准确率为：" << count_right / (float) DataNum << endl;

    system("pause");
}
