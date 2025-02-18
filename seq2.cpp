#include <mpi.h>
#include <stdio.h>

double get_lambda(double x, double y)
{
  return 0.25 <= x && x <= 0.65 &&
         0.10 <= y && y <= 0.25
         ? 0.01
         : 0.0001; 
}

double avg(double a, double b)
{
  return (a + b) / 2;
}

template <int size>
void solve_matrix(
  const bool   by_x,
  const int    idx,
  const double F[size][size],
  const double lambda[size][size],
  const double min,
  const double max,
  const double step,
  const double time_step,
  double       y[size][size]) 
{
  double A[size], B[size], C[size];
  A[0] = 0.0;
  B[0] = 0.0;
  C[0] = 1.0 / time_step;

  double alpha[size], beta[size];
  alpha[0] = 0; alpha[size - 1] = 0;
  beta[0] = min; beta[size - 1] = max;

  if (by_x)
  {
    for (int i = 1; i < size; i++) {
      A[i] = -avg(lambda[i - 1][idx], lambda[i][idx]) / (2 * step * step);
      B[i] = -avg(lambda[i][idx], lambda[i + 1][idx]) / (2 * step * step);
      C[i] = 1.0 / time_step - A[i] - B[i];
    }
  
    for (int i = 0; i < size - 2; i++) {
      alpha[i + 1] = -B[i+1] / (C[i+1] + A[i+1] * alpha[i]);
      beta[i + 1] = (F[i+1][idx] - A[i+1] * beta[i]) / (C[i+1] + A[i+1] * alpha[i]);
    }
  
    y[size - 1][idx] = max;
    for (int i = size - 2; i >= 0; i--) {
      y[i][idx] = alpha[i] * y[i + 1][idx] + beta[i];
    }
  }
  else
  {
    for (int i = 1; i < size; i++) {
      A[i] = -avg(lambda[idx][i - 1], lambda[idx][i]) / (2 * step * step);
      B[i] = -avg(lambda[idx][i], lambda[idx][i + 1]) / (2 * step * step);
      C[i] = 1.0 / time_step - A[i] - B[i];
    }
  
    for (int i = 0; i < size - 2; i++) {
      alpha[i + 1] = -B[i+1] / (C[i+1] + A[i+1] * alpha[i]);
      beta[i + 1] = (F[idx][i+1] - A[i+1] * beta[i]) / (C[i+1] + A[i+1] * alpha[i]);
    }
  
    y[idx][size - 1] = max;
    for (int i = size - 2; i >= 0; i--) {
      y[idx][i] = alpha[i] * y[idx][i + 1] + beta[i];
    }
  }
}

template <int size>
void restore_F(
  const bool   by_x,
  const int    idx,
  const double y[size][size],
  const double lambda[size][size],
  const double step,
  const double time_step,
  double       F[size][size]) 
{
  if (by_x)
  {
    for (int i = 1; i < size - 1; ++i) 
    {
      double lambda_plus_half = avg(lambda[i][idx], lambda[i + 1][idx]);
      double lambda_minus_half = avg(lambda[i][idx], lambda[i - 1][idx]);
      double temp = lambda_plus_half * (y[i + 1][idx] - y[i][idx]) - lambda_minus_half * (y[i][idx] - y[i - 1][idx]);
      F[i][idx] = y[i][idx] / time_step + temp / (2 * step * step);
    }
  }
  else 
  {
    for (int i = 1; i < size - 1; ++i) 
    {
      double lambda_plus_half = avg(lambda[idx][i], lambda[idx][i + 1]);
      double lambda_minus_half = avg(lambda[idx][i], lambda[idx][i - 1]);
      double temp = lambda_plus_half * (y[idx][i + 1] - y[idx][i]) - lambda_minus_half * (y[idx][i] - y[idx][i - 1]);
      F[idx][i] = y[idx][i] / time_step + temp / (2 * step * step);
    }
  }
}

void log_matrix(double* matrix, const int size, int time = -1) {
  for (int i = 1; i < size - 1; ++i) {
      for (int j = 1; j < size - 1; ++j) {
        if (time == -1) {
          printf("%d;%d;%lf\n", i, j, matrix[i * size + j]);
        }
        else {
          printf("%d;%d;%d;%lf\n", i, j, time, matrix[i * size + j]);
        }
      }
  }
}

int main(int argc, char **argv)
{
  const int size = 30;

  const double time_step = 0.2;
  // const double time_min = 0;
  // const double time_max = 10;
  const int time_iterations = 3000; // (time_max - time_min) / time_step; 

  const double x_min = 0;
  const double x_max = 1;
  const double x_step = (x_max - x_min) / size;

  const double y_min = 0;
  const double y_max = 0.5;
  const double y_step = (y_max - y_min) / size;

  const double T_left = 600;
  const double T_right = 1200;


  // y - исходная температура
  double temperature[size][size];
  double lambda[size][size];
  // F
  double F[size][size];
  for (int i = 0; i < size; ++i) {
      auto x = i * x_step;
      for (int j = 0; j < size; ++j) {
          auto y = j * y_step;
          lambda[i][j] = get_lambda(x, y);
          temperature[i][j] = 300;
          F[i][j] = 0;
      }
  }

  for (int i = 0; i < size; ++i) {
      restore_F(true, i, temperature, lambda, x_step, time_step, F);
  }

  for (int t = 0; t < time_iterations; ++t) {
    log_matrix((double *)temperature, size, t);

    // y^{n+1/2}
    for (int x = 0; x < size; ++x) {
      solve_matrix(true, x, F, lambda, T_left, T_right, x_step, time_step, temperature);
    }

    // F^{n+1/2}
    for (int x = 0; x < size; ++x) {
      restore_F(true, x, temperature, lambda, x_step, time_step, F);
    }

    // y^{n+1}
    for (int y = 0; y < size; ++y) {
      double x = x_min + y * x_step;
      solve_matrix(false, y, F, lambda, 600 * (1 + x), 600 * (1 + x * x * x), y_step, time_step, temperature);
    }

    // F^{n+1}
    for (int y = 0; y < size; ++y) {
      restore_F(false, y, temperature, lambda, y_step, time_step, F);
    }
  }
}