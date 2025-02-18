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
void solve_matrix_2d(
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
void restore_F_2d(
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

void solve_matrix_1d(
  const double *F,
  const double *lambda,
  const int     size,
  const double  min,
  const double  max,
  const double  step,
  const double  time_step,
  double       *y) 
{
  double A[size], B[size], C[size];
  A[0] = 0.0;
  B[0] = 0.0;
  C[0] = 1.0 / time_step;
  for (int i = 1; i < size; i++) {
    A[i] = -avg(lambda[i - 1], lambda[i]) / (2 * step * step);
    B[i] = -avg(lambda[i], lambda[i + 1]) / (2 * step * step);
    C[i] = 1.0 / time_step - A[i] - B[i];
  }

  double alpha[size], beta[size];
  alpha[0] = 0; alpha[size - 1] = 0;
  beta[0] = min; beta[size - 1] = max;

  for (int i = 0; i < size - 2; i++) {
    alpha[i + 1] = -B[i+1] / (C[i+1] + A[i+1] * alpha[i]);
    beta[i + 1] = (F[i+1] - A[i+1] * beta[i]) / (C[i+1] + A[i+1] * alpha[i]);
  }

  y[size - 1] = max;
  for (int i = size - 2; i >= 0; i--) {
    y[i] = alpha[i] * y[i + 1] + beta[i];
  }
}

void restore_F_1d(
  const double *y,
  const double *lambda,
  const int     size,
  const double  step,
  const double  time_step,
  double       *F) 
{
  for (int i = 1; i < size - 1; ++i) 
  {
    double lambda_plus_half = avg(lambda[i], lambda[i + 1]);
    double lambda_minus_half = avg(lambda[i], lambda[i - 1]);
    double temp = lambda_plus_half * (y[i + 1] - y[i]) - lambda_minus_half * (y[i] - y[i - 1]);
    F[i] = y[i] / time_step + temp / (2 * step * step);
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

  int rank, process_count;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &process_count);
  if (size != process_count) {
    if (rank == 0) {
      printf("Необходимо %d процессов, используются %d\n", size, process_count);
    }
    MPI_Finalize();
    return 0;
  }

  // https://stackoverflow.com/questions/10788180/sending-columns-of-a-matrix-using-mpi-scatter
  MPI_Datatype matrix_columns_type, column_type;
  MPI_Type_vector(size, 1, size, MPI_DOUBLE, &matrix_columns_type);
  MPI_Type_commit(&matrix_columns_type);
  MPI_Type_create_resized(matrix_columns_type, 0, sizeof(double), &column_type);
  MPI_Type_commit(&column_type);

  double lambda_x[size][size];
  double lambda_y[size][size];
  for (int i = 0; i < size; ++i) {
    auto x = i * x_step;
    for (int j = 0; j < size; ++j) {
        auto y = j * y_step;
        // в декартовых координатах x идёт горизонтально, у - вертикально
        // но в матрице x отвечает за строки (вертикально), а y - за столбцы (горизонтально)
        lambda_y[i][j] = get_lambda(x, y);
        lambda_x[j][i] = get_lambda(x, y);
    }
  }
  

  double recv_temperature[size];
  double recv_F[size];
  double temperature[size][size];
  double F[size][size];

  if (rank == 0) 
  {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
          temperature[i][j] = 300;
          F[i][j] = 0;
      }
    }

    for (int i = 0; i < size; ++i) {
      restore_F_2d(true, i, temperature, lambda_x, x_step, time_step, F);
    }
  }

  for (int t = 0; t < time_iterations; ++t) {
    if (rank == 0) {
      log_matrix((double *)temperature, size, t);
    }

    // y^{n+1/2}, F^{n+1/2}
    MPI_Scatter(&F[0][0], 1, column_type, recv_F, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    solve_matrix_1d(recv_F, lambda_x[rank], size, T_left, T_right, x_step, time_step, recv_temperature);
    restore_F_1d(recv_temperature, lambda_x[rank], size, x_step, time_step, recv_F);

    MPI_Gather(recv_temperature, size, MPI_DOUBLE, temperature, 1, column_type, 0, MPI_COMM_WORLD);
    MPI_Gather(recv_F, size, MPI_DOUBLE, F, 1, column_type, 0, MPI_COMM_WORLD);

    // y^{n+1}, F^{n+1}
    MPI_Scatter(&F[0][0], size, MPI_DOUBLE, recv_F, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(&temperature[0][0], size, MPI_DOUBLE, recv_temperature, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const auto x = x_min + rank * x_step; 
    solve_matrix_1d(recv_F, lambda_y[rank], size, 600 * (1 + x), 600 * (1 + x * x * x), y_step, time_step, recv_temperature);
    restore_F_1d(recv_temperature, lambda_y[rank], size, y_step, time_step, recv_F);

    MPI_Gather(recv_temperature, size, MPI_DOUBLE, temperature, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(recv_F, size, MPI_DOUBLE, F, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }  

  MPI_Type_free(&column_type);
  MPI_Type_free(&matrix_columns_type);
  MPI_Finalize();
}