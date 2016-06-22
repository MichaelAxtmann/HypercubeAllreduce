/*******************************************************************************
 * HypercubeAllreduce.cpp
 *
 * Hypercube Allreduce
 *
 *******************************************************************************
 * Copyright (C) 2016 Michael Axtmann <michael.axtmann@kit.edu>
 *
 * The MIT License (MIT)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 ******************************************************************************/

#include <iterator>
#include <ostream>
#include <random>
#include <assert.h>
#include <vector>
#include <iostream>
#include <mpi.h>

using value_type = long;

#include "HypercubeAllreduce.hpp"

template <class T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  if ( !v.empty() ) {
    out << '[';
    std::copy (v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "]";
  }
  return out;
}

template<class Generator, class T>
void FillRandomValues(Generator& gen, std::vector<T>& els) {
    for (auto& el : els) {
        el = gen();
    }
}

class Timer {
public:
    Timer() : start_time(0), total_time(0) {
    }
    void start(MPI_Comm comm) {
        MPI_Barrier(comm);
        start_time = MPI_Wtime();
    }
    void stop() {
        total_time += MPI_Wtime() - start_time;
    }
    double time() {
        return total_time;
    }
    void reset() {
        total_time = 0;
    }
private:
    double start_time;
    double total_time;
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    MPI_Comm comm = MPI::COMM_WORLD;
    int nprocs, myrank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);
    
    size_t log_n_min = 1;
    size_t log_n_max = 20;
    size_t iteration_cnt = 10;
    size_t stage_cnt = 1;
    int tag = 1234;

    std::random_device rd;
    std::mt19937 gen(rd());
    Timer timer;

    std::vector<value_type> els, tmps;

    // Benchmark different input sizes.
    for (size_t n = 1 << log_n_min; n <= (size_t)(1 << log_n_max); n*=2) {
        els.resize(n);
        tmps.resize(n);

        // Each input size is executed several times.
        for (size_t it_idx = 0; it_idx != iteration_cnt; ++it_idx) {
            FillRandomValues(gen, els);

            // Start measurements.
            timer.start(comm);
            // If the latency of MPI_Barrier has to be hidden,
            // execute the algorithm multiple times.
            for (size_t stage_idx = 0; stage_idx != stage_cnt; ++stage_idx) {
                // std::cout << "Rank " << myrank << ": Input=" << els << std::endl;
                HypercubeAllreduce(comm, nprocs, myrank, tag, els, tmps,
                                               [](const value_type& a,
                                                  const value_type& b)
                                               {return a + b;}
                    );
                // std::cout << "Rank " << myrank << ": Output=" << els << std::endl;
            }
            // Stop measurements.
            timer.stop();

            // Print result out.
            if (myrank == 0) {
                std::cout << "it=" << it_idx << "\t"
                          << "n=" << n << "\t"
                          << "stage_cnt=" << stage_cnt << "\t"
                          << "time=" << timer.time() << std::endl;
            }
            timer.reset();
        }
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
