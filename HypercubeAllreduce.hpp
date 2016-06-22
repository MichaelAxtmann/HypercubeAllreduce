/*******************************************************************************
 * HypercubeAllreduce.hpp
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

#include <math.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <mpi.h>

/* @brief Test that all vectors els and tmps have the same size.
 */
template<class T>
bool TestInit(MPI_Comm comm, const std::vector<T>& els, const std::vector<T>& tmps) {
    int own_el_cnt = els.size();
    int own_tmp_cnt = tmps.size();
    int min_el_cnt, max_el_cnt, min_tmp_cnt, max_tmp_cnt;

    MPI_Allreduce(&own_el_cnt, &min_el_cnt, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&own_el_cnt, &max_el_cnt, 1, MPI_INT, MPI_MAX, comm);

    MPI_Allreduce(&own_tmp_cnt, &min_tmp_cnt, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&own_tmp_cnt, &max_tmp_cnt, 1, MPI_INT, MPI_MAX, comm);

    return min_el_cnt == max_el_cnt
        && min_el_cnt == max_tmp_cnt
        && min_el_cnt == min_tmp_cnt;
}

template<class BinaryOperation, class T>
void HypercubeAllreduce(MPI_Comm comm, int nprocs, int rank, int tag,
                        std::vector<T>& els, std::vector<T>& tmps,
                        BinaryOperation reduce_op) {
    assert(els.size() == tmps.size());
    assert(TestInit(comm, els, tmps));

    // Hypercube exchange.
    const int comm_phases = log2(nprocs);
    for (int phase = 0; phase < comm_phases; phase++) {

        // Init communication pattern.
        int mask = 1 << phase;
        int comm_partner = rank;
        comm_partner ^= mask;
        std::vector<MPI_Request> requests(2);

        // Start send operation.
        MPI_Isend(els.data(), els.size() * sizeof(T), MPI_BYTE,
                  comm_partner, tag, comm, requests.data());

        // Start receive operation.
        MPI_Status recv_status;
        // Actually, we do not need a Probe and Get_count.
        // However, most of our hypercube algorithms do not
        // receive a deterministic number of elements.
        // So we would not know the recv_size.
        MPI_Probe(comm_partner, tag, comm, &recv_status);
        int recv_size;
        MPI_Get_count(&recv_status, MPI_BYTE, &recv_size);
        assert(recv_size % sizeof(T) == 0);
        assert(recv_size / sizeof(T) == tmps.size());
        MPI_Irecv(tmps.data(), tmps.size() * sizeof(T), MPI_BYTE,
                  comm_partner, tag, comm, requests.data() + 1);

        // Wait until send and receive operation is finished.
        MPI_Waitall(2, requests.data(), MPI_STATUSES_IGNORE);

        // Reduce data to els.
        for (size_t el_idx = 0; el_idx != els.size(); ++el_idx) {
            els[el_idx] = reduce_op(els[el_idx], tmps[el_idx]);
        }
    }
    
    return;
}
