/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
//
// Rephrasing as a benchmark for interface with QUDA by
// Bartosz Kostrzewa, Aniket Sen (Uni Bonn) 
//
//@HEADER
*/

#include <quda.h>
#include <util_quda.h>
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <getopt.h>

#define MAX(a,b) ((a)>(b)?(a):(b))
#define Nd 4
#define Nc 3

#define HLINE "-------------------------------------------------------------\n"

using real_t = double;
using val_t = Kokkos::complex<real_t>;

using RNG = Kokkos::Random_XorShift1024_Pool<Kokkos::DefaultExecutionSpace>;

using GaugeField = 
    Kokkos::View<val_t****[Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;

using SUNField1D = 
    Kokkos::View<val_t*[Nc][Nc], Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Restrict>>;

#if defined(KOKKOS_ENABLE_CUDA)
using constGaugeField = 
    Kokkos::View<const val_t****[Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

using constSUNField1D = 
    Kokkos::View<const val_t*[Nc][Nc], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
#else
using constGaugeField = 
    Kokkos::View<const val_t****[Nd][Nc][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;

using constSUNField1D =
    Kokkos::View<const val_t*[Nc][Nc], Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Restrict>>;
#endif

using StreamIndex = int;

template <int rank>
using Policy      = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;

template <std::size_t... Idcs>
constexpr Kokkos::Array<std::size_t, sizeof...(Idcs)>
make_repeated_sequence_impl(std::size_t value, std::integer_sequence<std::size_t, Idcs...>)
{
  return { ((void)Idcs, value)... };
}

template <std::size_t N>
constexpr Kokkos::Array<std::size_t,N> 
make_repeated_sequence(std::size_t value)
{
  return make_repeated_sequence_impl(value, std::make_index_sequence<N>{});
}

template <typename V>
auto
get_tiling(const V view)
{
  constexpr auto rank = view.rank_dynamic();
  // extract the dimensions from the view layout (assuming no striding)
  const auto & dimensions = view.layout().dimension;
  Kokkos::Array<std::size_t,rank> dims;
  for(int i = 0; i < rank; ++i){
    dims[i] = dimensions[i];
  }
  // extract the recommended tiling for this view from a "default" policy 
  const auto rec_tiling = Policy<rank>(make_repeated_sequence<rank>(0),dims).tile_size_recommended();
  
  if constexpr (std::is_same_v<typename V::execution_space, Kokkos::DefaultHostExecutionSpace>){
    // for OpenMP we parallelise over the two outermost (leftmost) dimensions and so the chunk size
    // for the innermost dimensions corresponds to the view extents
    return Kokkos::Array<std::size_t,rank>({1,1,view.extent(2),view.extent(3)});
  } else {
    // for GPUs we use the recommended tiling for now, we just need to convert it appropriately
    // from "array_index_type"
    // unfortunately the recommended tile size may exceed the maximum block size on GPUs 
    // for large ranks -> let's cap the tiling at 4 dims
    constexpr auto max_rank = rank > 4 ? 4 : rank;
    Kokkos::Array<std::size_t,max_rank> res;
    for(int i = 0; i < max_rank; ++i){
      res[i] = rec_tiling[i];
    }
    return res;
  }
}

struct deviceGaugeField {
  deviceGaugeField() = delete;

  deviceGaugeField(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, const RNG & rng)
  {
    do_init(N0,N1,N2,N3,view,rng);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, 
          GaugeField & V, const RNG & rng){
    Kokkos::realloc(Kokkos::WithoutInitializing, V, N0, N1, N2, N3);
    
    // need a const view to get the constexpr rank
    const GaugeField vconst = V;
    constexpr auto rank = vconst.rank_dynamic();
    const auto tiling = get_tiling(vconst);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), {N0,N1,N2,N3}, tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        auto rand = rng.get_state();
        #pragma unroll
        for(int mu = 0; mu < Nd; ++mu){
          #pragma unroll
          for(int c1 = 0; c1 < Nc; ++c1){
            #pragma unroll
            for(int c2 = 0; c2 < Nc; ++c2){
              V(i,j,k,l,mu,c1,c2) = val_t(rand.drand(-1., 1.), rand.drand(-1., 1.));
            }
          }
        }
        rng.free_state(rand);
      }
    );
    Kokkos::fence();
  }

  GaugeField view;
};

struct deviceGaugeField_quda {
  deviceGaugeField_quda() = delete;

  deviceGaugeField_quda(const deviceGaugeField g)
  {
    do_init(view[0],view[1],view[2],view[3],g);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(SUNField1D & V0, SUNField1D & V1,
          SUNField1D & V2, SUNField1D & V3, 
          const deviceGaugeField g){
    std::size_t N0 = g.view.extent(0);
    std::size_t N1 = g.view.extent(1);
    std::size_t N2 = g.view.extent(2);
    std::size_t N3 = g.view.extent(3);
    std::size_t volume = N0*N1*N2*N3;
    Kokkos::realloc(Kokkos::WithoutInitializing, V0, volume);
    Kokkos::realloc(Kokkos::WithoutInitializing, V1, volume);
    Kokkos::realloc(Kokkos::WithoutInitializing, V2, volume);
    Kokkos::realloc(Kokkos::WithoutInitializing, V3, volume);
    
    // need a const view to get the constexpr rank
    const GaugeField vconst = g.view;
    constexpr auto rank = vconst.rank_dynamic();
    const auto tiling = get_tiling(vconst);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), {N0,N1,N2,N3}, tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        // copy gauge and reorder according to QUDA_QDP_GAUGE_ORDER
        const StreamIndex i_lex = i + j*N0 + k*N0*N1 + l*N0*N1*N2;
        const StreamIndex oddBit = (i+j+k+l) & 1;
        const StreamIndex quda_idx = oddBit*volume/2 + i_lex/2;
          #pragma unroll
          for(int c1 = 0; c1 < Nc; ++c1){
            #pragma unroll
            for(int c2 = 0; c2 < Nc; ++c2){
              V0(quda_idx,c1,c2) = g.view(i,j,k,l,0,c1,c2);
              V1(quda_idx,c1,c2) = g.view(i,j,k,l,1,c1,c2);
              V2(quda_idx,c1,c2) = g.view(i,j,k,l,2,c1,c2);
              V3(quda_idx,c1,c2) = g.view(i,j,k,l,3,c1,c2);
            }
          }
      }
    );
    Kokkos::fence();
  }

  SUNField1D view[Nd];
};

val_t perform_plaquette(const deviceGaugeField g_in)
{
  constexpr auto rank = g_in.view.rank_dynamic();
  const auto stream_array_size = g_in.view.extent(0);
  const auto tiling = get_tiling(g_in.view);

  const constGaugeField g(g_in.view); 
  
  val_t res = 0.0;

  Kokkos::parallel_reduce(
    "suN_plaquette", 
    Policy<rank>(make_repeated_sequence<rank>(0), 
                 {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, 
                 tiling),
    KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l,
                  val_t & lres)
    {
      Kokkos::Array<Kokkos::Array<val_t,Nc>,Nc> lmu, lnu;

      val_t tmu, tnu;

      #pragma unroll
      for(int mu = 0; mu < Nd; ++mu){
        #pragma unroll
        for(int nu = 0; nu < Nd; ++nu){
          // unrolling only works well with constant-value loop limits
          if( nu > mu ){
            const StreamIndex ipmu = mu == 0 ? (i + 1) % stream_array_size : i;
            const StreamIndex jpmu = mu == 1 ? (j + 1) % stream_array_size : j;
            const StreamIndex kpmu = mu == 2 ? (k + 1) % stream_array_size : k;
            const StreamIndex lpmu = mu == 3 ? (l + 1) % stream_array_size : l;
            const StreamIndex ipnu = nu == 0 ? (i + 1) % stream_array_size : i;
            const StreamIndex jpnu = nu == 1 ? (j + 1) % stream_array_size : j;
            const StreamIndex kpnu = nu == 2 ? (k + 1) % stream_array_size : k;
            const StreamIndex lpnu = nu == 3 ? (l + 1) % stream_array_size : l;
            #pragma unroll
            for(int c1 = 0; c1 < Nc; ++c1){
              #pragma unroll
              for(int c2 = 0; c2 < Nc; ++c2){
                tmu = g(i,j,k,l,mu,c1,0) * g(ipmu,jpmu,kpmu,lpmu,nu,0,c2);
                tnu = g(i,j,k,l,nu,c1,0) * g(ipnu,jpnu,kpnu,lpnu,mu,0,c2);
                #pragma unroll
                for(int ci = 1; ci < Nc; ++ci){
                  tmu += g(i,j,k,l,mu,c1,ci) * g(ipmu,jpmu,kpmu,lpmu,nu,ci,c2);
                  tnu += g(i,j,k,l,nu,c1,ci) * g(ipnu,jpnu,kpnu,lpnu,mu,ci,c2);
                }
                lmu[c1][c2] = tmu;
                lnu[c1][c2] = tnu;
              }
            }
            #pragma unroll
            for(int c = 0; c < Nc; ++c){
              #pragma unroll
              for(int ci = 0; ci < Nc; ++ci){
                lres += lmu[c][ci] * Kokkos::conj(lnu[c][ci]);
              }
            }
          }
        }
      }
    }, Kokkos::Sum<val_t>(res) );
  Kokkos::fence();
  return res;
}

void _loadGaugeQuda(const deviceGaugeField g_in) {

  // create new quda gauge param
  QudaGaugeParam gauge_param = newQudaGaugeParam();

  // define the gauge field parameters
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.location = QUDA_CUDA_FIELD_LOCATION;

  gauge_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec_precondition = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_eigensolver = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;
  gauge_param.anisotropy = 1.0;
  gauge_param.tadpole_coeff = 1.0;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  // set the dimensions of the gauge field
  gauge_param.X[0] = g_in.view.extent(0);
  gauge_param.X[1] = g_in.view.extent(1);
  gauge_param.X[2] = g_in.view.extent(2);
  gauge_param.X[3] = g_in.view.extent(3);

  // For multi-GPU, ga_pad must be large enough to store a time-slice
  // should be set even while using single gpu, as long as QUDA
  // is compiled with multi-GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;

  deviceGaugeField_quda g_q(g_in);

  void *gauge[4];

  for(int dir = 0; dir < Nd; dir++) {
    gauge[dir] = g_q.view[dir].data();
  }

  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

}

int parse_args(int argc, char **argv, StreamIndex &stream_array_size, StreamIndex &seed) {
  // Defaults
  stream_array_size = 32;
  seed = 1234;

  const std::string help_string =
      "  -n <N>, --nelements <N>\n"
      "     Create stream views containing [4][Nc][Nc]<N>^4 elements.\n"
      "     Default: 32\n"
      "  -s <seed>, --seed <seed>\n"
      "     Seed for the random number generator.\n"
      "     Default: 1234\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"nelements", required_argument, NULL, 'n'},
      {"seed", required_argument, NULL, 's'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "n:s:h", long_options, &option_index)) !=
         -1)
    switch (c) {
      case 'n': stream_array_size = atoi(optarg); break;
      case 's': seed = atoi(optarg); break;
      case 'h':
        printf("%s", help_string.c_str());
        return -2;
        break;
      case 0: break;
      default:
        printf("%s", help_string.c_str());
        return -1;
        break;
    }
  return 0;
}

int main(int argc, char *argv[]) {
  printfQuda(HLINE);
  printfQuda("Plaquette comparison between Kokkos and QUDA\n");
  printfQuda(HLINE);

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int comms[4] = {1, 1, 1, 1};
  // initialize comms
  initCommsGridQuda(4, comms, NULL, NULL);
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);
  int rc;
  StreamIndex stream_array_size;
  StreamIndex seed;
  rc = parse_args(argc, argv, stream_array_size, seed);
  if (rc == 0) {
    // initialize quda
    initQuda(-1);

    // set verbosity
    // set to QUDA_DEBUG_VERBOSE for debugging
    setVerbosity(QUDA_SILENT);

    // initialize RNG
    RNG rng(seed);

    // create gauge field
    deviceGaugeField g(stream_array_size, stream_array_size, stream_array_size,
                       stream_array_size, rng);

    // load gauge to quda
    _loadGaugeQuda(g);

    real_t plaq[3];
    val_t plaq_kokkos;

    // compute plaquette in QUDA
    plaqQuda(plaq);

    // compute plaquette in Kokkos
    plaq_kokkos = perform_plaquette(g);

    printfQuda("Computed plaquette gauge in QUDA is %16.15e (spatial = %16.15e, temporal = %16.15e)\n", plaq[0], plaq[1],
               plaq[2]);

    real_t fac = stream_array_size*stream_array_size*stream_array_size*stream_array_size*Nc*Nd*(Nd-1)/2.0;
    printfQuda("Computed plaquette gauge in kokkos is %16.15e\n", plaq_kokkos.real()/fac);

  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();


}