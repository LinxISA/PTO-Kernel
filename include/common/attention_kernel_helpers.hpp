#ifndef PTO_COMMON_ATTENTION_KERNEL_HELPERS_HPP
#define PTO_COMMON_ATTENTION_KERNEL_HELPERS_HPP

#include <common/linx_lowp_types.hpp>
#include <common/pto_tileop.hpp>

namespace pto {
namespace kernels {

template <int N>
inline void dequant_fp16_array(const fp16_t *src, float *dst) {
  for (int i = 0; i < N; ++i)
    dst[i] = fp16_to_float(src[i]);
}

template <int N>
inline void quant_fp16_array(const float *src, fp16_t *dst) {
  for (int i = 0; i < N; ++i)
    dst[i] = float_to_fp16(src[i]);
}

template <int N>
inline void dequant_fp8_array(const fp8_e4m3_t *src, float *dst) {
  for (int i = 0; i < N; ++i)
    dst[i] = fp8_e4m3_to_float(src[i]);
}

template <int N>
inline void quant_fp8_array(const float *src, fp8_e4m3_t *dst) {
  for (int i = 0; i < N; ++i)
    dst[i] = float_to_fp8_e4m3(src[i]);
}

template <int N>
inline void dequant_fp4_array(const fp4_e2m1_t *src, float *dst) {
  for (int i = 0; i < N; ++i)
    dst[i] = fp4_e2m1_to_float(src[i]);
}

template <int N>
inline void quant_fp4_array(const float *src, fp4_e2m1_t *dst) {
  for (int i = 0; i < N; ++i)
    dst[i] = float_to_fp4_e2m1(src[i]);
}

template <int S, int D, int VD, int TM, int TK, bool Causal = false>
inline void flash_attention_f32(float *out_ptr, float *q_ptr, float *k_ptr,
                                float *v_ptr, float scale) {
  static_assert(S > 0 && D > 0 && VD > 0 && TM > 0 && TK > 0,
                "shape params must be positive");
  static_assert((TM * TK * D * static_cast<int>(sizeof(float))) <= 4096,
                "QK footprint must fit <=4KB");
  static_assert((TM * TK * VD * static_cast<int>(sizeof(float))) <= 4096,
                "PV footprint must fit <=4KB");
  static_assert(S % TM == 0 && S % TK == 0, "S must be divisible by TM/TK");

  using gmQ = global_tensor<float, RowMajor<S, D>>;
  using gmK = global_tensor<float, ColMajor<D, S>>;
  using gmV = global_tensor<float, ColMajor<S, VD>>;
  using gmO = global_tensor<float, RowMajor<S, VD>>;

  using tileQ = TileLeft<float, TM, D>;
  using tileK = TileRight<float, D, TK>;
  using tileV = TileRight<float, TK, VD>;
  using tileWOut = TileAcc<float, TM, TK>;
  using tileW = Tile<Location::Vec, float, TM, TK, BLayout::RowMajor>;
  using tileWLeft = TileLeft<float, TM, TK>;
  using tileOutAcc = TileAcc<float, TM, VD>;
  using tileOut = Tile<Location::Vec, float, TM, VD, BLayout::RowMajor>;
  using tileMax = Tile<Location::Vec, float, TM, 1, BLayout::RowMajor>;
  using tileSum = Tile<Location::Vec, float, TM, 1, BLayout::RowMajor>;

  using itQ = global_iterator<gmQ, tileQ>;
  using itK = global_iterator<gmK, tileK>;
  using itV = global_iterator<gmV, tileV>;
  using itO = global_iterator<gmO, tileOut>;

  itQ gQ(q_ptr);
  itK gK(k_ptr);
  itV gV(v_ptr);
  itO gO(out_ptr);

  constexpr int kQTiles = S / TM;
  constexpr int kKTiles = S / TK;

  for (int qi = 0; qi < kQTiles; ++qi) {
    tileQ tQ;
    TLOAD(tQ, gQ(qi, 0));

    tileMax tMax;
    tileSum tSum(0.0f);
    tileOutAcc tOutAcc(0.0f);
    tileOut tOut(0.0f);
    TEXPANDS(tMax, -1e30f);

    for (int kj = 0; kj < kKTiles; ++kj) {
      if constexpr (Causal) {
        if (kj > (qi * TM) / TK)
          break;
      }
      tileK tK;
      tileV tV;
      TLOAD(tK, gK(0, kj));
      TLOAD(tV, gV(kj, 0));

      tileWOut tWOut;
      tileW tW;
      TMATMUL(tWOut, tQ, tK);
      TCVT(tW, tWOut);
      TMULS(tW, tW, scale);

      tileMax tLocalMax;
      tileMax tNewMax;
      TROWMAX(tLocalMax, tW);
      TMAX(tNewMax, tMax, tLocalMax);

      tileSum tScaleOld;
      tileSum tScaledSum;
      TSUB(tScaleOld, tMax, tNewMax);
      TEXP(tScaleOld, tScaleOld);
      TMUL(tScaledSum, tSum, tScaleOld);

      tileW tMaxExpanded;
      TEXPANDCOL(tMaxExpanded, tNewMax);
      TSUB(tW, tW, tMaxExpanded);
      TEXP(tW, tW);

      tileSum tLocalSum;
      TROWSUM(tLocalSum, tW);
      TADD(tSum, tScaledSum, tLocalSum);

      tileOut tScaleExpanded;
      TEXPANDCOL(tScaleExpanded, tScaleOld);
      TMUL(tOut, tOut, tScaleExpanded);

      tileWLeft tWLeft;
      TCVT(tOutAcc, tOut);
      TCVT(tWLeft, tW);
      MATMACC(tOutAcc, tWLeft, tV);
      TCVT(tOut, tOutAcc);
      tMax = tNewMax;
    }

    tileSum tInvSum;
    tileOut tInvExpanded;
    TRECIP(tInvSum, tSum);
    TEXPANDCOL(tInvExpanded, tInvSum);
    TMUL(tOut, tOut, tInvExpanded);
    TSTORE(gO(qi, 0), tOut);
  }
}

template <int M, int N, int K, int TM, int TN, int TK, bool AddToDst = false>
inline void gemm_f32(float *a_ptr, float *b_ptr, float *c_ptr) {
  static_assert(M > 0 && N > 0 && K > 0, "shape params must be positive");
  static_assert(M % TM == 0 && N % TN == 0 && K % TK == 0,
                "global shapes must be divisible by tile shapes");
  static_assert((TM * TN * TK * static_cast<int>(sizeof(float))) <= 4096,
                "TMATMUL footprint must fit <=4KB");

  using tileA = TileLeft<float, TM, TK>;
  using tileB = TileRight<float, TK, TN>;
  using tileAcc = TileAcc<float, TM, TN>;
  using tileVec = Tile<Location::Vec, float, TM, TN, BLayout::RowMajor>;
  using gmA = global_tensor<float, RowMajor<M, K>>;
  using gmB = global_tensor<float, ColMajor<K, N>>;
  using gmC = global_tensor<float, RowMajor<M, N>>;
  using itA = global_iterator<gmA, tileA>;
  using itB = global_iterator<gmB, tileB>;
  using itC = global_iterator<gmC, tileVec>;

  itA gA(a_ptr);
  itB gB(b_ptr);
  itC gC(c_ptr);

  constexpr int kMT = M / TM;
  constexpr int kNT = N / TN;
  constexpr int kKT = K / TK;

  for (int mi = 0; mi < kMT; ++mi) {
    for (int nj = 0; nj < kNT; ++nj) {
      tileA a0;
      tileB b0;
      TLOAD(a0, gA(mi, 0));
      TLOAD(b0, gB(0, nj));
      tileAcc acc;
      TMATMUL(acc, a0, b0);

      for (int kk = 1; kk < kKT; ++kk) {
        tileA a;
        tileB b;
        TLOAD(a, gA(mi, kk));
        TLOAD(b, gB(kk, nj));
        TMATMUL_ACC(acc, acc, a, b);
      }

      tileVec out;
      TCVT(out, acc);
      if constexpr (AddToDst) {
        tileVec prev;
        tileVec merged;
        TLOAD(prev, gC(mi, nj));
        TADD(merged, prev, out);
        TSTORE(gC(mi, nj), merged);
      } else {
        TSTORE(gC(mi, nj), out);
      }
    }
  }
}

} // namespace kernels
} // namespace pto

#endif // PTO_COMMON_ATTENTION_KERNEL_HELPERS_HPP
