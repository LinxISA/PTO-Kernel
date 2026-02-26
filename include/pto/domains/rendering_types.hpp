#ifndef PTO_DOMAINS_RENDERING_TYPES_HPP
#define PTO_DOMAINS_RENDERING_TYPES_HPP

// Rendering-oriented type conventions for PTO kernels.
//
// Principle: tiles are generic 4KB carriers. Rendering defines conventions
// (packing + SoA structures) that can evolve without changing the core tile type.

#include <common/pto_tileop.hpp>
#include <cstdint>

namespace pto::domains::rendering {

// Canonical micro-batch shape used by early rendering kernels:
// - single-column layout: 1024 x 1 elements
// - 4 bytes/elem => exactly 4096B per tile
//
// Rationale:
// - matches the physical 4KB RawTile carrier cleanly for 32-bit element formats
// - keeps the data structure "list-like" and avoids binding semantics to screen-space tiling
inline constexpr int kTileW = 1;
inline constexpr int kTileH = 1024;

// Pixel-format conventions (initial):
// - RGBA8: packed little-endian 0xAABBGGRR in a uint32_t per element
// - Depth32F: float per element (stored as a separate tile)

using TileU32 = pto::Tile<pto::Location::Vec, uint32_t, kTileH, kTileW, pto::BLayout::RowMajor>;
using TileF32 = pto::Tile<pto::Location::Vec, float,    kTileH, kTileW, pto::BLayout::RowMajor>;

struct FragmentBatchSOA {
  // Coordinates / depth (as needed)
  TileF32 x;
  TileF32 y;
  TileF32 z;

  // Optional coverage/mask (0 or 0xffffffff per element by convention)
  TileU32 mask;

  // Color in float SoA (shader output)
  TileF32 r;
  TileF32 g;
  TileF32 b;
  TileF32 a;
};

// Mapping policy for how (row,col) corresponds to logical pixels/fragments.
//
// - ScreenTile: element (r,c) corresponds to pixel (tile_origin_x + c, tile_origin_y + r).
// - List: element i=r*kTileW+c is just a slot; x/y/z/mask carry explicit coordinates.
enum class MappingPolicy {
  // element (r,c) corresponds to pixel (tile_origin_x + c, tile_origin_y + r)
  ScreenTile,
  // element i=r*kTileW+c is just a slot; x/y/z/mask carry explicit coordinates
  List,
};

// Default for early bring-up: list-style batches built by CPU/BCC software logic.
inline constexpr MappingPolicy kDefaultMapping = MappingPolicy::List;

} // namespace pto::domains::rendering

#endif // PTO_DOMAINS_RENDERING_TYPES_HPP
