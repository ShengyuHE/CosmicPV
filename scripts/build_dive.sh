#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
module load GCC/12.3.0 CGAL/5.6 CMake/3.26.3 GMP/6.2.1 MPFR/4.2.0 Boost/1.82.0
mkdir -p build/dive-source
VENDOR_DIR=vendor
PATCH_DIR=patches
if [[ ! -d "$VENDOR_DIR" ]]; then VENDOR_DIR=logs/vendor; fi
if [[ ! -d "$PATCH_DIR" ]]; then PATCH_DIR=logs/patches; fi
cp "$VENDOR_DIR/DIVE/DIVE.cpp" "$VENDOR_DIR/DIVE/CMakeLists.txt" build/dive-source/
patch -d build/dive-source -p1 < "$PATCH_DIR/dive-parity.patch"
cmake -S build/dive-source -B build/dive -DCMAKE_BUILD_TYPE=Release \
  -DGMP_INCLUDE_DIR="$EBROOTGMP/include" -DGMP_LIBRARIES="$EBROOTGMP/lib/libgmp.so" \
  -DMPFR_INCLUDE_DIR="$EBROOTMPFR/include" -DMPFR_LIBRARIES="$EBROOTMPFR/lib/libmpfr.so" \
  -DBOOST_ROOT="$EBROOTBOOST"
cmake --build build/dive --parallel 2
