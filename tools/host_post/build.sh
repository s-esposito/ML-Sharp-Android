#!/bin/sh
# Builds tools/host_post/host_post from the app's C core (macOS/Linux, clang or gcc).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CORE="$HERE/../../app/src/main/cpp/core"
INC="$HERE/../../app/src/main/cpp/include"
CC=${CC:-cc}
CXX=${CXX:-c++}
OBJ="$HERE/.obj"; mkdir -p "$OBJ"
for f in normalizer pyramid split_patches merge_patches depth_from_disparity initializer \
         composer unproject save_ply exif pipeline_core; do
  $CC -O2 -std=c11 -I"$INC" -I"$CORE" -c "$CORE/$f.c" -o "$OBJ/$f.o"
done
$CXX -O2 -std=c++17 -I"$INC" -I"$CORE" -c "$CORE/prep_input.cpp" -o "$OBJ/prep_input.o"
$CC -O2 -std=c11 -I"$INC" -c "$HERE/host_post.c" -o "$OBJ/host_post.o"
$CXX -o "$HERE/host_post" "$OBJ"/*.o -lm
echo "built $HERE/host_post"
