#!/bin/bash
set -eo pipefail
base_dir="$(dirname $0)/.."
include_path="$(vivado_hls -l /dev/null -root_dir|tail -n2|head -n1)/include"
for file in "${base_dir}/tests/src/"*.soda
do
  echo -n "Compiling $(basename "${file}") ..."
  "${base_dir}/src/sodac" "${file}" --xocl-kernel - $@ \
    | g++ -x c++ -std=c++11 -fsyntax-only "-I${include_path}" -c -
  echo " PASS"
done
