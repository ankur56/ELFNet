#!/usr/bin/env bash

declare -a prop3d_code=(8 9 10 12)

for i in "${prop3d_code[@]}"
do 
  sed -e "2s/1/$i/" gen_3d.txt > gen_3d_${i}.txt
  for j in *.wfx
  do
    fname="${j%.*}"
    Multiwfn $j < gen_3d_${i}.txt > null
    new_file="${fname}_k14_gr52_${i}.txt"
    mv output.txt $new_file
    sed -i -r -e 's/\s+/,/g' $new_file
    sed -i -r -e 's/^,//g' $new_file
  done
done

