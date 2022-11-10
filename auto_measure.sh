#!/bin/bash

for (( i=1; i<=100; i++ ))
do
  echo "Measure CSI Motions: $i"
  ./log_to_file "test$i.dat"

  echo "sleep 1sec"
  sleep 1
done

echo "Finish"
