#!/bin/bash

for dir in $(ls ../data | sed "s@.*/@@g")
do
python3 Segmentation.py ../data/"$dir" | tee ./"logs/segmentation_$dir.log"
done
