for i in 50 500 1000 2000
do
   for j in 1 2
   do
       echo "Num of frames: $i, num of RNN:  $j is running."
       python keep.py $i $j 8 > "./snn_output/out_${i}_${j}.txt"
   done

done
