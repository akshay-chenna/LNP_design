cd mean_classification_chem
nohup bash run.sh &
wait
cd ..

cd sum_classification_chem
nohup bash run.sh &
wait
cd ..

cd mean_classification_nochem
nohup bash run.sh &
wait
cd ..

cd sum_classification_nochem
nohup bash run.sh &
wait
cd ..
