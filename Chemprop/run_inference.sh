mkdir logs
mv run*txt logs/.
mv *.png logs/.
for i in */best* ; do x=`echo $i | cut -d / -f 1` ; python inference_testdata.py -f $i -n $x &> ${x}.txt & done
wait
mv lightning_logs logs/.
