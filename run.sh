opt_path=$1 # Final_NAF_Middle_ID_Multi.yml
#cd Cloud_Removal
timeout 85200 bash train.sh $opt_path
bash test.sh  $opt_path