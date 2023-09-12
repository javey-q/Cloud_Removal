# train
opt_path=$1 # Final_NAF_Middle_ID_Multi.yml
bash prepare_data.sh
accelerate launch train.py  --opt ./options/$opt_path