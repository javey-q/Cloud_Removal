opt_path=$1 # Final_Rsipac_CR
cd Cloud_Removal
bash scripts/prepare_data.sh
accelerate launch --config_file ./options/card.yaml  train.py  --opt ./options/$opt_path
python test.py --opt ./options/test/$opt_path
python scripts/test_merge.py --opt  $opt_path

zip -r ./infer/results.zip ./infer/results
cp ./infer/results.zip /output