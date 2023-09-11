# test
opt_path=$1 # Final_NAF_Middle_ID_Multi.yml
python test.py --opt ./options/test/$opt_path
python scripts/test_merge.py --opt  $opt_path
cd infer
zip -q -r results.zip ./results
cp results.zip /output