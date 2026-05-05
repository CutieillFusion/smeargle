config_path="config.json"
model_path=$1
save_dir=$2

mkdir -p ../models/$save_dir
cp $config_path ../models/$save_dir/config.json
cp models/$model_path/best_model/draft_model.pt ../models/$save_dir/draft_model.pt