#!/bin/bash
# Cluster user and paths
USERNAME="matthies"
CLUSTER_HOST="data.bridges2.psc.edu"
PROJECT_PATH="/ocean/projects/mch240013p/$USERNAME"

LOCAL_UPLOAD_PATH=$HOME/Documents/wheel_sim_pipeline
REMOTE_UPLOAD_PATH="/jet/home/$USERNAME/moonranger_mobility"

LOCAL_DOWNLOAD_PATH="$HOME/Documents/wheel_sim_data"

JOB_SCRIPT="/jet/home/$USERNAME/moonranger_mobility/bash_scripts/automation_test.sh"

upload() {
wheel_folder="wheel_$1"
wheel_generation
echo "Uploading $LOCAL_UPLOAD_PATH/wheel.obj and $LOCAL_UPLOAD_PATH/whee_jsons/wheel_parameters.json to $REMOTE_UPLOAD_PATH"
ssh -i $HOME/Documents/wheel_sim_pipeline/key_pair_psc "$USERNAME@bridges2.psc.edu" "mkdir -p /jet/home/$USERNAME/moonranger_mobility/meshes/$wheel_folder/"
scp -r -i $LOCAL_UPLOAD_PATH/key_pair_psc "$LOCAL_UPLOAD_PATH/wheel.obj" "$USERNAME@$CLUSTER_HOST:$REMOTE_UPLOAD_PATH/meshes/$wheel_folder/"
scp -r -i $LOCAL_UPLOAD_PATH/key_pair_psc "$LOCAL_UPLOAD_PATH/wheel_jsons/wheel_parameters.json" "$USERNAME@$CLUSTER_HOST:$REMOTE_UPLOAD_PATH/meshes/$wheel_folder/"
echo "uploading to $USERNAME@$CLUSTER_HOST:$REMOTE_UPLOAD_PATH/meshes/$wheel_folder/"
echo "running simulation"
job_id=$(ssh -i $HOME/Documents/wheel_sim_pipeline/key_pair_psc "$USERNAME@bridges2.psc.edu" "sbatch $JOB_SCRIPT $REMOTE_UPLOAD_PATH/job_json/job_parameters.json" | awk '{print $4}')
echo "job id:$job_id"
}

download() {
job_id_arg="$1"
REMOTE_DOWNLOAD_PATH="$PROJECT_PATH/automation_test_$job_id_arg"
echo "Downloading from $REMOTE_DOWNLOAD_PATH to $LOCAL_DOWNLOAD_PATH"
scp -r -i $HOME/Documents/wheel_sim_pipeline/key_pair_psc "$USERNAME@$CLUSTER_HOST:$REMOTE_DOWNLOAD_PATH" "$LOCAL_DOWNLOAD_PATH"
echo "Downloaded File $PROJECT_PATH/automation_test_$job_id_arg"
}

wheel_generation() {
echo "generating wheel"
python gen_wheel.py -w wheel_jsons/wheel_parameters.json
echo "wheel generated"
}

# MAIN
case "$1" in
upload)
upload $2
;;
download)
download $2
;;
test)
echo "working"
;;
*)
echo "Usage: $0 {upload|download}"
;;
esac
