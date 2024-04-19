cd ../
#python tools/maptr/vis_pred.py /path/to/experiment/config /path/to/experiment/ckpt

#All the visualization samples will be saved in /path/to/MapTR/work_dirs/experiment/vis_pred/ automatically.
#If you want to customize the saving path, you can add --show-dir /customized_path
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=${SCRIPT_DIR}

echo $ROOT_DIR
export PYTHONPATH=$ROOT_DIR

python tools/maptr/vis_pred.py ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py ./ckpts/maptrv2_nusc_r50_24e.pth
