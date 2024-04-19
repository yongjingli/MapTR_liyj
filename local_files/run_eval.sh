cd ../
#export CUDA_VISIBLE_DEVICES=1,2
export CUDA_VISIBLE_DEVICES=0
echo "Start Eval"

./tools/dist_test_map.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py ./ckpts/maptrv2_nusc_r50_24e.pth 1