cd ../
export CUDA_VISIBLE_DEVICES=1,2
#./tools/dist_train.sh ./projects/configs/maptr/maptr_tiny_r50_24e.py 8
./tools/dist_train.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py 2
