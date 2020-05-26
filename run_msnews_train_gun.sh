PYTHONPATH='./' python mmbt/train.py --batch_sz 64 --gradient_accumulation_steps 40 \
 --savedir ./save/ --name mmbt_msnews_train_from_zero \
 --img_model 'resnet152' --img_path 'msnews-img' \
 --data_path  ./input/ \
 --use_tsv 1 \
 --multiGPU 1 --fp16 1 \
 --no_cuda 1 \
 --task msnews --task_type classification \
 --model mmbt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3  \
 --patience 5 --dropout 0.1 --lr 3e-05 --warmup 0.1 --max_epochs 50 --seed 1
