PYTHONPATH='./' python mmbt/train.py --batch_sz 64 --gradient_accumulation_steps 40 \
 --savedir ./save/ --name mmbt_vsnli_train_3 \
 --img_model 'resnet152' \
 --data_path  ./input/ \
 --use_tsv 1 \
 --multiGPU 1 --fp16 1 \
 --task vsnli --task_type classification \
 --model mmbt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3  \
 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 20 --seed 1
