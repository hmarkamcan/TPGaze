# Gaze360 Pretrain and Test
python3 main.py --train_status=train --train_loader_flag=Gaze360 --epochs=50 \
    --model_save_dir="${OUTPUT_DIR}/Train" --data_dir=$gaze360_dir \
    --resnet_model_path=$resnet_dir

python3 main.py --train_status=test --val_loader_flag=EyeDiap --epochs=50 \
    --test_data_dir=$eyediap_dir \
    --resnet_model_path=$resnet_dir --pre_trained_model_path="${OUTPUT_DIR}/Train"
    
python3 main.py --train_status=test -val_loader_flag=MPII --epochs=50 \
    --test_data_dir=$mpiiface_dir \
    --resnet_model_path=$resnet_dir --pre_trained_model_path="${OUTPUT_DIR}/Train"

# Gaze360 Meta Train
python3 main.py --train_status=meta_train --train_loader_flag=Gaze360 --val_loader_flag=EyeDiap --batch_size=20 \
    --model_save_dir="${OUTPUT_DIR}/Training_Meta/eyediap" \
    --data_dir=$gaze360_dir --test_data_dir=$eyediap_dir \
    --resnet_model_path=$resnet_dir --pre_trained_model_path="${pretrained_dir}/Train" --pre_trained_model_name="epoch_2_ckpt.pth.tar"

python3 main.py --train_status=meta_train --train_loader_flag=Gaze360 --val_loader_flag=MPII --batch_size=20 \
    --model_save_dir="${OUTPUT_DIR}/Training_Meta/mpiiface" \
    --data_dir=$gaze360_dir --test_data_dir=$mpiiface_dir \
    --resnet_model_path=$resnet_dir --pre_trained_model_path="${pretrained_dir}/Train" --pre_trained_model_name="epoch_4_ckpt.pth.tar"

# Gaze360 Persona
python3 main.py --train_status=persona --val_loader_flag=EyeDiap \
    --test_data_dir=$eyediap_dir \
    --resnet_model_path=$resnet_dir --pre_trained_model_path="${pretrained_dir}/Training_Meta/eyediap" --pre_trained_model_name="meta_model_epoch_0_ckpt.pth.tar" 

python3 main.py --train_status=persona --val_loader_flag=MPII \
    --test_data_dir=$mpiiface_dir \
    --resnet_model_path=$resnet_dir --pre_trained_model_path="${pretrained_dir}/Training_Meta/mpiiface" --pre_trained_model_name="meta_model_epoch_0_ckpt.pth.tar"

