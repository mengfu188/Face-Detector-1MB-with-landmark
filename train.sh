python3 -u train.py --save_folder checkpoint/RFB-RAW/ \
--num_workers 6 --resume_net checkpoint/RFB-RAW.bak/RFB_Final.pth \
--training_dataset data/widerface-mask/train/label.txt \
--resume_epoch 65 \
--lr 0.00001 \
2>&1 | tee checkpoint/RFB-RAW/log