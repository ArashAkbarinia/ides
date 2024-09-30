# ides

You can download the relevant THINGS-EEG data set and THINGS-MEG data set at https://osf.io/3jk45/ .

The raw and preprocessed EEG dataset, the training and test images and the DNN feature maps are available on osf.

To reproduce our results, please run:

```
for i in {1..10}; do \ 
  python decoder_things_eeg2_transformer.py --eeg_dir EEG2/ --subject $i --print_frequency 1 \
  --out_dir results/inter_subjects/ --epochs 40 --num_layers 8 --dropout 0.1 --hidden_dim 128 \
  --lr 3e-4 --batch_size 256 --experiment_name _new_baseline --optimiser AdamW --val_imgs 9 \
  --num_workers 4 --arch atm --feature_dir ViT-B-16__laion400m_e31__visual-language__1024/ \ 
  --avg_concept ; \
  done

```