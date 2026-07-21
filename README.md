

### pretraining 

# JEPA (unchanged behaviour, new filename)
python source_pretraining.py --method jepa --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
  --mode cross_domain_openset --train_spectrums WHT --output_dir ./output_jepa

# CompNet supervised
python source_pretraining.py --method compnet --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
  --mode cross_domain_openset --train_spectrums WHT --output_dir ./output_compnet

nohup python source_pretraining.py --method compnet --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI --mode cross_domain_openset --train_spectrums WHT --output_dir ./output_compnet --aug_multiplier 8 > logs/casiams_compnet_crossdomain_openset_WHT_8X.log 2>&1 &


### subspace analysis
subspace_analysis.py :
python subspace_analysis.py \
  --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI \
  --ckpt ./output_jepa/ckpt_source_CASIA-MS-ROI_cross_domain_WHT-940_8x.pth \
  --source_spectrum WHT --target_spectrum 940

Characterizes the source model's feature subspace from a trained JEPA checkpoint, offline and without further training. Given a checkpoint and the dataset directory, it rebuilds the frozen encoder, builds the source feature covariance C0=UΛU⊤C_0 = U\Lambda U^\top
C0​=UΛU⊤ from the gallery split, and runs four analyses: retention (keep top-kk
k directions → find k⋆k^\star
k⋆, the smallest subspace preserving identity matching, and hence the free capacity available to NS-CTTA), ablation (remove top-NN
N → exposes high-energy directions that carry no identity information), attribution (per-direction Fisher ratios against identity vs. spectrum labels → separates biometric signal from illumination/sensor nuisance), and target complement (projects an unseen spectrum into S0\mathcal{S}_0
S0​ and its complement → tests whether the freed directions are usable for adaptation). Outputs four figures and a JSON dump of all numbers.


### source subspace approximation

python subspace_reconstruct.py   --data_dir /home/pai-ng/Jamal/CASIA-MS-ROI   --ckpt ./output_jepa/ckpt_source_CASIA-MS-ROI_cross_domain_openset_WHT_8x.pth   --source_spectrum WHT --target_dataset cifar   --out_dir ./output_reconstruct/CIFAR


