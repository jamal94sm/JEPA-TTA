nohup python main.py --mode cross_domain_openset --data_dir "/home/pai-ng/Jamal/CASIA-MS-ROI" --train_spectrums WHT --aug_multiplier 8 > logs/casiams_crossdomain_openset_WHT_8X.log 2>&1 &



subspace_analysis.py :

Characterizes the source model's feature subspace from a trained JEPA checkpoint, offline and without further training. Given a checkpoint and the dataset directory, it rebuilds the frozen encoder, builds the source feature covariance C0=UΛU⊤C_0 = U\Lambda U^\top
C0​=UΛU⊤ from the gallery split, and runs four analyses: retention (keep top-kk
k directions → find k⋆k^\star
k⋆, the smallest subspace preserving identity matching, and hence the free capacity available to NS-CTTA), ablation (remove top-NN
N → exposes high-energy directions that carry no identity information), attribution (per-direction Fisher ratios against identity vs. spectrum labels → separates biometric signal from illumination/sensor nuisance), and target complement (projects an unseen spectrum into S0\mathcal{S}_0
S0​ and its complement → tests whether the freed directions are usable for adaptation). Outputs four figures and a JSON dump of all numbers.


