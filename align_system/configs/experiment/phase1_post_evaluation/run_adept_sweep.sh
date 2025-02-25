exp=phase1_post_evaluation/aligned_adm_adept_eval
adm=ALIGN-ADM-ComparativeRegression
backbone=EXAONE-3.5-32B-Instruct # Use backbone name from yaml file

for target in 0.2 0.3 0.4 0.5 0.6 0.7 0.8
do
HYDRA_FULL_ERROR=1 run_align_system +experiment=${exp} +alignment_target="ADEPT-DryRun-Moral judgement-${target}" hydra.run.dir="/data/shared/new_backbones/phase1_post_eval_local/${adm}-${backbone}-ADEPT/Moral-judgement-${target}"
done

for target in 0.2 0.3 0.4 0.5 0.6 0.7 0.8
do
HYDRA_FULL_ERROR=1 run_align_system +experiment=${exp} +alignment_target="ADEPT-DryRun-Ingroup Bias-${target}" hydra.run.dir="/data/shared/new_backbones/phase1_post_eval_local/${adm}-${backbone}-ADEPT/Ingroup-Bias-${target}"
done
