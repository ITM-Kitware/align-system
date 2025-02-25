exp=phase1_post_evaluation/aligned_adm_soartech_eval
adm=ALIGN-ADM-ComparativeRegression
backbone=EXAONE-3.5-32B-Instruct # Use backbone name from yaml file

for target in 'qol-group-target-ph1-1' 'qol-group-target-ph1-2' 'qol-group-target-train-1-ph1' 'qol-human-0000001-SplitEvenMulti-ph1' 'qol-human-3043871-SplitHighBinary-ph1' 'qol-human-5032922-SplitLowMulti-ph1' 'qol-human-6403274-SplitHighBinary-ph1' 'qol-human-7040555-SplitHighMulti-ph1' 'qol-human-8022671-SplitLowMulti-ph1' 'qol-synth-HighCluster-ph1' 'qol-synth-HighExtreme-ph1' 'qol-synth-LowCluster-ph1' 'qol-synth-LowExtreme-ph1'
do
    for train in 1 2 3 4
    do
        HYDRA_FULL_ERROR=1 run_align_system +experiment=${exp} +alignment_target=$target hydra.run.dir=/data/shared/new_backbones/phase1_post_eval_local/${adm}-${backbone}-Soartech/qol_train${train}_sweep/$target/ interface.scenario_ids=[qol-ph1-train-${train}]
    done
done

for target in 'vol-group-target-ph1-1' 'vol-group-target-ph1-2' 'vol-group-target-train-1-ph1' 'vol-human-1774519-SplitHighMulti-ph1' 'vol-human-5032922-SplitLowMulti-ph1' 'vol-human-6403274-SplitEvenBinary-ph1' 'vol-human-8022671-SplitHighMulti-ph1' 'vol-human-8478698-SplitLowMulti-ph1' 'vol-synth-HighCluster-ph1' 'vol-synth-LowCluster-ph1' 'vol-synth-LowExtreme-ph1'
do
    for train in 1 2 3 4
    do
        HYDRA_FULL_ERROR=1 run_align_system +experiment=${exp} +alignment_target=$target hydra.run.dir=/data/shared/new_backbones/phase1_post_eval_local/${adm}-${backbone}-Soartech/vol_train${train}_sweep/$target/ interface.scenario_ids=[vol-ph1-train-${train}]
    done
done
