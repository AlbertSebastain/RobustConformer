#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

source ./config/config.txt
source ./utils/yaml_to_shell.sh
create_variables ./config/train_config.yml
# set path

dictroot=${dataroot}/lang # to save the vocabulary dictionary
dict=${dictroot}/${train_set}_units.txt
lmexpdir=${checkpoints_dir}/train_rnnlm
if [ ! -f config/path_config ]; then
    touch config/path_config
fi
echo "checkpoints_dir=${checkpoints_dir}" > config/path_config
echo "dataroot=${dataroot}" >> config/path_config
echo "exp_root=${exp_root}" >> config/path_config
echo "dictroot=${dictroot}" >> config/path_config
echo "lmexpdir=${lmexpdir}" >> config/path_config
echo "dict=${dict}" >> config/path_config
echo "embed_init_file=${embed_init_file}" >> config/path_config

mkdir -p ${dictroot}
train_file=${dataroot}/${train_set}
test_file=${dataroot}/${test_set}
dev_file=${dataroot}/${dev_set} 
# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh
# check gpu option usage
if [ ! -z $gpu ]; then
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',     --resume $resume \
set -e
set -u
set -o pipefail

if [ ${stage} -le 0 ]; then

    # Download AISHELL 1 dataset and prepare files
    data=${resource_root}
    data_url=www.openslr.org/resources/33

    #data/download_and_untar.sh ${data} ${data_url} data_aishell || exit 1;
    #data/download_and_untar.sh ${data} ${data_url} resource_aishell || exit 1;

    # Lexicon Preparation,
    data/aishell_prepare_dict.sh ${data}/resource_aishell ${dataroot} || exit 1;

    # Data Preparation,
    data/aishell_data_prep.sh ${data}/data_aishell/wav ${data}/data_aishell/transcript ${dataroot} || exit 1;

    # Phone Sets, questions, L compilation
    #data/utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
        #"<SPOKEN_NOISE>" data/local/lang data/lang || exit 1;

    # Download musan corpus.
    musan_data_root=${resource_root}/musan_corpus
    mkdir -p ${musan_data_root}
    #wget -P ${musan_data_root} https://www.openslr.org/resources/17/musan.tar.gz
    tar -xzvf ${musan_data_root}/musan.tar.gz -C ${musan_data_root}
    
    # Download noise92 dataset
    data_root=${resource_root}/noise92
    mkdir -p ${data_root}
    noise_names=("white" "pink" "factory1" "factory2" "buccaneer1" "buccaneer2" "f16" "destroyerengine" "destroyerops" "leopard" "m109" "machinegun" "volvo" "hfchannel")
    for noise_name in ${noise_names[@]}
    do
        echo "Down noise ${noise_name}"
        noise_name=${noise_name}.mat
        wget -P ${data_root} http://spib.linse.ufsc.br/data/noise/${noise_name}
    done

fi

# you can skip this and remove --rnnlm option in the recognition (stage 5)
nlsyms=${dictroot}/silence.txt
mkdir -p ${lmexpdir}
if [ ${stage} -le 1 ]; then
    echo "stage 2: LM Preparation"
    echo "sil" > ${nlsyms}
    echo "sil" > ${dictroot}/non_lang_syms.txt
    for x in train dev test; do
        cp ${dataroot}/${x}/text ${dataroot}/${x}/text.org
        paste -d " " <(cut -f 1 -d" " ${dataroot}/${x}/text.org) <(cut -f 2- -d" " ${dataroot}/${x}/text.org | tr -d " ") \
            > ${dataroot}/${x}/text
        cp ${dataroot}/${x}/text ${dataroot}/${x}/text_char
        cp ${dataroot}/${x}/text ${dataroot}/${x}/text_word
    #rm data/${x}/text.org
    done
    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 ${dataroot}/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}
    


    lmdatadir=${lmexpdir}/local/lm_train
    mkdir -p ${lmdatadir}

    text2token.py -s 1 -n 1 -l ${nlsyms} --space "" ${dataroot}/train/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/train_trans.txt

    cat ${lmdatadir}/train_trans.txt | tr '\n' ' ' > ${lmdatadir}/train.txt
    text2token.py -s 1 -n 1 -l  ${nlsyms} --space ""  ${dataroot}/${train_dev}/text | cut -f 2- -d" " | perl -pe 's/\n/ <eos> /g' \
        > ${lmdatadir}/valid.txt
    # use only 1 gpu
    if [ ${ngpu} -gt 1 ]; then
        echo "LM training does not support multi-gpu. signle gpu will be used."
    fi
    echo "train language model"
    ${cuda_cmd} ${lmexpdir}/train.log \
        python3 lm_train.py \
        --ngpu 1 \
        --input-unit ${input_unit_lm} \
        --lm-type ${lmtype} \
        --unit ${hidden_unit_lm} \
        --dropout-rate ${dropout_lm} \
        --embed-init-file ${embed_init_file} \
        --verbose 1 \
        --batchsize ${batchsize_lm} \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --dict ${dict}
    echo "LM finish"
fi

if [ ${stage} -le 2 ]; then
    echo "stage 2: feature preparation"
    data_type=("train" "test" "dev")
    # construct noisy clean training sets, noisy clean dev sets, noisy clean test sets
    noise_dir=${resource_root}/noise92
    for((index=0;index<=2;index++)); do
        echo "construct ${data_type[${index}]} features"
        python3 ./data/prep_features.py \
            --data_dir ${dataroot} \
            --RequireClean \
            --RequireMix \
            --ParellelNoisyNum 1 \
            --NoiseRepeatNum ${noise_repeat_num} \
            --DataType ${data_type[${index}]} \
            --SavedType ${data_type[${index}]} \
            --ThreadNum ${thread_num} \
            --LowSNR ${low_snr} \
            --HighSNR ${high_snr} \
            --step ${step} \
            --noise_dir ${noise_dir}

        
        mkdir -p ${dataroot}/noisy_${data_type[${index}]}
        ln -s ${dataroot}/${data_type[${index}]}/* ${dataroot}/noisy_${data_type[${index}]}/
        cp ${dataroot}/${data_type[${index}]}/clean_feats.scp ${dataroot}/${data_type[${index}]}/feats.scp
        cp  ${dataroot}/noisy_${data_type[${index}]}/mix_feats.scp  ${dataroot}/noisy_${data_type[${index}]}/feats.scp
    done
    sed 's/__mix0//g' ${dataroot}/noisy_test/mix_feats.scp > ${dataroot}/noisy_test/feats.scp
    # construct random-mix noisy training set
    mkdir -p ${dataroot}/train_random_mix/
    ln -s ${dataroot}/train/* ${dataroot}/train_random_mix/
    rm  -rf ${dataroot}/train_random_mix/mix*
    rm ${dataroot}/train_random_mix/db.scp
    rm ${dataroot}/train_random_mix/noise.scp
    rm ${dataroot}/train_random_mix/feats.scp
    python3 ./data/prep_features.py \
        --data_dir ${dataroot} \
        --RequireMix \
        --ParellelNoisyNum ${parallel_noises} \
        --NoiseRepeatNum ${noise_repeat_num} \
        --DataType train \
        --SavedType train_random_mix \
        --ThreadNum ${thread_num} \
        --LowSNR ${low_snr} \
        --HighSNR ${high_snr} \
        --step ${step} \
        --noise_dir ${noise_dir}
    # construct unmatch noisy test set
    mkdir -p ${dataroot}/noisy_test_musan/
    ln -s ${dataroot}/noisy_test/* ${dataroot}/noisy_test_musan/
    rm -rf ${dataroot}/noisy_test_musan/mix*
    rm ${dataroot}/noisy_test_musan/db.scp
    rm ${dataroot}/noisy_test_musan/noise.scp
    rm ${dataroot}/noisy_test_musan/feats.scp
    noise_dir=${resource_root}/musan_corpus/musan/noise/free-sound
    python3 ./data/prep_features.py \
            --data_dir ${dataroot} \
            --RequireMix \
            --ParellelNoisyNum 1 \
            --NoiseRepeatNum ${noise_repeat_num} \
            --DataType test \
            --SavedType noisy_test_musan \
            --ThreadNum ${thread_num} \
            --LowSNR ${low_snr} \
            --HighSNR ${high_snr} \
            --step ${step} \
            --noise_dir ${noise_dir}
    sed 's/__mix0//g' ${dataroot}/noisy_test_musan/mix_feats.scp > ${dataroot}/noisy_test_musan/feats.scp

    echo "feature prepareation finish"
fi


# train clean teacher
if [ ${stage} -le 3 ]; then
    echo "stage 3: train teacher model"
    mkdir -p ${checkpoints_dir}/${teacher_name}
    python3 asr_train_conf.py \
        --dataroot ${dataroot} \
        --name ${teacher_name} \
        --config_file ${config_file_asr} \
        --config_file ${config_file_general} \
        --batch-size ${teacher_batch_size} \
        --epochs ${teacher_epochs} \
        --dict_dir ${dictroot} \
        --train_folder "train" \
        --dev_folder "dev" \
        --print_freq ${teacher_print_freq} \
        --validate_freq ${teacher_validate_freq} \
        --rnnlm ${lmexpdir}/rnnlm.model.best \
        --exp_path ${exp_root}/${teacher_name} \
        --checkpoints_dir ${checkpoints_dir} \
        --works_dir ${exp_root}/${teacher_name}
    echo "train teacher model finish"
fi
if [ ${stage} -le 4 ]; then
    echo "stage 4: decoding for teacher model"
    echo "Decoding clean test set"
    bash ./recog_conformer.sh "recog_test_${teacher_name}" "${teacher_name}" "test"
    echo "Decoding match noisy test set"
    bash ./recog_conformer.sh "recog_noisy_test_${teacher_name}" "${teacher_name}" "noisy_test"
    echo "Deocing unmatch noisy test set"
    bash ./recog_conformer.sh "recog_noisy_test_musan_${teacher_name}" "${teacher_name}" "noisy_test_musan"
    echo "teacher model recognition finish"
fi
if [ ${stage} -le 5 ]; then
    echo "stage 5: train model with data augmentation and adversarial samples"
    if [ ${da_use_random_mix_noise} == "True" ]; then
        train_name="train_random_mix"
    else
        train_name="noisy_train"
    fi
    mkdir -p ${checkpoints_dir}/${da_name}
    python3 asr_train_conf.py \
        --dataroot ${dataroot} \
        --name ${da_name} \
        --config_file ${config_file_asr} \
        --config_file ${config_file_general} \
        --batch-size ${da_batch_size} \
        --epochs ${da_epochs} \
        --FGSM_augmentation ${da_FGSM_augmentation} \
        --epsilon_FGSM ${da_epsilon_FGSM} \
        --alpha_FGSM ${da_alpha_FGSM} \
        --use_vat ${da_use_vat} \
        --vat_epsilon ${da_vat_epsilon} \
        --vat_delta_weight ${da_vat_delta_weight} \
        --vat_weight ${da_vat_weight} \
        --vat_iter ${da_vat_iter} \
        --p_aug ${da_p_aug} \
        --start_augmentation ${da_start_augmentation} \
        --use_spec_aug ${da_use_spec_aug} \
        --SpecF ${da_SpecF} \
        --SpecT ${da_SpecT} \
        --use_shift ${da_use_shift} \
        --Shift_frames ${da_Shift_frames} \
        --use_random_mix_noise ${da_use_random_mix_noise} \
        --use_delta ${da_use_delta} \
        --dict_dir ${dictroot} \
        --train_folder ${train_name} \
        --dev_folder "noisy_dev" \
        --print_freq ${da_print_freq} \
        --validate_freq ${da_validate_freq} \
        --rnnlm ${lmexpdir}/rnnlm.model.best \
        --exp_path ${exp_root}/${da_name} \
        --checkpoints_dir ${checkpoints_dir} \
        --works_dir ${exp_root}/${da_name}
    echo "train model with data augmentation and adversarial samples finish"
fi
if [ ${stage} -le 6 ]; then
    echo "stage 6 decoding for model with data augmentation and adversarial samples"
    echo "Decoding clean test set"
    bash ./recog_conformer.sh "recog_test_${da_name}" "${da_name}" "test"
    echo "Decoding match noisy test set"
    bash ./recog_conformer.sh "recog_noisy_test_${da_name}" "${da_name}" "noisy_test"
    echo "Deocing unmatch noisy test set"
    bash ./recog_conformer.sh "recog_noisy_test_musan_${da_name}" "${da_name}" "noisy_test_musan"
    echo "model recognition finish"
fi
if [ ${stage} -le 7 ]; then
    echo "stage 7 train teacher student model"
    if [ ${ts_use_random_mix_noise} == "True" ]; then
        train_name="train_random_mix"
    else
        train_name="noisy_train"
    fi
    config_file_ts=./config/teacher_student_config.yml
    teacher_resume=${checkpoints_dir}/${teacher_name}/model.acc.best
    mkdir -p ${checkpoints_dir}/${da_name}
    python3 Teacher_Student_train.py \
        --dataroot ${dataroot} \
        --name ${ts_name} \
        --config_file ${config_file_ts} \
        --config_file ${config_file_general} \
        --teacher_resume ${teacher_resume} \
        --student_model_config ${config_file_asr} \
        --batch-size ${ts_batch_size} \
        --epochs ${ts_epochs} \
        --FGSM_augmentation ${ts_FGSM_augmentation} \
        --epsilon_FGSM ${ts_epsilon_FGSM} \
        --alpha_FGSM ${ts_alpha_FGSM} \
        --use_vat ${ts_use_vat} \
        --vat_epsilon ${ts_vat_epsilon} \
        --vat_delta_weight ${ts_vat_delta_weight} \
        --vat_weight ${ts_vat_weight} \
        --vat_iter ${ts_vat_iter} \
        --p_aug ${ts_p_aug} \
        --start_augmentation ${ts_start_augmentation} \
        --use_spec_aug ${ts_use_spec_aug} \
        --SpecF ${ts_SpecF} \
        --SpecT ${ts_SpecT} \
        --use_shift ${ts_use_shift} \
        --Shift_frames ${ts_Shift_frames} \
        --use_random_mix_noise ${ts_use_random_mix_noise} \
        --use_delta ${ts_use_delta} \
        --dict_dir ${dictroot} \
        --train_folder ${train_name} \
        --dev_folder "noisy_dev" \
        --print_freq ${ts_print_freq} \
        --validate_freq ${ts_validate_freq} \
        --rnnlm ${lmexpdir}/rnnlm.model.best \
        --exp_path ${exp_root}/${ts_name} \
        --checkpoints_dir ${checkpoints_dir} \
        --works_dir ${exp_root}/${ts_name}
    echo "train teacher student model finish"
fi
if [ ${stage} -le 8 ]; then
    cp ${checkpoints_dir}/${ts_name}/fbank_noise_cmvn.npy ${checkpoints_dir}/${ts_name}/fbank_cmvn.npy
    echo "stage 8 decoding for teacher student model"
    echo "Decoding clean test set"
    bash ./recog_conformer.sh "recog_test_${ts_name}" "${ts_name}" "test"
    echo "Decoding match noisy test set"
    bash ./recog_conformer.sh "recog_noisy_test_${ts_name}" "${ts_name}" "noisy_test"
    echo "Deocing unmatch noisy test set"
    bash ./recog_conformer.sh "recog_noisy_test_musan_${ts_name}" "${ts_name}" "noisy_test_musan"
    echo "model recognition finish"
fi
if [ ${stage} -le 9 ]; then
    python3 utils/display_cer.py \
                --recog_dir ${checkpoints_dir}/recog \
                --test_set_name test \
                --test_set_name noisy_test \
                --test_set_name noisy_test_musan \
        | tee ${checkpoints_dir}/recog/evaluation.log
fi



