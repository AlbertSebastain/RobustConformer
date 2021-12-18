#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
source ./config/path_config
save_type=$1
model=$2
recog_set=$3
# general configuration
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
verbose=1      # verbose option
# rnnlm related
model_unit=char
fusion=${fusion:=none}
config_file_decode=config/conformer_decode_config.yml
config_file_train=config/train_conformer_kernel_15.yml
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

train_set=train
train_dev=dev

echo "dictionary: ${dict}"
nlsyms=${dictroot}/non_lang_syms.txt
save_dir="recog/${save_type}"
nj=8
rtask=${recog_set}
name=${save_dir}
echo "the recog result is saved in"
echo ${checkpoints_dir}/${name}  
mkdir -p ${checkpoints_dir}/${name}
feat_recog_dir=${dataroot}/${rtask}
expdir=${exp_root}/${model}
feat_recog_dir=${dataroot}/${rtask}
echo "${feat_recog_dir}"
sdata=${feat_recog_dir}/split$nj

utils/split_data.sh ${feat_recog_dir} $nj || exit 1;
echo $nj > ${expdir}/num_jobs
${decode_cmd} JOB=1:${nj} ${checkpoints_dir}/${save_dir}/log/decode.JOB.log \
    python3 asr_recog_conf.py \
    --dataroot ${dataroot} \
    --dict_dir ${dictroot} \
    --name $name \
    --model-unit $model_unit \
    --nj $nj \
    --gpu_ids 0 \
    --resume ${expdir}/model.acc.best \
    --config_file ${config_file_decode} \
    --recog-dir ${sdata}/JOB/ \
    --checkpoints_dir ${checkpoints_dir} \
    --result-label ${checkpoints_dir}/${save_dir}/data.JOB.json \
    --verbose ${verbose} \
    --embed-init-file ${embed_init_file} \
    --exp_path ${expdir} \
    --test_folder ${rtask} 
utils/score_sclite.sh --nlsyms ${nlsyms} ${checkpoints_dir}/${save_dir} ${dict}
echo "Finished"
