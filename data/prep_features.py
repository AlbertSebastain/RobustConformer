import os
import sys

from numpy.core.fromnumeric import argmax
import kaldi_io
import librosa
import numpy as np
import scipy.signal
import scipy.io

import torchaudio
import math
import random
from random import choice
from multiprocessing import Pool
import scipy.io.wavfile as wav
from os import walk
import argparse
from tqdm import tqdm


class Feats_Parses():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.AlreadyParsed = False

    def initialized(self):
        self.parser.add_argument('--data_dir', help='Path to the data directory', type=str, default='.')
        self.parser.add_argument('--RequireClean', help='Whether to generate clean features', action='store_true')
        self.parser.add_argument('--RequireMix', help='Whether to generate mixed features', action='store_true')
        self.parser.add_argument('--ParellelNoisyNum', help="Number of mixed noisy sets", type=int, default=1)
        self.parser.add_argument('--NoiseRepeatNum', help='Number of mixing the noise for one utterance', type=int, default=1)
        self.parser.add_argument('--DataType', help='Type of data,e.g., train, test, dev', type=str, default='data')
        self.parser.add_argument('--SavedType', help='The type of genreated data', type=str, default=None)
        self.parser.add_argument('--ThreadNum', help='Number of threads', type=int, default=4)
        self.parser.add_argument('--LowSNR', help="low SNR", type=int, default=5)
        self.parser.add_argument('--HighSNR', help="high SNR", type=int, default=5)
        self.parser.add_argument('--step', help='step of sampled SNR', type=int, default=5)
        self.parser.add_argument('--noise_dir',help='Path to noise directory', type=str, default='.')
        self.AlreadyParsed = True

    def parse(self):
        if not self.AlreadyParsed:
            self.initialized()
            self.opt = self.parser.parse_args()
            if self.opt.SavedType == None:
                self.opt.SavedType = self.opt.DataType
        return self.opt

def load_audio(path):
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if sound.shape[0] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=0)  # multiple channels, average
    return sound
def load_audio_noise(path):
    name = path.split(os.sep)[-1].split(".")[0]
    sound = scipy.io.loadmat(path)[name]
    sound = sound.squeeze()
    sound = sound / (2**15)
    sound = sound.astype('float32')
    return sound

class NoiseLoader():
    def __init__(self, WavNoise):
        if WavNoise:
            self.loader = load_audio
        else:
            self.loader = load_audio_noise
    
    def load(self, noise):
        return self.loader(noise)

def MakeMixture(speech, noise, db):
    if speech is None or noise is None:
        return None
    if np.sum(np.square(noise)) < 1.0e-6:
        return None

    spelen = speech.shape[0]

    exnoise = noise
    while exnoise.shape[0] < spelen:
        exnoise = np.concatenate([exnoise, noise], 0)
    noise = exnoise
    noilen = noise.shape[0]

    elen = noilen - spelen - 1
    if elen > 1:
        s = round(random.randint(0, elen - 1))
    else:
        s = 0
    e = s + spelen

    noise = noise[s:e]

    try:
        noi_pow = np.sum(np.square(noise))
        if noi_pow > 0:
            noi_scale = math.sqrt(np.sum(np.square(speech)) / (noi_pow * (10 ** (db / 10.0))))
        else:
            print(noi_pow, np.square(noise), "error")
            return None
    except:
        return None

    nnoise = noise * noi_scale
    mixture = speech + nnoise
    mixture = mixture.astype("float32")
    return mixture


def make_feature(wav_path_list, noise_wav_list, feat_dir, thread_num, argument=False, repeat_num=1, lowSNR=0, highSNR=20, step=5, WavNoise=True):
    mag_ark_scp_output = "ark:| copy-feats --compress=true ark:- ark,scp:{0}/feats{1}.ark,{0}/feats{1}.scp".format(feat_dir, thread_num)
    ang_ark_scp_output = "ark:| copy-feats --compress=true ark:- ark,scp:{0}/angles{1}.ark,{0}/angles{1}.scp".format(feat_dir, thread_num)
    if argument:
        LoadNoise = NoiseLoader(WavNoise)
        fwrite = open(os.path.join(feat_dir, "db" + str(thread_num)), "a")
        fNoiseWrite = open(os.path.join(feat_dir, "noise" + str(thread_num)), "a")
    f_mag = kaldi_io.open_or_fd(mag_ark_scp_output, "wb")
    f_ang = kaldi_io.open_or_fd(ang_ark_scp_output, "wb")
    for num in range(repeat_num):
        for tmp in wav_path_list:
            uttid, wav_path = tmp
            clean = load_audio(wav_path)
            y = None
            while y is None:
                if argument:
                    noise_path = choice(noise_wav_list)
                    noise_name = noise_path.split('/')[-1].split('.')[0]
                    n = LoadNoise.load(noise_path)
                    db = random.randrange(lowSNR,highSNR+step,step)
                    y = MakeMixture(clean, n, db)
                    uttid_new = uttid + "__mix{}".format(num)
                    #print(uttid_new + " " + str(db) + "\n")
                    fwrite.write(uttid_new + " " + str(db) + "\n")
                    fNoiseWrite.write(uttid_new + " "+ noise_name + "\n")
                else:
                    y = clean
                    uttid_new = uttid
            # STFT
            if y is not None:
                D = librosa.stft(y, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
                spect = np.abs(D)
                angle = np.angle(D)
                ##feat = np.concatenate((spect, angle), axis=1)
                ##feat = feat.transpose((1, 0))
                kaldi_io.write_mat(f_mag, spect.transpose((1, 0)), key=uttid_new)
                kaldi_io.write_mat(f_ang, angle.transpose((1, 0)), key=uttid_new)
            else:
                print(noise_path, tmp, "error")

    f_mag.close()
    f_ang.close()
    if argument:
        fwrite.close()
        fNoiseWrite.close()
def main():

    # 输入参数
    opt_parse = Feats_Parses()
    opt = opt_parse.parse()
    
    data_dir = opt.data_dir
    feat_dir = data_dir
    noise_repeat_num = opt.NoiseRepeatNum
    data_type = opt.DataType
    GenCleanFeature = opt.RequireClean
    GenMixFeature = opt.RequireMix
    parellel_num = opt.ParellelNoisyNum
    save_type = opt.SavedType
    threads_num = opt.ThreadNum
    noise_dir = opt.noise_dir
    lowSNR = opt.LowSNR
    highSNR = opt.HighSNR
    step = opt.step

    feat_dir = os.path.join(feat_dir, save_type)
    clean_wav_list = []
    data_dir = os.path.join(data_dir, data_type)
    # Save the clean wav paths
    clean_wav_scp = os.path.join(data_dir, "wav.scp") 
    with open(clean_wav_scp, "r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip().replace("\n", "")
            uttid, wav_path = line.split()
            clean_wav_list.append((uttid, wav_path))
    print(">> clean_wav_list len:", len(clean_wav_list))

    # Generate clean features
    if GenCleanFeature:
        # Build clean feature folder
        clean_feat_dir = os.path.join(feat_dir, "clean")
        if not os.path.exists(clean_feat_dir):
            os.makedirs(clean_feat_dir)
        wav_num = len(clean_wav_list)
        print(">> Parent process %s." % os.getpid())
        p = Pool()
        start = 0
        noise_wav_list = []
        for i in range(threads_num):
            end = start + int(wav_num / threads_num)
            if i == (threads_num - 1):
                end = wav_num
            wav_path_tmp_list = clean_wav_list[start:end]
            start = end
            p.apply_async(make_feature, args=(wav_path_tmp_list, noise_wav_list, clean_feat_dir, i, False))
        print(">> Waiting for all subprocesses done...")
        p.close()
        p.join()
        print(">> All subprocesses done.")
        command_line = "cat {}/feats*.scp > {}/clean_feats.scp".format(clean_feat_dir, feat_dir)
        os.system(command_line)
        command_line = "cat {}/angles*.scp > {}/clean_angles.scp".format(clean_feat_dir, feat_dir)
        os.system(command_line)

    # Generate mix features
    if GenMixFeature:
        noise_wav_list = []
        WavNoise = True
        for (_, _, filenames) in walk(noise_dir):
            noise_wav_list = [os.path.join(noise_dir,x) for x in filenames if x.split(".")[-1] == "mat" or x.split(".")[-1] == "wav"]
            break
        if noise_wav_list[0].split('.')[-1] == "mat":
            WavNoise = False
        print(">> noise_wav_list len",len(noise_wav_list))
        wav_num = len(clean_wav_list)
        for idx in range(parellel_num):
            suffix = ''
            if parellel_num != 1:
                suffix = "_"+str(idx)
            mix_feat_dir = os.path.join(feat_dir, "mix"+suffix)
            if not os.path.exists(mix_feat_dir):
                os.makedirs(mix_feat_dir)
            print(">>Mixing noisy features: "+str(idx)+" out of "+str(parellel_num))
            print(">> Parent process %s." % os.getpid())
            p = Pool()
            for i in range(threads_num):
                wav_path_tmp_list = clean_wav_list[int(i * wav_num / threads_num) : int((i + 1) * wav_num / threads_num)]
                p.apply_async(make_feature, args=(wav_path_tmp_list, noise_wav_list, mix_feat_dir, i, True, noise_repeat_num, lowSNR, highSNR, step, WavNoise))
                #make_feature(wav_path_tmp_list, noise_wav_list, mix_feat_dir, i, True, noise_repeat_num,lowSNR, highSNR, step, WavNoise)
            print(">> Waiting for all subprocesses done...")
            p.close()
            p.join()
            print(">> All subprocesses done.")
            command_line = "cat {}/feats*.scp > {}/mix_feats{}.scp".format(mix_feat_dir, feat_dir, suffix)
            os.system(command_line)
            command_line = "cat {}/angles*.scp > {}/mix_angles{}.scp".format(mix_feat_dir, feat_dir, suffix)
            os.system(command_line)
            command_line = "cat {}/db* > {}/db{}.scp".format(mix_feat_dir, feat_dir, suffix)
            os.system(command_line)
            command_line = "cat {}/noise* > {}/noise{}.scp".format(mix_feat_dir, feat_dir, suffix)
            os.system(command_line)
    

if __name__ == "__main__":
    main()
