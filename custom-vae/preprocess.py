import os
from os.path import join

import librosa
import numpy as np
import pyworld as pw
import tensorflow as tf


# Script for feature extraction

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dir_to_wav', './dataset/vcc2016/wav', 'Dir to *.wav')
tf.app.flags.DEFINE_string('dir_to_bin', './dataset/vcc2016/bin', 'Dir to output *.bin')
tf.app.flags.DEFINE_integer('fs', 16000, 'Global sampling frequency')
tf.app.flags.DEFINE_float('f0_ceil', 500, 'Global f0 ceiling')
tf.app.flags.DEFINE_string('train_file_pattern', 
    './dataset/vcc2016/bin/Testing Set/*/*.bin', 'testing dir (to *.bin)')

EPSILON = 1e-10
SETS = ['Training Set', 'Testing Set']  # NOTE: for VCC2016 only
SPEAKERS = ['TF2', 'SM1', 'SF2', 'SF3', 'TM1', 'TF1', 'SM2', 'SF1', 'TM2', 'TM3']
FFT_SIZE = 1024
SP_DIM = FFT_SIZE // 2 + 1
FEAT_DIM = SP_DIM + SP_DIM + 1 + 1 + 1  # [sp, ap, f0, en, s]
RECORD_BYTES = FEAT_DIM * 4  # all features saved in `float32`


def wav2pw(x, fs=16000, fft_size=FFT_SIZE):
    ''' Extract WORLD feature from waveform '''
    _f0, t = pw.dio(x, fs, f0_ceil=args.f0_ceil) # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs) # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, fs, fft_size=fft_size) # extract aperiodicity
    return {
        'f0': f0,
        'sp': sp,
        'ap': ap,
    }


def extract(filename, fft_size=FFT_SIZE, dtype=np.float32):
    ''' Basic (WORLD) feature extraction ''' 
    x, _ = librosa.load(filename, sr=args.fs, mono=True, dtype=np.float64)
    features = wav2pw(x, args.fs, fft_size=fft_size)
    ap = features['ap']
    f0 = features['f0'].reshape([-1, 1])
    sp = features['sp']
    en = np.sum(sp + EPSILON, axis=1, keepdims=True)
    sp = np.log10(sp / en)
    return np.concatenate([sp, ap, f0, en], axis=1).astype(dtype)


def read_whole_features(file_pattern, num_epochs=1):
    '''
    Return
        `feature`: `dict` whose keys are `sp`, `ap`, `f0`, `en`, `speaker`
    '''
    files = tf.gfile.Glob(file_pattern)
    print('{} files found'.format(len(files)))
    filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    print("Processing {}".format(key), flush=True)
    value = tf.decode_raw(value, tf.float32)
    value = tf.reshape(value, [-1, FEAT_DIM])
    return {
        'sp': value[:, :SP_DIM],
        'ap': value[:, SP_DIM : 2*SP_DIM],
        'f0': value[:, SP_DIM * 2],
        'en': value[:, SP_DIM * 2 + 1],
        'speaker': tf.cast(value[:, SP_DIM * 2 + 2], tf.int64),
        'filename': key,
    }


def extract_and_save_bin_to(dir_to_bin, dir_to_source):
    ''' Runs feature extraction on audio files and saves to corresponding binary file directories '''
    sets = [s for s in os.listdir(dir_to_source) if s in SETS]
    for d in sets:
        path = join(dir_to_source, d)
        speakers = [s for s in os.listdir(path)]
        for s in speakers:
            path = join(dir_to_source, d, s)
            output_dir = join(dir_to_bin, d, s)
            if not tf.gfile.Exists(output_dir):
                tf.gfile.MakeDirs(output_dir)
            for f in os.listdir(path):
                filename = join(path, f)
                print(filename)
                if not os.path.isdir(filename):
                    features = extract(filename)
                    labels = SPEAKERS.index(s) * np.ones(
                        [features.shape[0], 1],
                        np.float32,
                    )
                    b = os.path.splitext(f)[0]
                    features = np.concatenate([features, labels], 1)
                    with open(join(output_dir, '{}.bin'.format(b)), 'wb') as fp:
                        fp.write(features.tostring())
                        print(join(output_dir, '{}.bin'.format(b)))


def save_training_stats():
    ''' 
    Saves features for each speaker as numpy array.
    '''
    tf.gfile.MkDir('./etc')

    # ==== Save max and min value ====
    x = read_whole_features(args.train_file_pattern)
    x_all = list()
    y_all = list()
    f0_all = list()
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        while True:
            try:
                features = sess.run(x)
                print('Processing {}'.format(features['filename']))
                x_all.append(features['sp'])
                y_all.append(features['speaker'])
                f0_all.append(features['f0'])
            finally:
                pass

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    f0_all = np.concatenate(f0_all, axis=0)


    # ==== Save f0 stats for each speaker ====
    for s in SPEAKERS:
        print('Speaker {}'.format(s), flush=True)
        f0 = f0_all[SPEAKERS.index(s) == y_all]
        print('  len: {}'.format(len(f0)))
        f0 = f0[f0 > 2.]
        f0 = np.log(f0)
        mu, std = f0.mean(), f0.std()

        # Save as `float32`
        with open('./etc/{}.npf'.format(s), 'wb') as fp:
            fp.write(np.asarray([mu, std]).tostring())


    # Save values at the extreme quartiles as we will need these for normalization later
    q005 = np.percentile(x_all, 0.5, axis=0)
    q995 = np.percentile(x_all, 99.5, axis=0)

    # Save as `float32`
    with open('./etc/xmin.npf', 'wb') as fp:
        fp.write(q005.tostring())

    with open('./etc/xmax.npf', 'wb') as fp:
        fp.write(q995.tostring())


if __name__ == '__main__':
    extract_and_save_bin_to(
        args.dir_to_bin,
        args.dir_to_wav,
    )
    save_training_stats()