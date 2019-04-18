import numpy as np
import pyworld as pw
import soundfile as sf
import tensorflow as tf
from analyzer import SPEAKERS, pw2wav, read, read_whole_features


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'train_file_pattern',
    './dataset/vcc2016/bin/Testing Set/*/*.bin',
    'testing dir (to *.bin)')


def main():
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


    # ==== Save 0.5th and 99.5th percentile values ====
    q005 = np.percentile(x_all, 0.5, axis=0)
    q995 = np.percentile(x_all, 99.5, axis=0)

    # Save as `float32`
    with open('./etc/xmin.npf', 'wb') as fp:
        fp.write(q005.tostring())

    with open('./etc/xmax.npf', 'wb') as fp:
        fp.write(q995.tostring())


if __name__ == '__main__':
    main()
