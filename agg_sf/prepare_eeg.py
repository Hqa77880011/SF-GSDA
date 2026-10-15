import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', default='data/eeg.npz')
    parser.add_argument('--montage', default='data/montage.npy')
    parser.add_argument('--sample-rate', type=float, default=128)
    parser.add_argument('--window-seconds', type=float, default=4)
    parser.add_argument('--stride-seconds', type=float, default=2)
    parser.add_argument('--mains-hz', type=float, default=50)
    parser.add_argument('--bands', action='store_true')
    args = parser.parse_args()
    path = Path(args.manifest)
    frame = pd.read_csv(path).fillna('')
    required = {'file', 'key', 'sample_rate', 'channels', 'subject', 'split', 'label'}
    if not required <= set(frame):
        raise ValueError('Trial manifest requires ' + ','.join(sorted(required)))
    arrays, labels, groups, splits = [], [], [], []
    channel_order = None
    for row in frame.itertuples():
        source = path.parent / row.file
        if source.suffix == '.mat':
            signal = loadmat(source)[row.key]
        elif source.suffix == '.npz':
            with np.load(source, allow_pickle=False) as data:
                signal = data[row.key]
        else:
            signal = np.load(source, allow_pickle=False)
        channels = row.channels.split(';')
        if signal.shape[0] != len(channels):
            raise ValueError('Signal must be channels by samples')
        if channel_order is not None and channels != channel_order:
            raise ValueError('Use one declared common channel order per output archive')
        channel_order = channels
        raw = mne.io.RawArray(signal.astype(float), mne.create_info(channels, row.sample_rate, ch_types='eeg'), verbose=False)
        raw.set_montage('standard_1020')
        raw.set_eeg_reference('average', projection=False, verbose=False)
        if args.mains_hz < row.sample_rate / 2:
            raw.notch_filter(args.mains_hz, verbose=False)
        raw.filter(1, 45, verbose=False)
        raw.resample(args.sample_rate, verbose=False)
        signal = raw.get_data()
        width = round(args.sample_rate * args.window_seconds)
        stride = round(args.sample_rate * args.stride_seconds)
        for start in range(0, signal.shape[1] - width + 1, stride):
            window = signal[:, start:start + width]
            if not np.isfinite(window).all():
                raise ValueError('A raw trial contains nonfinite samples')
            if args.bands:
                features = []
                for lower, upper in [(1,4),(4,8),(8,13),(13,30),(30,45)]:
                    sos = butter(4, [lower, upper], btype='bandpass', fs=args.sample_rate, output='sos')
                    filtered = sosfiltfilt(sos, window, axis=-1)
                    features.append(.5 * np.log(2 * np.pi * np.e * np.var(filtered, axis=-1).clip(1e-12)))
                window = np.stack(features, 1)[..., None]
            arrays.append(window.astype(np.float32))
            labels.append(int(row.label))
            groups.append(str(row.subject))
            splits.append(row.split)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.array(arrays), y=np.array(labels), group=np.array(groups), split=np.array(splits))
    montage = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    coordinates = np.array([montage[name] for name in channel_order])
    target = Path(args.montage)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, coordinates)


if __name__ == '__main__':
    main()
