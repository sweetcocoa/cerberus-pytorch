import torch
import torch.nn as nn
import numpy as np

from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.util import midi_to_hz


def get_tf(roll, midi_min, scaling):
    # roll (freq, time)
    time = np.arange(roll.shape[1])
    freqs = [roll[:, t].nonzero(as_tuple=True)[0] for t in time]
    t_ref, f_ref = time, freqs
    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [np.array([midi_to_hz(midi_min + midi) for midi in freqs]) for freqs in f_ref]
    return t_ref, f_ref


class TrMetrics:
    def __init__(self, config):
        self.config = config
        self.scaling = self.config.hop_length / self.config.sr

    def __call__(self, pred, label, threshold=None):
        """
        pred : (freq, time) float
        label : (freq, time) float

        return : dict

        e.g. OrderedDict([('Precision', 0.07260726072607261),
             ('Recall', 0.056921086675291076),
             ('Accuracy', 0.03295880149812734),
             ('Substitution Error', 0.3738680465717982),
             ('Miss Error', 0.5692108667529108),
             ('False Alarm Error', 0.35316946959896506),
             ('Total Error', 1.296248382923674),
             ('Chroma Precision', 0.23927392739273928),
             ('Chroma Recall', 0.18758085381630013),
             ('Chroma Accuracy', 0.11750405186385737),
             ('Chroma Substitution Error', 0.24320827943078913),
             ('Chroma Miss Error', 0.5692108667529108),
             ('Chroma False Alarm Error', 0.35316946959896506),
             ('Chroma Total Error', 1.165588615782665)])

        """
        if threshold is None:
            threshold = self.config.transcription_threshold
            
        t_ref, f_ref = get_tf(label.int(), self.config.midi_min, scaling=self.scaling)
        t_est, f_est = get_tf((pred > threshold).int(), self.config.midi_min, scaling=self.scaling)
        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        
        return frame_metrics