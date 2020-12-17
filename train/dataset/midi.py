import torch
import pretty_midi
import numpy as np
from pretty_midi.containers import PitchBend
from pretty_midi.utilities import pitch_bend_to_semitones, note_number_to_hz
import mido

program_dict = dict(
    Piano=2,
    Bass=35,
    Drums=119,
    Guitar=28
)

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    
    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def parse_midi(path):
    """
    Original Source : https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/midi.py
    """

    """open midi file and return np.array of (onset, offset, note, velocity) rows"""
    midi = mido.MidiFile(path)

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
            events.append(event)

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        if len(events) == i + 1:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n is events[-1])

        note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    return np.array(notes)


def parsed_midi_to_roll(mid_np, audio_length, hop_length, sample_rate, num_pitches, midi_min, hops_in_onset=1, hops_in_offset=1):
    n_keys = num_pitches
    n_steps = (audio_length - 1) // hop_length + 1

    label = torch.zeros(n_steps, n_keys, dtype=torch.uint8)
    velocity = torch.zeros(n_steps, n_keys, dtype=torch.uint8)

    for onset, offset, note, vel in mid_np:
        left = int(round(onset * sample_rate / hop_length))
        onset_right = min(n_steps, left + hops_in_onset)
        frame_right = int(round(offset * sample_rate / hop_length))
        frame_right = min(n_steps, frame_right)
        offset_right = min(n_steps, frame_right + hops_in_offset)

        f = int(note) - midi_min

        if f >= n_keys:
            continue
        
        label[left:onset_right, f] = 3
        label[onset_right:frame_right, f] = 2
        label[frame_right:offset_right, f] = 1
        velocity[left:frame_right, f] = vel

    data = dict(label=label, velocity=velocity)
    return data