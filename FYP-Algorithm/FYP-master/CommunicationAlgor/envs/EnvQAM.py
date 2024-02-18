# ============================================ ReceiverEnvWithArbitaryOrderIIRwithDelayedTargetSNR ============================================
# - The original definitions can be found in the notebook 'notebook-stage_2.3.4.ipynb'. 
# - This is the latest environment as by 31/01/2024. 
# - Includes time delaying of the target signal using linear interpolation. 

import os, sys, json

import numpy as np
from scipy.signal import lfilter, freqz

import gymnasium as gym
from gymnasium import spaces

from typing import Callable

sys.path.append('../../')
from stage2_Comm_helper import create_target_and_jammed_signals, NormalizeState, action2IIR, power

# constants
SAMPLING_FREQ = 44_100 # Hz

class RecieverEnvWithQAM(gym.Env):
    """
    A custom environment developed in the accordance with gym environment API that immitates a receiver of jammed audio with IIR filtering.
    :param order: the order of the filter
    :param S: signal partition size which represents a state
    :param cut_off_freq: the frequency to truncate the audio spectrum to generate the target signal; equivalent to the ideal cut-off frequency of the learned filter
    :param interference_center_freq: the frequency/frequencies to shift the target spectrum to generate the non-overlapping interference
    :param interference_scalar: scaling factors to scale the shifted interference spectrums; default is 1
    :param zero_magnitude_mapping: a callable function to convert the values of actions for the magnitude of zeros from range [0, 1] to [0, inf); \
        the function must take a float in the range [0, 1] as the input and return a float in the range [0, inf); \
        default mapping is linear (no mapping)
    :param gradient: the gradient argument for the ``zero_magnitude_mapping`` function; represents the gradient of the function around the middle point of the input domain (x=0.5), \
        which corresponds to the unit circle; default is None, indicating no gradient argument is required in ``zero_magnitude_mapping``
    :param fix_zeros_magnitude: a boolean value specifying to fix the magnitudes of zeros to 1 (useful to reduce the dimensionality of the action space); \
        default is False
    :param automatic_gain: a boolean value to automatically estimate the constant gain factor 'k' of the filter so that the filter gain becomes unity at zero frequency \
        (useful to reduce the dimensionality of the action space); default is False
    :param SNR_as_dB: a boolean value specifying whether to take the reward SNR value in dB or as a ratio; defaults to True (i.e., dB rewards will be taken)
    :param show_effect: a boolean value specifying whether to show the reward and action of each step; default is True
    :param audio_json: path of a json file containing the names of the audio wav files the environment can access\
        put the audio file names without the .wav extension in a json array inside the file
    """

    # define constants
    MIN_BUFFER_SIZE = 10 # RAISE THIS LATER!!!
    EPISODE_LENGTH  = np.inf # np.inf
    MAX_TOTAL_NUM_OF_STEPS = np.inf
    # OBSERVATION_SPACE_BOUND = 5

    def __init__(self,
                 order: int,
                 S: int,
                 cut_off_freq: int,
                 interference_center_freq: int | float | list[int | float],
                 interference_scalar: int | float | list[int | float],
                 zero_magnitude_mapping: Callable[[float], float] = None,
                 gradient: float | int = None,
                 fix_zeros_magnitude: bool = False,
                 automatic_gain: bool = False,
                 SNR_as_dB: bool = True,
                 show_effect: bool = True,
                 audio_json: str = '../stage_1/audio_files/audio_files.json'
        ):

        super(ReceiverEnvWithArbitaryOrderIIRwithDelayedTargetSNR, self).__init__()

        # ----- verifying input arguments and setting them as class atributes ----
        # filter order
        if type(order) != int or order < 2:
            raise Exception(f"the filter order must be a positive integer greater than 2: given {order}")
        self.order = order

        # signal partition size
        if S < self.MIN_BUFFER_SIZE:
            raise Exception(f"the buffer size 'S' must be larger than MIN_BUFFER_SIZE, {self.MIN_BUFFER_SIZE}: given {S}")
        self.S = S

        # other parameters
        self.cut_off_freq = cut_off_freq
        self.interference_center_freq = interference_center_freq
        self.interference_scalar = interference_scalar
        self.audio_json = audio_json
        self.show_effect = show_effect
        self.zero_magnitude_mapping = zero_magnitude_mapping
        self.gradient = gradient
        self.fix_zeros_magnitude = fix_zeros_magnitude
        self.automatic_gain = automatic_gain
        self.as_dB = SNR_as_dB

        # ----------------------------- Action Space -----------------------------
        # action - choosing fixed gain k, zeros, and poles of an N-th order IIR filter
        # note that the action is NOT TUNING/,ADJUSTING, or CHANGING the coefficeints of an existing filter.
        # the dimensionality of the action space depends on whether the filter order is even or not and the two arguments `fix_zeros_magnitude` and `automatic_gain`
        if order % 2 == 0:
            action_shape = 2*order + 1
            if (fix_zeros_magnitude):
                action_shape -= order//2 # N/2 number of actions get fixed
            if (automatic_gain):
                action_shape -= 1 # the constant gain 'k' becomes fixed
        else:
            raise Exception(f"the environment still does not support odd order IIR filters: given {order}")
        print(f"creating action space with {action_shape} dimensions...")
        self.action_space = spaces.Box(low=0, high=1, shape=[action_shape], dtype=np.float32) # float16 -> float32, lower limit must be 0.

        # ----------------------------- State Space ------------------------------
        state_shape = (self.S, )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float64)

        # ------------------------------ Counters --------------------------------
        self.global_counter = 0  # a counter to keep track of the number of elapsed time steps of the environment
        self.counter = 0         # a counter to keep track of the number of elapsed time steps in the current episode
        self.episode_counter = 0 # a counter to keep track of the number of total episodes

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=None) # options must be forced to None

        # reset the counters
        if isinstance(options, dict) and 'reset_all' in options and options['reset_all'] == True:
            self.global_counter  = 0
            self.episode_counter = 0
        self.counter = 0
        self.episode_counter += 1

        print('\n' + "-" * 50 + f"episode no: {self.episode_counter}" + "-" * 50)

        # for each episode, choose the audio signal specified by `audio_num` in the options
        with open(self.audio_json) as audio_json_file:
            train_audio_names = json.load(audio_json_file)["train"]
        audio_num = 1 # default audio track - 'arms_around_you-MONO.wav'
        if isinstance(options, dict) and 'audio_num' in options:
            audio_num = options['audio_num']

        # i = np.random.randint(low=1, high=self.audio_num) # len(train_audio_names)
        # create the target and jammed signals
        target_signal, jammed_signal = create_target_and_jammed_signals(
            audio_name = train_audio_names[audio_num],
            truncation_freq = self.cut_off_freq,
            interference_center_freq = self.interference_center_freq,
            interference_scalar = self.interference_scalar,
            signal_partition_size = self.S
        )
        self.target_signal = target_signal
        self.jammed_signal = jammed_signal

        # return the initial state
        state = NormalizeState(jammed_signal[:self.S])

        # declare the initial conditions in the start of the audio
        self.initial_conds = np.zeros(self.order)

        info = {}

        # return the initial state and info
        return state, info

    def step(self, action):

        info = {}

        # increment the counters
        self.global_counter += 1
        self.counter += 1

        # ----- create the filter -----
        b, a, (_, _, k) = action2IIR(
            action = action,
            order = self.order,
            zero_magnitude_mapping = self.zero_magnitude_mapping,
            gradient = self.gradient
        )

        # filter the current jammed signal partition
        jammed = self.jammed_signal[(self.counter - 1) * self.S : self.counter * self.S]
        filtered, self.initial_conds = lfilter(b, a, x=jammed, zi=self.initial_conds)

        # find the time delay applied by the filter
        freq_resp = freqz(b, a, worN=[0, self.cut_off_freq/2, self.cut_off_freq], fs=SAMPLING_FREQ)[1]
        phase_resp = np.angle(freq_resp)
        group_delay = -np.mean([(phase_resp[2] - phase_resp[0]) / self.cut_off_freq, 2 * (phase_resp[1] - phase_resp[0]) / self.cut_off_freq]) / (2 * np.pi)
        time_delay  = group_delay * SAMPLING_FREQ

        # find the target signal partition
        target = self.target_signal[(self.counter - 1) * self.S : self.counter * self.S]

        # delay the target partition with linear interpolation
        delayed_target = np.zeros_like(target).astype(np.float64)
        int_delay  = np.abs(int(time_delay))
        frac_delay = np.abs(time_delay - int(time_delay))

        if time_delay >= 0: # a positive delay
            if (self.counter - 1) * self.S - int_delay - 1 < 0:
                target_init_conds = np.zeros([int_delay + 1], dtype='float64')
            else:
                target_init_conds = self.target_signal[(self.counter - 1) * self.S - int_delay - 1 : (self.counter - 1) * self.S]

            target = np.concatenate((target, target_init_conds))
            for i in range(len(delayed_target)):
                delayed_target[i] = target[i - int_delay] - (target[i - int_delay] - target[i - int_delay - 1]) * frac_delay

        else: # a negative delay
            target_end_conds = self.target_signal[self.counter * self.S : self.counter * self.S + int_delay + 1]

            target = np.concatenate((target, target_end_conds))
            for i in range(len(delayed_target)):
                delayed_target[i] = target[i + int_delay] + (target[i + int_delay + 1] - target[i + int_delay]) * frac_delay

        # find the SNR reward value
        reward = SNR = power(delayed_target) / power(filtered - delayed_target)
        if self.as_dB: reward = 10 * np.log10(SNR) # convert SNR to dBs
        
        # generating the next state
        terminated = False
        if self.S * (self.counter + 2) >= len(self.jammed_signal):
            terminated = True
            info["cause"] = "signal over"

        # find the next state
        state = NormalizeState(self.jammed_signal[self.counter * self.S : (self.counter + 1) * self.S])

        # show log
        if self.show_effect: print(f"step: {self.counter}, SNR: {reward}, filter: {k}, {b}, {a}")

        # truncating the episode
        truncated = False
        if self.episode_counter == self.EPISODE_LENGTH or self.global_counter == self.MAX_TOTAL_NUM_OF_STEPS:
            truncated = True

        return state, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass