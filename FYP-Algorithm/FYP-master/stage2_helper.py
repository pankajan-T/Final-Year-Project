import os, json
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
# %matplotlib inline

from scipy.io import wavfile
from scipy.fft import irfft
from scipy.signal import zpk2tf, freqz

from typing import Callable
from stage1_helper import Spectrum, to_min_size_int_array, trim_audio

SAMPLING_FREQ = 44_100 # Hz
EPS = 1e-4

# --------------------- new `create_target_and_jammed_signals` function for multicarrier interference --------------------
# the following function has been updated to generate all signals in 'np.float64' datatype instead of converting into an integer array of minimum size. 
def create_target_and_jammed_signals(
        audio_name: str,
        truncation_freq: int | float,
        interference_center_freq: int | float | list[int | float],
        signal_partition_size: int,
        interference_scalar: int | float | list[int | float] = 1,
        audio_file_dir: str = 'stage_1/audio_files/',
        save_files: bool = False
    ):
    """
    Creates the target and jammed signals with the specified truncation frequnecy and the interference center frequencies.

    :param audio_name: name of the audio .wav file (without the .wav extension)
    :param truncation_freq: the frequency to truncate the spectrum of the given audio to generate the target signal
    :param interference_center_freq: the frequency or the frequencies to shift the target spectrum to generate the non-overlapping interference. \
        (``truncation_freq`` and ``interference_center_freq`` must be chosen appropriately to create a non-overlapping interference with the target spectrum; \
        otherwise, errors will be raised.)
    :param signal_partition_size: the size of a signal partition considered to trim the audio signal
    :param interference_scalar: non-negative value/values to scale each interference component; must have the same number of elements as ``interference_center_freq``.
    :param audio_file_dir: the path to the directory containing the audio file
    :param save_files: boolean value indicating whether to write the resulting (MONO), truncated, and jammed signals

    Returns the target signal and the jammed signal.
    """

    SAMPLING_FREQ = 44_100 # each audio file must have a constant sampling freq to equally apply the truncation and interference center frequencies
    if type(interference_center_freq) == int or type(interference_center_freq) == float:
        interference_center_freq = [interference_center_freq]
    if type(interference_scalar) == int or type(interference_scalar) == float:
        interference_scalar = np.ones((len(interference_center_freq), )) * interference_scalar
    if len(interference_scalar) != len(interference_center_freq):
        raise Exception(f"the numbers of specified interference center frequencies and scalars are incompatible: given {len(interference_center_freq)} and {len(interference_scalar)}")

    print(f"audio name: '{audio_name}'")

    # ------------------------------------------ read the input audio ------------------------------------------
    audio_src_file = os.path.join(audio_file_dir, audio_name+'.wav')
    if not os.path.exists(audio_src_file):
        raise Exception(f"the specified audio file doesn't exist: given {audio_src_file}")
    sampling_rate, audio = wavfile.read(audio_src_file)
    sampling_space = 1/sampling_rate
    print(f"sampling rate: {sampling_rate} Hz")

    if (sampling_rate != SAMPLING_FREQ):
        raise Exception(f"Error - the sampling rate must be equal to {SAMPLING_FREQ}Hz: given {sampling_rate}")
    if (min(interference_center_freq) < 2 * truncation_freq):
        raise Exception(f"Error - non-overlapping interferene is impossible with the provided truncation and interference center frequencies: given {truncation_freq} and {interference_center_freq}")
    # if (interference_center_freq + truncation_freq > sampling_rate):
    #     raise Exception(f"Error - interference signal surpasses the sampling freqency of the original audio.")

    # ----------------------------------------- converting to MONO audio ---------------------------------------
    print(f"audio shape: {audio.shape}")
    print(f"data type: {audio.dtype}")

    if len(audio.shape) == 1: # MONO
        print("MONO audio file...")

    elif len(audio.shape) == 2 and audio.shape[-1] == 2: # STEREO
        print("converting audio from STEREO to MONO...")
        audio = np.average(audio, axis=-1).astype(audio.dtype)
        print(f"\taudio shape: {audio.shape}")
        print(f"\tdata type  : {audio.dtype}")

        # if save_files:
        # saving the MONO audio file
        mono_dst_file = os.path.join(audio_file_dir, audio_name+'-MONO.wav')
        print(f"\tsaving MONO audio: '{mono_dst_file}'...")
        wavfile.write(mono_dst_file, rate=sampling_rate, data=audio)
    else:
        raise TypeError("unsupported wav file format")

    # --------------------------------------- creating the target signal ---------------------------------------
    print(f"generating the target signal...")
    freq_bins, spectrum = Spectrum(audio, sampling_space=sampling_space, type='complex')

    # apply the cut-off frequency to audio spectrum
    print(f"\ttruncating the spectrum at {truncation_freq}Hz")
    rect_filter = (freq_bins < truncation_freq).astype(np.uint8)
    target_spectrum = spectrum * rect_filter
    target_signal = irfft(target_spectrum)

    # trim the signal (until a partition with sufficient non-zero audio samples)
    target_signal = trim_audio(target_signal, signal_partition_size)

    if save_files:
        # save the target signal
        target_dst_file = os.path.join(audio_file_dir, audio_name + "-target-MONO.wav")
        print(f"saving target signal: '{target_dst_file}'...")
        wavfile.write(target_dst_file, rate=sampling_rate, data=target_signal)

    # ---------------------------------------- create the jammed signal -----------------------------------------
    print(f"generating the jammed signal...")

    # creating a non-overlapping interference signal
    print(f"\tcreating a non-overlapping interference signal with target spectrum shifted to {', '.join(map(str, interference_center_freq))}Hz with scales {', '.join(map(str, interference_scalar))}")
    t = np.arange(len(target_signal)) * sampling_space

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    interference_center_freqs = np.array(interference_center_freq).reshape((-1, 1)) # column vector
    shifter = np.cos(2 * np.pi * interference_center_freqs * t)
    shifter *= np.array(interference_scalar).reshape((-1, 1))

    interference = 2 * target_signal * shifter
    interference = np.sum(interference, axis=0)

    jammed_signal = target_signal + interference
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    if save_files:
        # save the jammed signal
        jammed_dst_file = os.path.join(audio_file_dir, audio_name + "-jammed-MONO.wav")
        print(f"saving jammed signal: '{jammed_dst_file}'...")
        wavfile.write(jammed_dst_file, rate=sampling_rate, data=jammed_signal)

    return target_signal, jammed_signal

# --------------------------------------- functions for model training and testing ---------------------------------------
def train(model_, env_, audio_num, max_num_steps, noise_schedule=False, noise_step_size=1_000, noise_exponent=0.5, reward_history=None, action_history=None):
    """
    Trains a model in a given environment over a specified number of time steps.
    :param model_: the DRL model 
    :param env_: the environment the model is going to trained in
    :param audio_num: the number/index of the audio track to be used as the training data
    :param max_num_steps: the maximum number of time steps for training
    ... 
    
    Returns None
    """

    if reward_history is None: reward_history = []
    if action_history is None: action_history = [][:]
    step_count = 0

    # reset the environment
    state, _ = env_.reset(options={'reset_all': True, 'audio_num':audio_num}) # FOR SINGLE EPISODE
    done = False

    while not done:
        # feed the state to the agent (model) and get an action
        action = model_.choose_action(state, noise_schedule, noise_step_size, noise_exponent).numpy() # this includes the exploration noise

        # take the action in the environment
        next_state, reward, terminated, truncated, _ = env_.step(action)
        done = terminated | truncated
        step_count += 1

        # store the transition in the replay buffer of the DDPG agent
        model_.remember(state, action, reward, next_state, done)

        # train the model
        model_.learn()

        # set the `next_state` as `state`
        state = next_state

        # keep track of `reward` and `action`
        reward_history.append(reward)
        action_history.append(action)
        
        if step_count >= max_num_steps:
            done = True

    return reward_history, action_history

def test(model_, env_, audio_num, num_steps, fixed_action=None):
    """
    Tests a model in a given environment with the specified number of time steps and with provided testing data.
    :param model_: the trained DRL model to be tested 
    :param env_: the environment the model is going to tested in
    :param audio_num: the number/index of the audio track to be used as the testing data
    :param num_steps: the number of time steps for testing
    :param fixed_action: if not None, the provided action would be taken to take steps in the environment instead of the agent's predictions

    Returns the rewards and the actions taken
    """

    reward_history = []
    action_history = []
    step_count = 0

    # reset the environment
    state, _ = env_.reset(options={'reset_all': True, 'audio_num':audio_num}) # FOR SINGLE EPISODE
    done = False

    while not done:
        # feed the state to the agent (model) and get an action
        if fixed_action is None:
            action = model_.choose_action(state, evaluate=True).numpy() # this DOES NOT include the exploration noise
        else:
            action = fixed_action

        # take the action in the environment
        next_state, reward, terminated, truncated, _ = env_.step(action)
        done = terminated | truncated
        step_count += 1

        # set the `next_state` as `state`
        state = next_state

        # keep track of the `reward` and `action`
        reward_history.append(reward)
        action_history.append(action)
        
        if step_count >= num_steps:
            done = True

    return reward_history, action_history

# ------------------------------------- function to convert action vector to filter --------------------------------------
def action2IIR(action, order:int, zero_magnitude_mapping:Callable[[float], float]=None, gradient:float=None):
    """
    Creates an IIR filter from a given action vector considering the specified filter order. 
    
    :param action: action vector
    :param order: filter order; must be a positive integer greater than 2
    :param zero_magnitude_mapping: a callable function to transform the zero's magnitude (see the 'notebook-stage_2.3.ipynb'); \
        default is None, indicating no mapping function would be used
    :param gradient: gradient argument of the ``zero_magnitude_mapping`` function, if required; \
        default is None, indicating that the ``zero_magnitude_mapping`` function does not need a gradient argument
    """

    # ----- create the filter -----
    if order % 2 == 0:
        fix_zeros_magnitude_ = automatic_gain_ = False
        if (len(action) == 3*order//2+1): fix_zeros_magnitude_ = True
        elif (len(action) == 2*order): automatic_gain_ = True
        elif (len(action) == 3*order//2): fix_zeros_magnitude_ = automatic_gain_ = True

        shift = order // 2
        zs = []; ps = [][:]
        for i in range(order // 2):
            # extract the zero and pole
            if (fix_zeros_magnitude_):
                if (1 - EPS <= action[i + shift]): action[i + shift] = 1 - EPS # clipping the pole magnitudes
                z = polar2cmplx(1, np.pi * action[i])
                p = polar2cmplx(action[i + shift], np.pi * action[i + 2*shift])
            elif (zero_magnitude_mapping == None):
                # no zero magnitude mapping function is required (defaults to linear)
                if (1 - EPS <= action[i + 2*shift]): action[i + 2*shift] = 1 - EPS # clipping the pole magnitudes
                z = polar2cmplx(action[i], np.pi * action[i + shift])
                p = polar2cmplx(action[i + 2*shift], np.pi * action[i + 3*shift])
            elif (gradient == None):
                # a mapping function is used for zeros' magnitudes, but no gradient argument is required
                if (1 - EPS <= action[i + 2*shift]): action[i + 2*shift] = 1 - EPS # clipping the pole magnitudes
                z = polar2cmplx(zero_magnitude_mapping(action[i]), np.pi * action[i + shift])
                p = polar2cmplx(action[i + 2*shift], np.pi * action[i + 3*shift])
            else:
                # a mapping function is used for zeros' magnitudes with gradient
                if (1 - EPS <= action[i + 2*shift]): action[i + 2*shift] = 1 - EPS # clipping the pole magnitudes
                z = polar2cmplx(zero_magnitude_mapping(action[i], gradient), np.pi * action[i + shift])
                p = polar2cmplx(action[i + 2*shift], np.pi * action[i + 3*shift])

            # check whether the pole locates on (1 + 0j)
            if (p == 1+0j or p == 1-0j):
                p = p - EPS

            zs += [z, np.conjugate(z)] # array of zeros of TF
            ps += [p, np.conjugate(p)] # array of poles of TF
    
    else:
        raise Exception(f"the environment still does not support odd order IIR filters: given {order}")

    # find the transfer function of the filter with UNITY GAIN
    b, a = zpk2tf(zs, ps, 1)

    if (automatic_gain_):
        # estimate k
        k = 1/(freqz(b, a, worN=[0], fs=SAMPLING_FREQ)[1][0].real + EPS)
    else:
        k = action[-1]
    b *= k

    return b, a, (zs, ps, k) 

# --------------------------------------------- pole-zero plotting functions ---------------------------------------------
def pole_zero_plot(
        zs: list[complex], 
        ps: list[complex], 
        ax, # projection must be set to 'polar'
        ref_zs: list[complex] = [],
        ref_ps: list[complex] = [],
        plot_zp_vlines: bool = False,
        title: str = "Pole-Zero Plot of the Learned Filter"
    ):
    """
    Plots the given zeros and poles on the provided axis object along with references (ideal ploe-zero locations).

    Returns three artists to plot zeros, poles, and the reward text respectively. 
    """
    
    # plotting the ideal locations for zeros -> references
    ax.plot(np.angle(ref_zs), np.abs(ref_zs), 'o', color='g', markersize=9)
    ax.plot(np.angle(ref_ps), np.abs(ref_ps), 'x', color='c', markersize=9)
    ax.vlines(np.angle([ref_zs[i] for i in range(0, len(ref_zs), 2) if ref_zs[i] != 0]), ymin=0, ymax=1, color='g', linestyle='--', linewidth=1)
    ax.vlines(np.angle([ref_ps[i] for i in range(0, len(ref_ps), 2) if ref_ps[i] != 0]), ymin=0, ymax=1, color='c', linestyle='--', linewidth=1)

    # plotting zeros and poles
    z_artist = ax.plot(np.angle(zs), np.abs(zs), 'o', color='r', markersize=9)[0]
    p_artist = ax.plot(np.angle(ps), np.abs(ps), 'x', color='b', markersize=9)[0]

    if plot_zp_vlines:
        ax.vlines(np.angle([zs[i] for i in range(0, len(zs), 2) if zs[i] != 0]), ymin=0, ymax=1, color='r', linestyle='--', linewidth=1)
        ax.vlines(np.angle([ps[i] for i in range(0, len(zs), 2) if ps[i] != 0]), ymin=0, ymax=1, color='b', linestyle='--', linewidth=1)

    # add a text
    text = ax.text(x=0.9, y=1.8, s="", fontsize='large')

    # setting the axis configuration
    ax.set_rmax(1.5)
    ax.set_rticks([1])
    ax.set_rlabel_position(-22.5)
    ax.grid(True)
    ax.set_title(title, fontsize='x-large')

    return z_artist, p_artist, text

def pole_zero_plot_video(action_vector_arr, reward_arr, order, interference_center_freqs, file_name, fps=20, frames=None, lr=None, noise=None, dir_path=''):
    """
    Creates a video using the action vectors generated by a model during its training to show the traversal of poles and zeros of the learned IIR filter. \
    The generated video would be saved as a .mp4 file in the specified directory with the provided file name. 

    :param action_vector_arr: the array containing the actions the model took while training
    :param reward_arr: an array containing the rewards observed for each action
    :param order: IIR filter order
    :param interference_center_freqs: the interference center frequencies to place ideal zero locations on the unit circle
    :param file_name: name of the video file without the .mp4 extension
    :param fps: frames per second of the video; defaults to 20
    :param frames: the number of frames to visualize in the ``action_vector_arr``; defaults to None, indicating frames would be set to the length of the ``action_vector_arr``
    :param lr: learning rate of the model to be displayed on the video (can be a string in a format specifying the learning rates of both the networks, actor and critic)
    :param noise: exploration noise of the model to be displayed on the video
    :param dir_path: the path to the directory to save the video
    """

    if dir_path != '' and not os.path.exists(dir_path):
        raise Exception(f"the directory specified by the path does not exist: given '{dir_path}'")
    file_path = os.path.join(dir_path, file_name + '.mp4')

    # create the initial plot
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})

    b, a, (zs, ps, k) = action2IIR(action_vector_arr[0], order)

    ideal_zs = []
    for freq in interference_center_freqs:
        ideal_z = polar2cmplx(1, 2 * np.pi * (freq/SAMPLING_FREQ))
        ideal_zs += [ideal_z, np.conjugate(ideal_z)]
    z_artist, p_artist, text = pole_zero_plot(zs, ps, ax, ref_zs=ideal_zs, title="Training Pole-Zero Traversal")

    def move_poles_and_zeros(frame):

        _, _, (zs, ps, _) = action2IIR(action_vector_arr[frame], order)
        z_artist.set_xdata([np.angle(z) for z in zs])
        z_artist.set_ydata([np.abs(z)   for z in zs])

        p_artist.set_xdata([np.angle(p) for p in ps])
        p_artist.set_ydata([np.abs(p)   for p in ps])
        # THERE ARE TWO MORE ARTISTS TO PLOT THE REFERENCE ZEROS AND POLES

        annotation = f"n: {frame} \nSNR: {round(reward_arr[frame], 3)}"
        if lr is not None: annotation += f" \nlr: {lr}"
        if noise is not None: annotation += f" \nnoise: {noise}"
        text.set_text(annotation)

        return (z_artist, p_artist, text)
    
    ani = animation.FuncAnimation(fig=fig, func=move_poles_and_zeros, frames=frames if frames is not None else len(action_vector_arr))
    ani.save(file_path, writer='ffmpeg', fps=fps)

    print(file_path)

# ------------------------------------------- zero magnitude mapping functions -------------------------------------------
linear = lambda x: x
tan_mapping = lambda x: np.tan(np.pi/2 * x)
scaled_linear_mapping = lambda x, A=2: A*x

def conditional_linear(x, m=2, break_pt1=0.2, break_pt2=0.8, A=5):
    if (break_pt1 < x <= break_pt2):
        return m * (x - 0.5) + 1
    elif (x <= break_pt1):
        return (m * (break_pt1 - 0.5) + 1) / break_pt1 * x
    else: # break_pt2 < x 
        return (A - m * (break_pt2 - 0.5) - 1) / (1 - break_pt2) * (x - break_pt2) + m * (break_pt2 - 0.5) + 1
    
rational_mapping = lambda x: x/(1-x+EPS)
parameterized_rational_mapping = lambda x, m: x**(m/4) / (1-x+EPS)**(m/4)

# ----------------------------------------------- stage specific functions -----------------------------------------------
def polar2cmplx(r, theta):
    """
    Creates a complex number in the format a+bj when radius and angle in polar form are given.
    :param r: absolute value/modulus/magnitude of the complex number
    :param theta: argument/angle of the complex number

    Returns the corresponding complex number (in the format a+bj). 
    """
    return r * np.exp((0+1j) * theta)

def NormalizeState(state):
    """
    Normalizes a given state (equivalent to performing layer normalization)
    :param state: the state (observation) (1-D) vector to normalize
    """

    return (state - np.mean(state)) / np.std(state)

# new power function 
def power(signal):
  return np.sum(np.array(signal, dtype='float64')**2)
