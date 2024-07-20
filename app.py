import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr

def plot_spectrogram(y, sr, title):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(4, 2))
    img = librosa.display.specshow(D, x_axis='time', y_axis='hz', sr=sr, ax=ax)
    ax.set_title(title, fontsize=10)
    plt.colorbar(img, ax=ax, format="%+2.f dB", shrink=0.7)
    plt.tight_layout()
    return fig

st.title("Audio Separation Demo")

st.header("LibriSpeech_2spk_seen_during_training")

# Paths to your audio files
mix_path = "trainSet/mix.wav"
wav1_path = "trainSet/src1.wav"
wav2_path = "trainSet/src2.wav"

# Create three columns
col1, col2, col3 = st.columns(3)

# Original Audio - Mixed
with col1:
    st.subheader("Original Audio - Mixed")
    if os.path.exists(mix_path):
        y_mixed, sr_mixed = load_audio(mix_path)
        st.audio(mix_path)
        st.pyplot(plot_spectrogram(y_mixed, sr_mixed, "Original - Mixed"))
    else:
        st.error(f"File not found: {mix_path}")

# Separated Audio - Speaker 1
with col2:
    st.subheader("Original Audio - Speaker 1")
    if os.path.exists(wav1_path):
        y_speaker1, sr_speaker1 = load_audio(wav1_path)
        st.audio(wav1_path)
        st.pyplot(plot_spectrogram(y_speaker1, sr_speaker1, "Speaker 1"))
    else:
        st.error(f"File not found: {wav1_path}")

# Separated Audio - Speaker 2
with col3:
    st.subheader("Original Audio - Speaker 2")
    if os.path.exists(wav2_path):
        y_speaker2, sr_speaker2 = load_audio(wav2_path)
        st.audio(wav2_path)
        st.pyplot(plot_spectrogram(y_speaker2, sr_speaker2, "Speaker 2"))
    else:
        st.error(f"File not found: {wav2_path}")


wham_mix_path = "trainSet/mix.wav"
wham_wav1_path = "trainSet/sep1.wav"
wham_wav2_path = "trainSet/sep2.wav"

# Create three columns for the second row
col4, col5, col6 = st.columns(3)

# WHAM Original Audio - Mixed
with col4:
    st.subheader("Original Audio - Mixed")
    if os.path.exists(wham_mix_path):
        y_wham_mixed, sr_wham_mixed = load_audio(wham_mix_path)
        st.audio(wham_mix_path)
        st.pyplot(plot_spectrogram(y_wham_mixed, sr_wham_mixed, "WHAM Original - Mixed"))
    else:
        st.error(f"File not found: {wham_mix_path}")

# WHAM Separated Audio - Speaker 1
with col5:
    st.subheader("Separated Audio - Speaker 1")
    if os.path.exists(wham_wav1_path):
        y_wham_speaker1, sr_wham_speaker1 = load_audio(wham_wav1_path)
        st.audio(wham_wav1_path)
        st.pyplot(plot_spectrogram(y_wham_speaker1, sr_wham_speaker1, "WHAM Speaker 1"))
    else:
        st.error(f"File not found: {wham_wav1_path}")

# WHAM Separated Audio - Speaker 2
with col6:
    st.subheader("Separated Audio - Speaker 2")
    if os.path.exists(wham_wav2_path):
        y_wham_speaker2, sr_wham_speaker2 = load_audio(wham_wav2_path)
        st.audio(wham_wav2_path)
        st.pyplot(plot_spectrogram(y_wham_speaker2, sr_wham_speaker2, "WHAM Speaker 2"))
    else:
        st.error(f"File not found: {wham_wav2_path}")


st.header("LibriSpeech_2spk_not_seen_during_training")


mix_path = "testSet/mix.wav"
wav1_path = "testSet/src1.wav"
wav2_path = "testSet/src2.wav"

# Create three columns
col1, col2, col3 = st.columns(3)

# Original Audio - Mixed
with col1:
    st.subheader("Original Audio - Mixed")
    if os.path.exists(mix_path):
        y_mixed, sr_mixed = load_audio(mix_path)
        st.audio(mix_path)
        st.pyplot(plot_spectrogram(y_mixed, sr_mixed, "Original - Mixed"))
    else:
        st.error(f"File not found: {mix_path}")

# Separated Audio - Speaker 1
with col2:
    st.subheader("Original Audio - Speaker 1")
    if os.path.exists(wav1_path):
        y_speaker1, sr_speaker1 = load_audio(wav1_path)
        st.audio(wav1_path)
        st.pyplot(plot_spectrogram(y_speaker1, sr_speaker1, "Speaker 1"))
    else:
        st.error(f"File not found: {wav1_path}")

# Separated Audio - Speaker 2
with col3:
    st.subheader("Original Audio - Speaker 2")
    if os.path.exists(wav2_path):
        y_speaker2, sr_speaker2 = load_audio(wav2_path)
        st.audio(wav2_path)
        st.pyplot(plot_spectrogram(y_speaker2, sr_speaker2, "Speaker 2"))
    else:
        st.error(f"File not found: {wav2_path}")


wham_mix_path = "testSet/mix.wav"
wham_wav1_path = "testSet/sep1.wav"
wham_wav2_path = "testSet/sep2.wav"

# Create three columns for the second row
col4, col5, col6 = st.columns(3)

# WHAM Original Audio - Mixed
with col4:
    st.subheader("Original Audio - Mixed")
    if os.path.exists(wham_mix_path):
        y_wham_mixed, sr_wham_mixed = load_audio(wham_mix_path)
        st.audio(wham_mix_path)
        st.pyplot(plot_spectrogram(y_wham_mixed, sr_wham_mixed, "WHAM Original - Mixed"))
    else:
        st.error(f"File not found: {wham_mix_path}")

# WHAM Separated Audio - Speaker 1
with col5:
    st.subheader("Separated Audio - Speaker 1")
    if os.path.exists(wham_wav1_path):
        y_wham_speaker1, sr_wham_speaker1 = load_audio(wham_wav1_path)
        st.audio(wham_wav1_path)
        st.pyplot(plot_spectrogram(y_wham_speaker1, sr_wham_speaker1, "WHAM Speaker 1"))
    else:
        st.error(f"File not found: {wham_wav1_path}")

# WHAM Separated Audio - Speaker 2
with col6:
    st.subheader("Separated Audio - Speaker 2")
    if os.path.exists(wham_wav2_path):
        y_wham_speaker2, sr_wham_speaker2 = load_audio(wham_wav2_path)
        st.audio(wham_wav2_path)
        st.pyplot(plot_spectrogram(y_wham_speaker2, sr_wham_speaker2, "WHAM Speaker 2"))
    else:
        st.error(f"File not found: {wham_wav2_path}")



