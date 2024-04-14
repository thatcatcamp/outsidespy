import json
import os
import whisper
import pyaudio
import wave
import time
import paho.mqtt.client as mqtt
from ctypes import *
from contextlib import contextmanager
import pyaudio
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit integer)
CHANNELS = 1  # Number of channels (1 for mono)
SAMPLE_RATE = 44100  # Sampling rate in Hz
RECORD_SECONDS = 5  # Duration of each recording

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

def on_publish(client, userdata, mid, reason_code, properties):
    # reason_code and properties will only be present in MQTTv5. It's always unset in MQTTv3
    try:
        userdata.remove(mid)
    except KeyError:
        print("on_publish() is called with a mid not present in unacked_publish")
        print("This is due to an unavoidable race-condition:")
        print("* publish() return the mid of the message sent.")
        print("* mid from publish() is added to unacked_publish by the main thread")
        print("* on_publish() is called by the loop_start thread")
        print("While unlikely (because on_publish() will be called after a network round-trip),")
        print(" this is a race-condition that COULD happen")
        print("")
        print("The best solution to avoid race-condition is using the msg_info from publish()")
        print("We could also try using a list of acknowledged mid rather than removing from pending list,")
        print("but remember that mid could be re-used !")


def record_audio(filename):
    p = pyaudio.PyAudio()  # Create PyAudio instance

    # Open audio stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Recording started")
    frames = []

    # Record audio for specified duration
    for i in range(0, int(SAMPLE_RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Recording finished")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save audio to a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

unacked_publish = set()
mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_publish = on_publish

mqttc.user_data_set(unacked_publish)
mqttc.connect("localhost")
mqttc.loop_start()

model = whisper.load_model("base")

while True:
    with noalsaerr() as n:
        record_audio("/tmp/out.wav")
    audio = whisper.load_audio("/tmp/out.wav")
    audio = whisper.pad_or_trim(audio)
    options = whisper.DecodingOptions()
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    result = whisper.decode(model, mel, options)
    print("raw heard>>", result.text)
    # ignore empty, ignore guesses, ignore non-ascii (sometimes db languages, usually noise), and strings too short to do stuff with
    if result.text != "" and result.no_speech_prob > .50 and result.text.isascii() and len(result.text) > 10:
        print("heard >>", result)
        packet = {"type": "whisper", "value": result.text}
        msg_info = mqttc.publish("eventq", json.dumps(packet), qos=1)
        msg_info.wait_for_publish()
    else:
        print("quiet...")


exit(1)
# Set up the audio stream
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

while True:
    # Record audio
    frames = []
    data = stream.read(8096)
    frames.append(data)
    wf = wave.open('output.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()
    os.remove('output.mp3')
    import subprocess
    subprocess.call(['ffmpeg', '-i', 'output.wav', 'output.mp3'])
    print('saved...')
    audio = whisper.load_audio("output.mp3")
    audio = whisper.pad_or_trim(audio)
    options = whisper.DecodingOptions()
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    result = whisper.decode(model, mel, options)
    print(result.t)


# Stop the audio stream
stream.stop_stream()
stream.close()
p.terminate()

