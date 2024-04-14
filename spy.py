#!/usr/bin/env python3
import socket
import click
import speech_recognition as sr
import whisper
from whisper_mic import WhisperMic
import torch
from typing import Optional
model = whisper.load_model("base")
# obtain path to "english.wav" in the same folder as this script
from os import path

AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "english.wav")
socket_path = "/tmp/voice.socket"
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(socket_path)


@click.command()
@click.option("--model", default="base", help="Model to use",
              type=click.Choice(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]))
@click.option("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="Device to use",
              type=click.Choice(["cpu", "cuda", "mps"]))
@click.option("--english", default=False, help="Whether to use English model", is_flag=True, type=bool)
@click.option("--verbose", default=False, help="Whether to print verbose output", is_flag=True, type=bool)
@click.option("--pause", default=0.8, help="Pause time before entry ends", type=float)
@click.option("--save_file", default=False, help="Flag to save file", is_flag=True, type=bool)
@click.option("--loop", default=False, help="Flag to loop", is_flag=True, type=bool)
@click.option("--dictate", default=True, help="Flag to dictate (implies loop)", is_flag=True, type=bool)
@click.option("--mic_index", default=9, help="Mic index to use", type=int)
@click.option("--list_devices", default=False, help="Flag to list devices", is_flag=True, type=bool)
@click.option("--faster", default=False, help="Use faster_whisper implementation", is_flag=True, type=bool)
@click.option("--hallucinate_threshold", default=400,
              help="Raise this to reduce hallucinations.  Lower this to activate more often.", is_flag=True, type=int)
def main(model: str, english: bool, verbose: bool, pause: float, save_file: bool,
         device: str, loop: bool, dictate: bool, mic_index: Optional[int], list_devices: bool, faster: bool,
         hallucinate_threshold: int) -> None:
    print("Possible devices: ", sr.Microphone.list_microphone_names())
    mic = WhisperMic(model=model, english=english, verbose=verbose, pause=pause,
                     save_file=save_file, device=device, mic_index=mic_index,
                     implementation=("faster_whisper" if faster else "whisper"),
                     hallucinate_threshold=hallucinate_threshold)

    try:
        while True:
            result = mic.listen(timeout=5, phrase_time_limit=5)
            print("You said: " + result)
    except KeyboardInterrupt:
        print("Operation interrupted successfully")
    finally:
        if save_file:
            mic.file.close()


# Close the connection
# use the audio file as the audio source

print("TESTING")
main()
exit(1)

r = sr.Recognizer()
with sr.Microphone() as source:
    while True:
        print("Listening...")
        try:
            audio = r.listen(source=source, phrase_time_limit=5, timeout=5)
            transcription = whisper.transcribe(audio=audio, model=model)
            in_data = r.recognize_sphinx(audio)
            print("<<" + in_data)
            client.sendall(in_data.encode())
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))
        except sr.WaitTimeoutError:
            print("nothing happening....")
        except BlockingIOError:
            print("reconnecting...")
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(socket_path)
        except Exception as e:
            print(str(e))
client.close()
