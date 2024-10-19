import asyncio
import importlib
import io
import logging
import multiprocessing
from idlelib.rpc import request_queue, response_queue

log = logging.getLogger(__name__)

import re
import tempfile
from argparse import Namespace
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchaudio
import tqdm
from cached_path import cached_path
from einops import rearrange
from pydub import AudioSegment, silence
from vocos import Vocos

from model import CFM, DiT, MMDiT, UNetT
from model.utils import (convert_char_to_pinyin, get_tokenizer,
                         load_checkpoint, save_spectrogram)

class TTSEngine:
    def __init__(self, reqQueue: multiprocessing.Queue, resQueue: multiprocessing.Queue, config):

        self.target_sample_rate = 24000
        self.n_mel_channels = 100
        self.hop_length = 256
        self.target_rms = 0.1
        self.nfe_step = 32  # 16, 32
        self.cfg_strength = 2.0
        self.ode_method = "euler"
        self.sway_sampling_coef = -1.0
        self.speed = 1.0
        self.fix_duration = None
        self.F5TTS_model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )

        self.max_chars = None
        self.ref_text = None
        self.ema_model = None

        self.sr = None
        self.audio = None
        self.vocos = None
        self.device = None
        cfg = importlib.import_module(config.configuration)
        self.cfg = Namespace(**{
            "ref_audio": cfg.ref_audio,
            "ref_text": cfg.ref_text,
            "listen_addr": "0.0.0.0:6000",
        })
        self.IsReady = True
        self.remove_silence = False
        self.req_queue = reqQueue
        self.res_queue = resQueue


    def load(self, config):
        try:

            self.device = "cuda" if torch.cuda.is_available()  else "mps" if torch.backends.mps.is_available() else "cpu"
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
            ref_audio_orig = self.cfg.ref_audio
            print("Converting reference audio if needed...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                aseg = AudioSegment.from_file(ref_audio_orig)

                non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50,
                                                           keep_silence=1000)
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    non_silent_wave += non_silent_seg
                aseg = non_silent_wave

                audio_duration = len(aseg)
                if audio_duration > 15000:
                    print("Audio is over 15s, clipping to only first 15s.")
                    aseg = aseg[:15000]
                aseg.export(f.name, format="wav")
                self.ref_audio = f.name

            print("Loading reference audio")
            self.audio, self.sr = torchaudio.load(self.ref_audio)
            if self.audio.shape[0] > 1:
                self.audio = torch.mean(self.audio, dim=0, keepdim=True)

            self.rms = torch.sqrt(torch.mean(torch.square(self.audio)))
            if self.rms < self.target_rms:
                self.audio = self.audio * self.target_rms / self.rms
            if self.sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(self.sr, self.target_sample_rate)
                audio = resampler(self.audio)
            self.audio = self.audio.to(self.device)
            self.ref_text = self.cfg.ref_text
            if len(self.ref_text[-1].encode('utf-8')) == 1:
                self.ref_text = self.ref_text + " "

            if not self.ref_text.endswith(". ") and not self.ref_text.endswith("。"):
                if self.ref_text.endswith("."):
                    self.ref_text += " "
                else:
                    self.ref_text += ". "
            self.max_chars = int(len(self.ref_text.encode('utf-8')) / (self.audio.shape[-1] / self.sr) * (
                        25 - self.audio.shape[-1] / self.sr))

            print(f"Using {self.device} device")


            self.ema_model = self.load_model("F5-TTS", "F5TTS_Base", DiT, self.F5TTS_model_cfg, 1200000)
            self.res_queue.put_nowait("LOADED")
        except Exception as ex:
            self.res_queue.put_nowait(f"ERROR {ex}")
            return
        while True:
            message = self.req_queue.get()
            if message["control"] == "EXIT":
                self.res_queue.put_nowait("EXIT")
                return
            try:
                self.res_queue.put_nowait(self.infer(message["say"]))
            except Exception as ex:
                logging.warning(f"cannot infer message: {ex}")


    def load_model(self, repo_name, exp_name, model_cls, model_cfg, ckpt_step):
        ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt" # .pt | .safetensors
        if not Path(ckpt_path).exists():
            ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
        vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
        model = CFM(
            transformer=model_cls(
                **model_cfg, text_num_embeds=vocab_size, mel_dim=self.n_mel_channels
            ),
            mel_spec_kwargs=dict(
                target_sample_rate=self.target_sample_rate,
                n_mel_channels=self.n_mel_channels,
                hop_length=self.hop_length,
            ),
            odeint_kwargs=dict(
                method=self.ode_method,
            ),
            vocab_char_map=vocab_char_map,
        ).to(self.device)

        model = load_checkpoint(model, ckpt_path, self.device, use_ema = True)

        return model


    def chunk_text(self, text):
        """
        Splits the input text into chunks, each with a maximum number of characters.
        Args:
            text (str): The text to be split.
            max_chars (int): The maximum number of characters per chunk.
        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        current_chunk = ""
        # Split the text into sentences based on punctuation followed by whitespace
        sentences = re.split(r'(?<=[;:,.!?])\s+|(?<=[；：，。！？])', text)

        for sentence in sentences:
            if len(current_chunk.encode('utf-8')) + len(sentence.encode('utf-8')) <= self.max_chars:
                current_chunk += sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def infer_batch(self, gen_text_batches, cross_fade_duration=0.15):
        generated_waves = []
        spectrograms = []

        for i, gen_text in enumerate(tqdm.tqdm(gen_text_batches)):
            # Prepare the text
            text_list = [self.ref_text + gen_text]
            final_text_list = convert_char_to_pinyin(text_list)

            # Calculate duration
            ref_audio_len = self.audio.shape[-1] // self.hop_length
            zh_pause_punc = r"。，、；：？！"
            ref_text_len = len(self.ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, self.ref_text))
            gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / self.speed)

            # inference
            with torch.inference_mode():
                generated, _ = self.ema_model.sample(
                    cond=self.audio,
                    text=final_text_list,
                    duration=duration,
                    steps=self.nfe_step,
                    cfg_strength=self.cfg_strength,
                    sway_sampling_coef=self.sway_sampling_coef,
                )

            generated = generated[:, ref_audio_len:, :]
            generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
            generated_wave = self.vocos.decode(generated_mel_spec.cpu())
            if self.rms < self.target_rms:
                generated_wave = generated_wave * self.rms / self.target_rms

            # wav -> numpy
            generated_wave = generated_wave.squeeze().cpu().numpy()

            generated_waves.append(generated_wave)
            spectrograms.append(generated_mel_spec[0].cpu().numpy())

        # Combine all generated waves with cross-fading
        if cross_fade_duration <= 0:
            # Simply concatenate
            final_wave = np.concatenate(generated_waves)
        else:
            final_wave = generated_waves[0]
            for i in range(1, len(generated_waves)):
                prev_wave = final_wave
                next_wave = generated_waves[i]

                # Calculate cross-fade samples, ensuring it does not exceed wave lengths
                cross_fade_samples = int(cross_fade_duration * self.target_sample_rate)
                cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

                if cross_fade_samples <= 0:
                    # No overlap possible, concatenate
                    final_wave = np.concatenate([prev_wave, next_wave])
                    continue

                # Overlapping parts
                prev_overlap = prev_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]

                # Fade out and fade in
                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)

                # Cross-faded overlap
                cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

                # Combine
                new_wave = np.concatenate([
                    prev_wave[:-cross_fade_samples],
                    cross_faded_overlap,
                    next_wave[cross_fade_samples:]
                ])

                final_wave = new_wave

            max_int24 = 2 ** 23 - 1
            final_wave_int32 = (np.clip(final_wave, -1.0, 1.0)*max_int24).astype(np.int32)
            final_wave_int32 = final_wave_int32.astype('<i4')
            num_samples = len(final_wave_int32)

            int32_bytes = final_wave_int32.view(np.uint8)
            int32_bytes_array = int32_bytes.reshape(num_samples, 4)
            int24_bytes_array = int32_bytes_array[:, :3]
            int24_bytes = int24_bytes_array.flatten()
            assert len(int24_bytes) == num_samples * 3, f"Data length {len(int24_bytes)} is not a multiple of 3."
            int24_bytes = int24_bytes.tobytes()
            # Remove silence
            aseg = AudioSegment(int24_bytes,
                    frame_rate=self.target_sample_rate,
                    sample_width=3,
                    channels=1)

            if self.remove_silence:
                non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    non_silent_wave += non_silent_seg
                aseg = non_silent_wave

            with io.BytesIO() as wav_buffer:
                aseg.export(wav_buffer, format='wav', codec="pcm_s24le")
                wav_bytes = wav_buffer.getvalue()

            return wav_bytes

    def infer(self, gen_text, cross_fade_duration=0.15):
        print(gen_text)

        gen_text_batches = self.chunk_text(gen_text)
        print('ref_text', self.ref_text)
        for i, gen_text in enumerate(gen_text_batches):
            print(f'gen_text {i}', gen_text)

        return self.infer_batch(gen_text_batches, cross_fade_duration)

    async def Say(self, text) -> bytes:
        self.req_queue.put({"control":"", "say": text})
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, self.res_queue.get)
        return res