import torch
import torchaudio

torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__)
print(torchaudio.__version__)
print(device)

import os
import matplotlib.pyplot as plt

"""
The text-to-speech pipeline goes as follows:

1. Text preprocessing

    First, the input text is encoded into a list of symbols. In this tutorial, we will use English characters and phonemes as the symbols.

2. Spectrogram generation

    From the encoded text, a spectrogram is generated. We use the Tacotron2 model for this.

3. Time-domain conversion

    The last step is converting the spectrogram into the waveform. The process to generate speech from spectrogram is also called a Vocoder.
    Once the spectrogram is generated, the last process is to recover the waveform from the spectrogram using a vocoder. `torchaudio` provides vocoders based on GriffinLim and WaveRNN.
    `Tacotron2.infer` method perfoms multinomial sampling, therefore, the process of generating the spectrogram incurs randomness.

"""

"""
Experiments : https://colab.research.google.com/drive/18XqwFNRuw9iUepGFnGn4g2SVuz4ftjfZ#scrollTo=mjVxbFAznoxM
"""

def character_based_encoder(text, print_result=False):
    """ 
    ####### How text_processor works #######
    symbols = "_-!'(),.:;? abcdefghijklmnopqrstuvwxyz"
    look_up = {s: i for i, s in enumerate(symbols)}
    symbols = set(symbols)

    def text_to_sequence(text):o
        text = text.lower()
        return [look_up[s] for s in text if s in symbols]

    text = "Hello world! Text to speech!"
    print(text_to_sequence(text))
    # [19, 16, 23, 23, 26, 11, 34, 26, 29, 23, 15, 2, 11, 31, 16, 35, 31, 11, 31, 26, 11, 30, 27, 16, 16, 14, 19, 2]

    """
    processor = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH.get_text_processor()
    processed, lengths = processor(text)

    if print_result: 
        print(processed)
        print(lengths)
        print([processor.tokens[i] for i in processed[0, : lengths[0]]])
    
    return processed, lengths


def phoneme_based_encoder(text, print_result=False):
    processor = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH.get_text_processor()

    with torch.inference_mode():
        processed, lengths = processor(text)

    if print_result:
        print(processed)
        print(lengths)
        print([processor.tokens[i] for i in processed[0, : lengths[0]]])
    
    return processed, lengths



class Tacotron2:
    def __init__(self, encoder="phoeneme", vocoder="wavernn"):
        vocoder = vocoder.lower()
        if  vocoder      == "wavernn":
            self.bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH if encoder == "Phoneme" else torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        elif vocoder     in ("griffinlim", "waveglow"):
            self.bundle = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_PHONE_LJSPEECH if encoder == "Phoneme" else torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH
        else:    
            raise ValueError(f"Unsupported Vocode: {vocoder}")
            
        self._processor  = self.bundle.get_text_processor()
        self._tacotron2  = self.bundle.get_tacotron2().to(device)
        if vocoder != "waveglow":
            self.isWaveGlow  = False
            self._vocoder    = self.bundle.get_vocoder().to(device)
        else:
            self.isWaveGlow = True
            waveglow = torch.hub.load(
                "NVIDIA/DeepLearningExamples:torchhub",
                "nvidia_waveglow",
                model_math="fp32",
                pretrained=False,
            )
            checkpoint = torch.hub.load_state_dict_from_url(
                "https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth",  # noqa: E501
                progress=False,
                map_location=device,
            )
            state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
            waveglow.load_state_dict(state_dict)
            waveglow = waveglow.remove_weightnorm(waveglow)
            self._vocoder = waveglow.to(device)
            


    def inference(self, text, output_dir="./", save_plot=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory {output_dir} created.")

        with torch.no_grad():
            processed, lengths = self._processor(text)
            processed = processed.to(device)
            lengths = lengths.to(device)
            spec, spec_lengths, _ = self._tacotron2.infer(processed, lengths)
            if not self.isWaveGlow:
                waveforms, lengths = self._vocoder(spec, spec_lengths)
                sample_rate        = self._vocoder.sample_rate
            else:
                waveforms          = self._vocoder.infer(spec)
                sample_rate        = 22050
            

        file_append = text.split(" ")[0].lower()
        tts_path = os.path.join(output_dir, f"{file_append}-tts.wav")

        torchaudio.save(tts_path, waveforms, sample_rate)
        # IPython.display.Audio(waveforms[0:1], rate=self._vocoder.sample_rate)       # ipynb
        if save_plot:
            self.plot_and_save(waveforms, spec, sample_rate, output_dir, file_append)


    def plot_and_save(self, waveforms, spec, sample_rate, output_dir="./", file_append=""):
        waveforms = waveforms.cpu().detach()
        
        # Create figure and subplots
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot waveform
        ax1.plot(waveforms[0])
        ax1.set_xlim(0, waveforms.size(-1))
        ax1.grid(True)
        ax1.set_title("Waveform")
        
        # Plot spectrogram
        ax2.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
        ax2.set_title("Spectrogram")
        
        # Save the figure containing both plots
        plot_path = os.path.join(output_dir, f"waveform_and_spectrogram_{file_append}.png")
        fig.savefig(plot_path)
        print(f"Plots saved: {plot_path}")

    
    def print_model_structure(self, output_dir="./"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory {output_dir} created.")

        with open(os.path.join(output_dir, "torchNN.txt"), "w") as f:
            f.write(f"{'='*50}Processor{'='*50}\n")
            f.write(f"{str(self._processor)}\n")
            f.write(f"{'='*50}Tacotron{'='*50}\n")
            f.write(f"{str(self._tacotron2)}\n")
            f.write(f"{'='*50}Vocoder{'='*50}\n")
            f.write(f"{str(self._vocoder)}\n")


if __name__ == "__main__":
    text = "Hello world! Text to speech!"
    model = Tacotron2(encoder="phoeneme", vocoder="wavernn")
    model.print_model_structure(output_dir="test")
    model.inference(text, output_dir="test", save_plot=True)