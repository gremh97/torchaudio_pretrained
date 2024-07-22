# TORCHAUDIO TACOTRON2 w/ Various Vocoders

## ENV Set
```bash
cat requirements.txt | xargs -n 1 pip install           # Linux Centos
```

For macOS only specific version of torch, torchaudio can execute .py
```bash
 conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch
```

## RUN
```bash
python3  tacotrontest.py
```

Just use colab environment, if the the script doesn't work well :(  
    [click here to see more info](https://github.com/facebookresearch/demucs/issues/570)



## Reference
https://pytorch.org/audio/stable/pipelines.html#module-torchaudio.pipelines