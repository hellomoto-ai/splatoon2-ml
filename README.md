# VAE-GAN on Splatoon 2

VAE-GAN code and model trained on Splatoon 2 game screens.

![](https://raw.githubusercontent.com/hellomoto-ai/splatoon2-ml/62adac7364ae03caae9b5b1fde46279857aaaed4/assets/2019-04-11/images/equip.gif?raw=true "Equip")
![](https://raw.githubusercontent.com/hellomoto-ai/splatoon2-ml/62adac7364ae03caae9b5b1fde46279857aaaed4/assets/2019-04-11/images/turf_battle.gif?raw=true "Turf Battle")
![](https://raw.githubusercontent.com/hellomoto-ai/splatoon2-ml/62adac7364ae03caae9b5b1fde46279857aaaed4/assets/2019-04-11/images/green_and_orange.gif?raw=true "Green and Orange")
![](https://raw.githubusercontent.com/hellomoto-ai/splatoon2-ml/62adac7364ae03caae9b5b1fde46279857aaaed4/assets/2019-04-11/images/map.gif?raw=true "Map")
![](https://raw.githubusercontent.com/hellomoto-ai/splatoon2-ml/62adac7364ae03caae9b5b1fde46279857aaaed4/assets/2019-04-11/images/splash.gif?raw=true "Splash")

![](https://raw.githubusercontent.com/hellomoto-ai/splatoon2-ml/62adac7364ae03caae9b5b1fde46279857aaaed4/assets/2019-04-11/images/yellow.gif?raw=true "Yellow")
![](https://raw.githubusercontent.com/hellomoto-ai/splatoon2-ml/62adac7364ae03caae9b5b1fde46279857aaaed4/assets/2019-04-11/images/purple_and_green.gif?raw=true "Purple and Green")
![](https://raw.githubusercontent.com/hellomoto-ai/splatoon2-ml/62adac7364ae03caae9b5b1fde46279857aaaed4/assets/2019-04-11/images/finish.gif?raw=true "Finish line")
![](https://raw.githubusercontent.com/hellomoto-ai/splatoon2-ml/62adac7364ae03caae9b5b1fde46279857aaaed4/assets/2019-04-11/images/winered_and_green.gif?raw=true "Winered and Green")
![](https://raw.githubusercontent.com/hellomoto-ai/splatoon2-ml/62adac7364ae03caae9b5b1fde46279857aaaed4/assets/2019-04-11/images/judges.gif "Judges")

Check out the detail from [here](https://hellomoto-ai.github.io/splatoon2-ml/).

## Requirements

- Python > 3.5

### Dependencies
- torch
- cv2
- imageio
- numpy

### Dataset

Due to copy right concerns, we cannot share the data set used to train the model. 
But there are a lot of videos available online. Once you get a list of streaming links, such as m3u8 playlist, you can download them to your local file system using ffmpeg.

```shell
ffmpeg -y -i "${url}" -c copy "${output_path}"
```

We only used keyframes so that we do not have to worry about the best frame rate to extract frames. With the following command, only keyframes are saved.

```shell
ffmpeg -y -skip_frame nokey -i "${url}" -vsync 0 "${output_prefix}-%06d.png"
```

In the end we had 296 different videos, which are around 3 to 5 minutes, with 44676 key frames. By splitting them into 9:1, we obtained training data which consist of 40620 frames from 269 different videos, and test data which consists of 4056 frames from 27 different videos.

## Usage

```
./train_vae_gan.py --train-flist TRAIN_FLIST --test-flist TEST_FLIST --data-dir DATA_DIR
```
