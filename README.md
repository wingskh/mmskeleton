# Action Recognition Using mmskeleton

## Installation

Follow the below link to install mmskeleton, mmdetection and openpose.
(Must use "python setup.py develop --mmdet" to install the older version of mmdetection)
https://github.com/open-mmlab/mmskeleton

### Suggested version

pip install mmcv==0.4.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install pytorch==1.3.0 torchvision cudatoolkit=10.1 -c pytorch

## Instruction

1. Rescale the training video to 340\*256 and set frame rate to 30fps
   Useful ffmpeg comman under video folder:

   ```
   for i in *.mp4; do ffmpeg -i "$i" -vf scale=340:256 "[path to output folder]/${i}"; done
   for i in *.mp4; do ffmpeg -i "$i" -filter:v fps=fps=30 "[path to output folder]/${i}"; done
   ```

2. Put the training video to ./resource/data_example/

3. Modify the ./resource/data_example/category_annotation_example to add labels of different videos

4. run the command below in ./ to extract the skeleton data of the videos
   The extracted skeleton can be found in ./data/dataset_example/

   ```
   mmskl configs/utils/build_dataset_example.yaml --gpus [num of gpus]
   ```

5. Run the command below in ./ to train a new model.

   ```
   mmskl configs/recognition/st_gcn/dataset_example/train.yaml
   ```

6. Run the command below in ./ to train a new model.

   ```
   mmskl configs/recognition/st_gcn/dataset_example/test.yaml
   ```

7. Run the command below in ./ to get the visualized video.

   a. If you want to get a visualized video, you can run the command below.
   The default path of the visualized video: "./visualized_video.avi"
   You the modify the file in "./mmskeleton/processor/demo_offline.py" in line 83.

   ```
   python main.py demo_offline --openpose "[path to openpose build folder]" --video "[path to target video]"
   ```

   b. If you want to get a classification result of lists of video, you can run the command below after change variable visualization in "./mmskeleton/processor/demo_offline.py" in line 33 to False.
   The default path of the classification result: "./label_output.csv"
   You the modify the file in "./mmskeleton/processor/demo_offline.py" in line 43.

   ```
   python main.py demo_offline --openpose "[path to openpose build folder]" --video "[path to target folder]"
   ```
