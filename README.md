# FINet: Feature Interactions Across Dimensions and Hierarchies for Camera Localization

This is PyTorch implementation of our visual localization framework **FINet**.

## Environment

In order to run this repository you will need:

1. Python3 (tested with Python 3.7.5)
2. PyTorch (tested with version 1.6.0 and 1.7.1)
3. Download the [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset and the [Cambridge Landmarks](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset) dataset, and put them in the `data` directory.

* You can directly install PyTorch and other Python packages by running 

```
pip install -r requirements.txt
```

* If you use Conda more often, you can create the environment for FINet by running 

```
conda create -f environment.yml
conda activate FINet
```

## Testing

The inference script is the test.py in the root directory. The trained models for partial experiments need to be download from [here](https://drive.google.com/drive/folders/1XDAc2dB9tWunlX9cONllmRSOS1GKL2yp?usp=sharing) to the corresponding folder in the `checkpoints` directory. 

- Example of testing the model saved at epoch `500` on the `Stairs` scenario in the `7-Scenes` dataset.

```
python test.py --dataset 7Scenes --scene stairs --which_epoch 500
```

## Training

The executable script is `train.py` in the root directory. Before training, all datasets needs to be normalized , and the Cambridge Landmarks dataset need to be rescaled. You can achieve these things with `pretreatment.sh` script in the `data` directory.

* Example of training on the `Stairs` scenario in the `7-Scenes` dataset.

```
python train.py --dataset 7Scenes --scene stairs
```

## Acknowledgements

Our code is partially built on [geomapnet](https://github.com/NVlabs/geomapnet).

