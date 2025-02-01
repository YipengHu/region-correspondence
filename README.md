# region-correspondence

This repository implements methods to obtain dense correspondence from region correspondence. 
Region correspondence is a higher-level correspondence represented by multiple, paired (corresponding) region of interest (ROI) masks.
3D and 2D ROIs are supported.  


```python
conda create -n roi2ddf  && 
conda activate roi2ddf  && 
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia  && #TODO: update to pip install going forward
pip install nibabel pillow  # for file io
```