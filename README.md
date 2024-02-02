# region-correspondence

This repository implements methods to obtain dense correspondence from region correspondence. 
Region correspondence is a higher-level correspondence represented by multiple, paired (corresponding) region of interest (ROI) masks.
3D and 2D ROIs are supported.  


```python
conda create -n roi2ddf numpy pytorch pytorch-cuda=11.8 -c pytorch -c nvidia && 
conda activate roi2ddf && 
pip install nibabel pillow  # for file io
```