
[Keras]Protein Interface Prediction using Graph Convolutional Networks
===
Unofficial Keras implementation of Protein Interface Prediction using Graph Convolutional Networks[1].   

## Usage
```
$ unsip dataset.zip
$ python run.py
```

## Dataset
Number of samples
-------------------------------
|Set|Complex|Positive|Negative|
|:--|------:|-------:|-------:|
|Training|140|12866 (0.1%)|128660 (90.9%)|
|Validation|35|3138 (0.2%)|31380 (99.8%)|
|Test|55|4871 (0.1%)|4953446 (99.9%)|

## Result
- AUC(0.86)

## Requirements
- TensorFlow 2.3.0

## Reference
[1] Fout, Alex M. <a href="https://mountainscholar.org/handle/10217/185661">Protein interface prediction using graph convolutional networks</a>. Diss. Colorado State University, 2017.  
[2] Supplementary Data for NIPS Publication: Protein Interface Prediction using Graph Convolutional Networks. https://zenodo.org/record/1127774#.WkLewGGnGcY