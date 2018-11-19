# Acoustic Novelty Detection

Tenforflow implementation of the Acoustic Novelty Detection using Recurent Neural Networks with Stochastic Layers (RNNSLs).

Duong Nguyen, Oliver S. Kirsebom, Fábio Frazão, Ronan Fablet and Stan Matwin; "Recurrent Neural Networks with Stochastic Layers
for Acoustic Novelty Detection", submitted to ICASSP 2019.

## Credits
We use the FIVO implementation of the Tensorflow Research Group (https://github.com/tensorflow/models/blob/master/research/fivo) for the VRNN training phase of this code.

## Prerequisites
- Numpy 1.15.2 (Python 2.7)
- [Tensorflow](http://tensorflow.org) 1.8.0 
- [scipy](https://www.scipy.org/)
- [sonnet](https://github.com/deepmind/sonnet)

## Datasets
The dataset is provided by [A3Lab](http://www.a3lab.dii.univpm.it) and is availble at http://a3lab.dii.univpm.it/webdav/audio/Novelty_Detection_Dataset.tar.gz.

## Preprocess the Data


## Training
First we need to train the model:
```
python run_audidoNovelty.py \
       --mode=train \
       --model=vrnn \
       --data_dimension=160 \
       --latent_size=160 \
       --batch_size=4
```


## Anomaly Detection

## Acknowledgments
We would like to thank the Tensorflow Research Group for the implementation of VRNN and A3Lab for the Dataset.

## Contact
This codebase is maintained by Duong NGUYEN (van.nguyen1@imt-atlantique.fr). For questions and issues please open an issue on the issues tracker.




