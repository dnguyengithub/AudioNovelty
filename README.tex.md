# Acoustic Novelty Detection

Tenforflow implementation of the **Acoustic Novelty Detection using Recurent Neural Networks with Stochastic Layers** (RNNSLs).

*Duong Nguyen, Oliver S. Kirsebom, Fábio Frazão, Ronan Fablet and Stan Matwin; "Recurrent Neural Networks with Stochastic Layers for Acoustic Novelty Detection", submitted to ICASSP 2019.*

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
Process the raw *.wav files and create a set of TFRecords:
```
python audioNovelty/data/create_tfrecords.py \
       --raw_wav_dir="./datasets/" \
       --out_dir=""./datasets/"\
       --duration=3
```

For the training phase, the duration of the series should be small (3, 5 or 10s). For the test phase, the duration of the series should be long to reduce the effect of the "warming up phase" of VRNN (the initiation of the hidden states).
In the paper, we created a set of 3s-long series for the training set and a set of 30s-long series for the test/validation set.

## Training
First we need to train the model:
```
python run_audidoNovelty.py \
       --mode=train \
       --model=vrnn \
       --dataset_path="./datasets/train_3_160.tfrecord" \
       --data_dimension=160 \
       --latent_size=160 \
       --batch_size=4 \
```

## Anomaly Detection
After learning the distribution over the series in the training set, we calculate the log probability of each series in the test set to state the detection.

- Calculate the log probabilities in the validation set to choose the threshold:
```
python eval.py \
       --mode=test \
       --model=vrnn \
       --dataset_path="./datasets/valid_30_160.tfrecord" \
       --data_dimension=160 \
       --latent_size=160 \
       --batch_size=4 \
       --rerun_graph=True \
       --plot=False \
```

We will get:

```
Dataset: valid_30_160.tfrecord
Log probability: mean=67.93385, std=129.93369
```


The threshold is then set as: $\theta = m_{valid} - 3\sigma_{valid}$. 


We also set another threshold: *peak_threshold*, which marks the starting point of the abnormal events. Note that we evaluate $\log p(\boldsymbol{\mathrm{x}}_t \ \boldsymbol{\mathrm{x}}_{<t})$, at the beginning of the abnormal event, the historical information $\boldsymbol{\mathrm{x}}_{<t}$ encoded in the hidden states of the RNN and the latent variable is well known by the model, so $\log p(\boldsymbol{\mathrm{x}}_t \ \boldsymbol{\mathrm{x}}_{<t})$ will be very low. After that, the model updates its hidden states, however, these hidden states are "unstable" (because it starts to take in account the unknown abnormal event), $\log p(\boldsymbol{\mathrm{x}}_t\boldsymbol{\mathrm{x}}_{<t})$ will not be as low as the one at the starting point. This peak threshold is then set as: $\theta_p = m_{valid} - 5\sigma_{valid}$.


For the posprocessing step, we simply applied an minimum filter to the log probability sequence. We also implemented an **A contrario detection** (see https://hal-imt-atlantique.archives-ouvertes.fr/hal-01808176v4/document), however the minimum filter is already sufficient to give a good result.

Due to the page limit, we did not mention the peak_threshold and the minimum filter in the paper. 

- Now run the detection:
```
python eval.py \
       --mode=test \
       --model=vrnn \
       --dataset_path="./datasets/test_30_160.tfrecord" \
       --data_dimension=160 \
       --latent_size=160 \
       --batch_size=4 \
       --rerun_graph=True \
       --plot=True \
       --use_contrario=False \
       --anomaly_threshold=-322 \
       --peak_threshold=-582
```

## Acknowledgments
We would like to thank the Tensorflow Research Group for the implementation of VRNN and A3Lab for the Dataset.

## Contact
This codebase is maintained by Duong NGUYEN (van.nguyen1@imt-atlantique.fr). For questions and issues please open an issue on the issues tracker.
