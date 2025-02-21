# Itihāsa Translation using ByT5 Model

This project uses the ByT5 model, fine-tuned for Sanskrit to Eng translation. ByT5 is a pre-trained byte-to-byte model that is used for sequence-to-sequence tasks, and in this project, it is fine tuned and trained to translate Sanskrit text into English.

## Project Structure
- `train.py`: Contains the code to train the ByT5 model. It loads the training, validation, and test datasets, preprocesses the data, and sets up the training arguments using `Seq2SeqTrainer`.
- `dataset/`: containing 93,000 Sanskrit shlokas and their English translations that are segregated into train, development, and test sets.
- `models/`: Directory to save the trained model.

You can install the required dependencies using the following:
```bash
pip install transformers datasets torch tensorboard
```

## Dataset
The dataset used for training the model is the [Itihāsa dataset]([https://huggingface.co/datasets/rahular/itihasa]).
Itihāsa is a Sanskrit-English translation corpus containing 93,000 Sanskrit shlokas and their English translations extracted from M. N. Dutt's seminal works on The Rāmāyana and The Mahābhārata.

## Training the Model
The `train.py` file will load the dataset from the specified disk locations, preprocess the data, and then fine-tune the ByT5 model on the provided training data.

### To start training:
Run the `train.py` script:
```bash
python train.py
```

### Results & Logs:
During training, logs will be saved for monitoring, and the best model will be saved based on the BLEU score. TensorBoard can be used to visualize the training process.
To view training logs with TensorBoard, run:

``` bash
tensorboard --logdir=runs
```
