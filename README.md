# DoS-Attack-Packet-Data-Generator

## INTRODUCTION
A Conditional-GRU model was used to generate unique packet data of about 500k samples. The model was trained on about 112,000 structured datapoints with 32 columns (features).
A loss of 0.0778 was achieved!

## INPUT
It takes a batch of start tag ('<p>') as input

## OUTPUT
It recurrently predicts the next character in the sequence for every feature.

# METHOD
The dataset was cleaned and empty cells inputed. Then, it was passed to the custom model for training. The features/columns of the dataset are then encoded using one hot encoding and fed
to a dense layer which outputs a vector representation of this columns. This representation is then passed to a GRU layer that make predictions at the character level.
This way, the model models the underlying probability distribution for each of the features.


For more info on the code, model architecture and explanations, read the .ipynb file attached to this repository. You can find a copy of the ouput generation in a csv file in the repo.
