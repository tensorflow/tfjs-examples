$ python translation.py ~/git/javascript-concist-chit-chat/data/fra.txt 
Using TensorFlow backend.
Number of samples: 10000
Number of unique input tokens: 69
Number of unique output tokens: 93
Max sequence length for inputs: 16
Max sequence length for outputs: 59
Saved metadata at: /tmp/translation.keras/metadata.json
Train on 8000 samples, validate on 2000 samples
Epoch 1/20
2019-03-07 23:40:53.879374: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
8000/8000 [==============================] - 24s 3ms/step - loss: 0.9242 - val_loss: 0.9666
Epoch 2/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.7361 - val_loss: 0.7743
Epoch 3/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.6229 - val_loss: 0.6941
Epoch 4/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.5664 - val_loss: 0.6479
Epoch 5/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.5270 - val_loss: 0.6120
Epoch 6/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.4960 - val_loss: 0.5716
Epoch 7/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.4691 - val_loss: 0.5653
Epoch 8/20
8000/8000 [==============================] - 23s 3ms/step - loss: 0.4458 - val_loss: 0.5349
Epoch 9/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.4261 - val_loss: 0.5236
Epoch 10/20
8000/8000 [==============================] - 23s 3ms/step - loss: 0.4085 - val_loss: 0.5067
Epoch 11/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.3930 - val_loss: 0.4963
Epoch 12/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.3783 - val_loss: 0.4877
Epoch 13/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.3646 - val_loss: 0.4846
Epoch 14/20
8000/8000 [==============================] - 23s 3ms/step - loss: 0.3517 - val_loss: 0.4780
Epoch 15/20
8000/8000 [==============================] - 23s 3ms/step - loss: 0.3396 - val_loss: 0.4721
Epoch 16/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.3278 - val_loss: 0.4631
Epoch 17/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.3168 - val_loss: 0.4683
Epoch 18/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.3061 - val_loss: 0.4607
Epoch 19/20
8000/8000 [==============================] - 24s 3ms/step - loss: 0.2961 - val_loss: 0.4574
Epoch 20/20
8000/8000 [==============================] - 23s 3ms/step - loss: 0.2863 - val_loss: 0.4564
WARNING:tensorflow:Layer lstm_1 was passed non-serializable keyword arguments: {'initial_state': [<tf.Tensor 'lstm/while/Exit_2:0' shape=(?, 256) dtype=float32>, <tf.Tensor 'lstm/while/Exit_3:0' shape=(?, 256) dtype=float32>]}. They will not be included in the serialized model (and thus will be missing at deserialization time).
-
Input sentence: Go.
Target sentence: Va !
Decoded sentence: Continuez à nouveau.

-
Input sentence: Hi.
Target sentence: Salut !
Decoded sentence: Restez aven !

-
Input sentence: Run!
Target sentence: Cours !
Decoded sentence: Sais-tou ?
