$ node --version
v10.15.3
$ npm ls @tensorflow/tfjs
tfjs-examples-translation@0.1.0 /translation
+-- @tensorflow/tfjs@0.15.3 
+-- @tensorflow/tfjs-node@0.3.1
| `-- @tensorflow/tfjs@0.15.3  deduped
`-- @tensorflow/tfjs-node-gpu@0.3.1
  `-- @tensorflow/tfjs@0.15.3  deduped

$ yarn train dist/fra.txt 

> tfjs-examples-translation@0.1.0 train
> node -r ts-node/register translation.ts "dist/fra.txt"

2019-03-07 15:50:35.736465: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Number of samples: 10000
Number of unique input tokens: 69
Number of unique output tokens: 93
Max sequence length for inputs: 16
Max sequence length for outputs: 59
Saved metadata at:  /tmp/translation.keras/metadata.json
Orthogonal initializer is being called on a matrix with more than 2000 (262144) elements: Slowness may result.
Orthogonal initializer is being called on a matrix with more than 2000 (262144) elements: Slowness may result.
Epoch 1 / 20
eta=0.0 =============================================================================================================================> 
65620ms 8203us/step - loss=1.11 val_loss=1.10 
Epoch 2 / 20
eta=0.0 =============================================================================================================================> 
63968ms 7996us/step - loss=0.937 val_loss=1.03 
Epoch 3 / 20
eta=0.0 =============================================================================================================================> 
64948ms 8119us/step - loss=0.853 val_loss=0.918 
Epoch 4 / 20
eta=0.0 =============================================================================================================================> 
65234ms 8154us/step - loss=0.766 val_loss=0.850 
Epoch 5 / 20
eta=0.0 =============================================================================================================================> 
65472ms 8184us/step - loss=0.710 val_loss=0.802 
Epoch 6 / 20
eta=0.0 =============================================================================================================================> 
67866ms 8483us/step - loss=0.673 val_loss=0.777 
Epoch 7 / 20
eta=0.0 =============================================================================================================================> 
67232ms 8404us/step - loss=0.647 val_loss=0.743 
Epoch 8 / 20
eta=0.0 =============================================================================================================================> 
67345ms 8418us/step - loss=0.626 val_loss=0.721 
Epoch 9 / 20
eta=0.0 =============================================================================================================================> 
66351ms 8294us/step - loss=0.607 val_loss=0.706 
Epoch 10 / 20
eta=0.0 =============================================================================================================================> 
67364ms 8421us/step - loss=0.591 val_loss=0.695 
Epoch 11 / 20
eta=0.0 =============================================================================================================================> 
65983ms 8248us/step - loss=0.577 val_loss=0.673 
Epoch 12 / 20
eta=0.0 =============================================================================================================================> 
67150ms 8394us/step - loss=0.565 val_loss=0.663 
Epoch 13 / 20
eta=0.0 =============================================================================================================================> 
66238ms 8280us/step - loss=0.553 val_loss=0.648 
Epoch 14 / 20
eta=0.0 =============================================================================================================================> 
67594ms 8449us/step - loss=0.543 val_loss=0.639 
Epoch 15 / 20
eta=0.0 =============================================================================================================================> 
67431ms 8429us/step - loss=0.534 val_loss=0.626 
Epoch 16 / 20
eta=0.0 =============================================================================================================================> 
66997ms 8375us/step - loss=0.525 val_loss=0.619 
Epoch 17 / 20
eta=0.0 =============================================================================================================================> 
64901ms 8113us/step - loss=0.513 val_loss=0.610 
Epoch 18 / 20
eta=0.0 =============================================================================================================================> 
64770ms 8096us/step - loss=0.502 val_loss=0.606 
Epoch 19 / 20
eta=0.0 =============================================================================================================================> 
73507ms 9188us/step - loss=0.494 val_loss=0.593 
Epoch 20 / 20
eta=0.0 =============================================================================================================================> 
64811ms 8101us/step - loss=0.486 val_loss=0.587 
Layer decoderLstm was passed non-serializable keyword arguments: [object Object]. They will not be included in the serialized model (and thus will be missing at deserialization time).
-
Input sentence: Go.
Target sentence: Va !																								
Decoded sentence: Nous êtes soures.

-
Input sentence: Hi.
Target sentence: Salut !																							
Decoded sentence: Tom est coure.

-
Input sentence: Run!
Target sentence: Cours !																							
Decoded sentence: Nous êtes soure.
