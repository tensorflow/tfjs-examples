# node -r ts-node/register translation.ts --epochs 100 --num_samples 10000 --num_test_sentences 30 --batch_size 64 dist/fra.txt  
2019-03-09 17:30:26.155671: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-03-09 17:30:26.265046: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-03-09 17:30:26.265451: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:01:00.0
totalMemory: 7.93GiB freeMemory: 7.19GiB
2019-03-09 17:30:26.265467: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-03-09 17:30:26.454137: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-09 17:30:26.454169: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-03-09 17:30:26.454189: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-03-09 17:30:26.454323: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6927 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
Number of samples: 10000
Number of unique input tokens: 69
Number of unique output tokens: 93
Max sequence length for inputs: 16
Max sequence length for outputs: 59
Saved metadata at:  /tmp/translation.keras/metadata.json
Orthogonal initializer is being called on a matrix with more than 2000 (262144) elements: Slowness may result.
Orthogonal initializer is being called on a matrix with more than 2000 (262144) elements: Slowness may result.
Epoch 1 / 100
eta=0.0 ==============================================================> 
47576ms 5947us/step - loss=1.10 val_loss=1.11 
Epoch 2 / 100
eta=0.0 ==============================================================> 
47072ms 5884us/step - loss=0.943 val_loss=1.05 
Epoch 3 / 100
eta=0.0 ==============================================================> 
47378ms 5922us/step - loss=0.867 val_loss=0.936 
Epoch 4 / 100
eta=0.0 ==============================================================> 
47330ms 5916us/step - loss=0.777 val_loss=0.851 
Epoch 5 / 100
eta=0.0 ==============================================================> 
47418ms 5927us/step - loss=0.718 val_loss=0.805 
Epoch 6 / 100
eta=0.0 ==============================================================> 
47580ms 5948us/step - loss=0.680 val_loss=0.777 
Epoch 7 / 100
eta=0.0 ==============================================================> 
47541ms 5943us/step - loss=0.652 val_loss=0.746 
Epoch 8 / 100
eta=0.0 ==============================================================> 
47870ms 5984us/step - loss=0.630 val_loss=0.732 
Epoch 9 / 100
eta=0.0 ==============================================================> 
48659ms 6082us/step - loss=0.611 val_loss=0.706 
Epoch 10 / 100
eta=0.0 ==============================================================> 
48895ms 6112us/step - loss=0.593 val_loss=0.701 
Epoch 11 / 100
eta=0.0 ==============================================================> 
49345ms 6168us/step - loss=0.578 val_loss=0.679 
Epoch 12 / 100
eta=0.0 ==============================================================> 
49397ms 6175us/step - loss=0.565 val_loss=0.662 
Epoch 13 / 100
eta=0.0 ==============================================================> 
49217ms 6152us/step - loss=0.554 val_loss=0.649 
Epoch 14 / 100
eta=0.0 ==============================================================> 
49159ms 6145us/step - loss=0.543 val_loss=0.646 
Epoch 15 / 100
eta=0.0 ==============================================================> 
48713ms 6089us/step - loss=0.531 val_loss=0.627 
Epoch 16 / 100
eta=0.0 ==============================================================> 
48556ms 6070us/step - loss=0.519 val_loss=0.617 
Epoch 17 / 100
eta=0.0 ==============================================================> 
48621ms 6078us/step - loss=0.509 val_loss=0.609 
Epoch 18 / 100
eta=0.0 ==============================================================> 
48590ms 6074us/step - loss=0.501 val_loss=0.604 
Epoch 19 / 100
eta=0.0 ==============================================================> 
48529ms 6066us/step - loss=0.493 val_loss=0.593 
Epoch 20 / 100
eta=0.0 ==============================================================> 
49103ms 6138us/step - loss=0.486 val_loss=0.586 
Epoch 21 / 100
eta=0.0 ==============================================================> 
49117ms 6140us/step - loss=0.479 val_loss=0.584 
Epoch 22 / 100
eta=0.0 ==============================================================> 
49950ms 6244us/step - loss=0.472 val_loss=0.571 
Epoch 23 / 100
eta=0.0 ==============================================================> 
48556ms 6070us/step - loss=0.466 val_loss=0.567 
Epoch 24 / 100
eta=0.0 ==============================================================> 
48505ms 6063us/step - loss=0.460 val_loss=0.562 
Epoch 25 / 100
eta=0.0 ==============================================================> 
59577ms 7447us/step - loss=0.454 val_loss=0.558 
Epoch 26 / 100
eta=0.0 ==============================================================> 
49145ms 6143us/step - loss=0.449 val_loss=0.559 
Epoch 27 / 100
eta=0.0 ==============================================================> 
48726ms 6091us/step - loss=0.443 val_loss=0.547 
Epoch 28 / 100
eta=0.0 ==============================================================> 
48730ms 6091us/step - loss=0.438 val_loss=0.544 
Epoch 29 / 100
eta=0.0 ==============================================================> 
49623ms 6203us/step - loss=0.434 val_loss=0.537 
Epoch 30 / 100
eta=0.0 ==============================================================> 
50317ms 6290us/step - loss=0.429 val_loss=0.537 
Epoch 31 / 100
eta=0.0 ==============================================================> 
49745ms 6218us/step - loss=0.424 val_loss=0.531 
Epoch 32 / 100
eta=0.0 ==============================================================> 
49007ms 6126us/step - loss=0.419 val_loss=0.529 
Epoch 33 / 100
eta=0.0 ==============================================================> 
49762ms 6220us/step - loss=0.415 val_loss=0.524 
Epoch 34 / 100
eta=0.0 ==============================================================> 
52068ms 6509us/step - loss=0.411 val_loss=0.520 
Epoch 35 / 100
eta=0.0 ==============================================================> 
53217ms 6652us/step - loss=0.406 val_loss=0.527 
Epoch 36 / 100
eta=0.0 ==============================================================> 
53240ms 6655us/step - loss=0.402 val_loss=0.515 
Epoch 37 / 100
eta=0.0 ==============================================================> 
51462ms 6433us/step - loss=0.398 val_loss=0.515 
Epoch 38 / 100
eta=0.0 ==============================================================> 
51860ms 6483us/step - loss=0.393 val_loss=0.509 
Epoch 39 / 100
eta=0.0 ==============================================================> 
51200ms 6400us/step - loss=0.389 val_loss=0.507 
Epoch 40 / 100
eta=0.0 ==============================================================> 
48849ms 6106us/step - loss=0.384 val_loss=0.504 
Epoch 41 / 100
eta=0.0 ==============================================================> 
49511ms 6189us/step - loss=0.380 val_loss=0.500 
Epoch 42 / 100
eta=0.0 ==============================================================> 
49569ms 6196us/step - loss=0.376 val_loss=0.494 
Epoch 43 / 100
eta=0.0 ==============================================================> 
49377ms 6172us/step - loss=0.371 val_loss=0.492 
Epoch 44 / 100
eta=0.0 ==============================================================> 
49998ms 6250us/step - loss=0.368 val_loss=0.489 
Epoch 45 / 100
eta=0.0 ==============================================================> 
49633ms 6204us/step - loss=0.363 val_loss=0.493 
Epoch 46 / 100
eta=0.0 ==============================================================> 
50019ms 6252us/step - loss=0.359 val_loss=0.486 
Epoch 47 / 100
eta=0.0 ==============================================================> 
48356ms 6045us/step - loss=0.356 val_loss=0.492 
Epoch 48 / 100
eta=0.0 ==============================================================> 
49584ms 6198us/step - loss=0.352 val_loss=0.485 
Epoch 49 / 100
eta=0.0 ==============================================================> 
49620ms 6203us/step - loss=0.348 val_loss=0.486 
Epoch 50 / 100
eta=0.0 ==============================================================> 
48718ms 6090us/step - loss=0.345 val_loss=0.483 
Epoch 51 / 100
eta=0.0 ==============================================================> 
56570ms 7071us/step - loss=0.341 val_loss=0.485 
Epoch 52 / 100
eta=0.0 ==============================================================> 
49167ms 6146us/step - loss=0.338 val_loss=0.482 
Epoch 53 / 100
eta=0.0 ==============================================================> 
48749ms 6094us/step - loss=0.334 val_loss=0.483 
Epoch 54 / 100
eta=0.0 ==============================================================> 
48929ms 6116us/step - loss=0.330 val_loss=0.479 
Epoch 55 / 100
eta=0.0 ==============================================================> 
48822ms 6103us/step - loss=0.327 val_loss=0.473 
Epoch 56 / 100
eta=0.0 ==============================================================> 
48925ms 6116us/step - loss=0.324 val_loss=0.475 
Epoch 57 / 100
eta=0.0 ==============================================================> 
48378ms 6047us/step - loss=0.320 val_loss=0.479 
Epoch 58 / 100
eta=0.0 ==============================================================> 
48598ms 6075us/step - loss=0.317 val_loss=0.475 
Epoch 59 / 100
eta=0.0 ==============================================================> 
48589ms 6074us/step - loss=0.314 val_loss=0.475 
Epoch 60 / 100
eta=0.0 ==============================================================> 
48969ms 6121us/step - loss=0.311 val_loss=0.473 
Epoch 61 / 100
eta=0.0 ==============================================================> 
49302ms 6163us/step - loss=0.307 val_loss=0.474 
Epoch 62 / 100
eta=0.0 ==============================================================> 
48541ms 6068us/step - loss=0.304 val_loss=0.479 
Epoch 63 / 100
eta=0.0 ==============================================================> 
48915ms 6114us/step - loss=0.301 val_loss=0.474 
Epoch 64 / 100
eta=0.0 ==============================================================> 
48746ms 6093us/step - loss=0.298 val_loss=0.474 
Epoch 65 / 100
eta=0.0 ==============================================================> 
49186ms 6148us/step - loss=0.294 val_loss=0.477 
Epoch 66 / 100
eta=0.0 ==============================================================> 
49448ms 6181us/step - loss=0.291 val_loss=0.471 
Epoch 67 / 100
eta=0.0 ==============================================================> 
48877ms 6110us/step - loss=0.288 val_loss=0.477 
Epoch 68 / 100
eta=0.0 ==============================================================> 
49378ms 6172us/step - loss=0.285 val_loss=0.475 
Epoch 69 / 100
eta=0.0 ==============================================================> 
49014ms 6127us/step - loss=0.282 val_loss=0.471 
Epoch 70 / 100
eta=0.0 ==============================================================> 
48808ms 6101us/step - loss=0.279 val_loss=0.477 
Epoch 71 / 100
eta=0.0 ==============================================================> 
49325ms 6166us/step - loss=0.276 val_loss=0.477 
Epoch 72 / 100
eta=0.0 ==============================================================> 
48762ms 6095us/step - loss=0.273 val_loss=0.479 
Epoch 73 / 100
eta=0.0 ==============================================================> 
49796ms 6225us/step - loss=0.270 val_loss=0.473 
Epoch 74 / 100
eta=0.0 ==============================================================> 
49788ms 6223us/step - loss=0.267 val_loss=0.480 
Epoch 75 / 100
eta=0.0 ==============================================================> 
49301ms 6163us/step - loss=0.265 val_loss=0.483 
Epoch 76 / 100
eta=0.0 ==============================================================> 
48026ms 6003us/step - loss=0.261 val_loss=0.472 
Epoch 77 / 100
eta=0.0 ==============================================================> 
47988ms 5998us/step - loss=0.259 val_loss=0.479 
Epoch 78 / 100
eta=0.0 ==============================================================> 
48119ms 6015us/step - loss=0.256 val_loss=0.479 
Epoch 79 / 100
eta=0.0 ==============================================================> 
48105ms 6013us/step - loss=0.254 val_loss=0.479 
Epoch 80 / 100
eta=0.0 ==============================================================> 
48125ms 6016us/step - loss=0.251 val_loss=0.479 
Epoch 81 / 100
eta=0.0 ==============================================================> 
48042ms 6005us/step - loss=0.248 val_loss=0.521 
Epoch 82 / 100
eta=0.0 ==============================================================> 
48111ms 6014us/step - loss=0.246 val_loss=0.484 
Epoch 83 / 100
eta=0.0 ==============================================================> 
48190ms 6024us/step - loss=0.243 val_loss=0.485 
Epoch 84 / 100
eta=0.0 ==============================================================> 
48154ms 6019us/step - loss=0.241 val_loss=0.487 
Epoch 85 / 100
eta=0.0 ==============================================================> 
48214ms 6027us/step - loss=0.238 val_loss=0.484 
Epoch 86 / 100
eta=0.0 ==============================================================> 
48015ms 6002us/step - loss=0.236 val_loss=0.490 
Epoch 87 / 100
eta=0.0 ==============================================================> 
48053ms 6007us/step - loss=0.234 val_loss=0.487 
Epoch 88 / 100
eta=0.0 ==============================================================> 
48214ms 6027us/step - loss=0.231 val_loss=0.493 
Epoch 89 / 100
eta=0.0 ==============================================================> 
48086ms 6011us/step - loss=0.229 val_loss=0.492 
Epoch 90 / 100
eta=0.0 ==============================================================> 
48004ms 6000us/step - loss=0.226 val_loss=0.493 
Epoch 91 / 100
eta=0.0 ==============================================================> 
48079ms 6010us/step - loss=0.224 val_loss=0.493 
Epoch 92 / 100
eta=0.0 ==============================================================> 
47936ms 5992us/step - loss=0.222 val_loss=0.499 
Epoch 93 / 100
eta=0.0 ==============================================================> 
47797ms 5975us/step - loss=0.220 val_loss=0.498 
Epoch 94 / 100
eta=0.0 ==============================================================> 
48095ms 6012us/step - loss=0.218 val_loss=0.500 
Epoch 95 / 100
eta=0.0 ==============================================================> 
48077ms 6010us/step - loss=0.216 val_loss=0.498 
Epoch 96 / 100
eta=0.0 ==============================================================> 
48056ms 6007us/step - loss=0.213 val_loss=0.504 
Epoch 97 / 100
eta=0.0 ==============================================================> 
48085ms 6011us/step - loss=0.211 val_loss=0.510 
Epoch 98 / 100
eta=0.0 ==============================================================> 
48000ms 6000us/step - loss=0.209 val_loss=0.506 
Epoch 99 / 100
eta=0.0 ==============================================================> 
47799ms 5975us/step - loss=0.208 val_loss=0.510 
Epoch 100 / 100
eta=0.0 ==============================================================> 
47819ms 5977us/step - loss=0.205 val_loss=0.509 
Layer decoderLstm was passed non-serializable keyword arguments: [object Object]. They will not be included in the serialized model (and thus will be missing at deserialization time).
-
Input sentence: Go.
Target sentence: Va !
Decoded sentence: Attrapez ça !

-
Input sentence: Hi.
Target sentence: Salut !
Decoded sentence: C'est la mienne.

-
Input sentence: Run!
Target sentence: Cours !
Decoded sentence: Prends plaisir.

-
Input sentence: Run!
Target sentence: Courez !
Decoded sentence: Prends plaisir.

-
Input sentence: Wow!
Target sentence: Ça alors !
Decoded sentence: Quel coup tort !

-
Input sentence: Fire!
Target sentence: Au feu !
Decoded sentence: Prends plaisant au bonn !

-
Input sentence: Help!
Target sentence: À l'aide !
Decoded sentence: Personne ne sait.

-
Input sentence: Jump.
Target sentence: Saute.
Decoded sentence: Vous êtes maline.

-
Input sentence: Stop!
Target sentence: Ça suffit !
Decoded sentence: Parle avec moi.

-
Input sentence: Stop!
Target sentence: Stop !
Decoded sentence: Parle avec moi.

-
Input sentence: Stop!
Target sentence: Arrête-toi !
Decoded sentence: Parle avec moi.

-
Input sentence: Wait!
Target sentence: Attends !
Decoded sentence: Quel coup tort !

-
Input sentence: Wait!
Target sentence: Attendez !
Decoded sentence: Quel coup tort !

-
Input sentence: Go on.
Target sentence: Poursuis.
Decoded sentence: Prends tout !

-
Input sentence: Go on.
Target sentence: Continuez.
Decoded sentence: Prends tout !

-
Input sentence: Go on.
Target sentence: Poursuivez.
Decoded sentence: Prends tout !

-
Input sentence: Hello!
Target sentence: Bonjour !
Decoded sentence: Personne ne sombie.

-
Input sentence: Hello!
Target sentence: Salut !
Decoded sentence: Personne ne sombie.

-
Input sentence: I see.
Target sentence: Je comprends.
Decoded sentence: Je me sens prêt.

-
Input sentence: I try.
Target sentence: J'essaye.
Decoded sentence: Je te pronsse pas.

-
Input sentence: I won!
Target sentence: J'ai gagné !
Decoded sentence: Je l'ai appelé.

-
Input sentence: I won!
Target sentence: Je l'ai emporté !
Decoded sentence: Je l'ai appelé.

-
Input sentence: Oh no!
Target sentence: Oh non !
Decoded sentence: Tom est applabé.

-
Input sentence: Attack!
Target sentence: Attaque !
Decoded sentence: Pardonnez-moi.

-
Input sentence: Attack!
Target sentence: Attaquez !
Decoded sentence: Pardonnez-moi.

-
Input sentence: Cheers!
Target sentence: Santé !
Decoded sentence: Tout le monde est rimion.

-
Input sentence: Cheers!
Target sentence: À votre santé !
Decoded sentence: Tout le monde est rimion.

-
Input sentence: Cheers!
Target sentence: Merci !
Decoded sentence: Tout le monde est rimion.

-
Input sentence: Cheers!
Target sentence: Tchin-tchin !
Decoded sentence: Tout le monde est rimion.

-
Input sentence: Get up.
Target sentence: Lève-toi.
Decoded sentence: Attrapez ça !

