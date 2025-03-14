""" Below are the sections of code that were modified to change the sampling rate and kernel width """

# Path: config\transforms\log_spectrogram.yaml
logspec:
  _target_: emg2qwerty.transforms.LogSpectrogram
  n_fft: 64
  hop_length: 16  # This line was modified

# Path: emg2qwerty\transforms.py
class LogSpectrogram:
    # ...
    n_fft: int = 64
    hop_length: int = 16 # This line was modified

# Path: config\model\tds_conv_ctc.yaml
module:
  _target_: emg2qwerty.lightning.TDSConvCTCModule
  in_features: 528 
  mlp_features: [384]
  block_channels: [24, 24, 24, 24]
  kernel_width: 32  # This line was modified


""" Below are the 5 different models I tested, in the order as seen on the graph sampling_rates_and_kernel_widths.png """

# Model 1: hop_length=40 (50Hz), kernel_width=32
    # Validation CER = 83.27426147460938
    # Validation loss = 2.4778265953063965
    # Test CER = 83.81240844726562
    # Test loss = 2.361758232116699

# Model 2: hop_length=40 (50Hz), kernel_width=14
    # Validation CER = 20.602569580078125
    # Validation loss = 0.6915518641471863
    # Test CER = 19.53749656677246
    # Test loss = 0.6235761046409607

# Model 3: hop_length=16 (125Hz), kernel_width=32 (base model)
    # Validation CER = 20.425342559814453
    # Validation loss = 0.6693854331970215
    # Test CER = 21.56905174255371
    # Test loss = 0.6914742588996887

# Model 4: hop_length=8 (250Hz), kernel_width=32
    # Validation CER = 32.18874740600586
    # Validation loss = 1.0388929843902588
    # Test CER = 35.35768127441406
    # Test loss = 1.2299225330352783

# Model 5: hop_length=8 (250Hz), kernel_width=64
    # Validation CER = 23.836952209472656
    # Validation loss = 0.7680050134658813
    # Test CER = 24.832504272460938
    # Test loss = 0.767642080783844