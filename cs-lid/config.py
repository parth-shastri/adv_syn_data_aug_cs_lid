CONFIG = {"train_data_dir": "data/Language Identification/spectrograms new/train",
        #   "validation_data_dir": "<path to dataset dir>/validation.csv",
          "test_data_dir": "data/Language Identification/spectrograms/test",
          "train_csv": "train_an_augmented.csv",
          "test_csv": "test.csv",

          "batch_size": 32,
          "learning_rate": 0.001,
          "num_epochs": 30  ,

          "data_loader": "ImageLoader",
          "color_mode": "L",  # L = bw or RGB,
          "input_shape": [128, 128, 1],

          "model": "topcoder_crnn_finetune",  # _finetune"

          "segment_length": 4,  # number of seconds each spectogram represents,
          "pixel_per_second": 50,

          "label_names": ["English", "Hindi", "Hindi-English"],
          "num_classes": 3,

          # label_names: ["EN", "DE", "FR", "ES", "CN", "RUS"]
          # num_classes: 6

          }