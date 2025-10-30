# Self-supervised Multi-level Trajectory Representation Model for Field-Road Trajectory Segmentation

The source code used for our paper "Self-supervised Multi-level Trajectory Representation Model for Field-Road Trajectory Segmentation"

Requirements at least one GPU is required to run the code.
Before running, you need to first install the required packages by typing following commands (Using a virtual environment is recommended):

```shell
pip3 install -r requirements.txt
```

Running the field road trajectory segmentation model:

1. Agricultural machinery trajectory data cleaning:

   ```shell
   python data_cleaning.py
   ```

2. Calculate 25-dimensional features for each trajectory point:

   ```shell
   python cal_25dim_feature.py
   ```

3. Trajectory similarity comparison:

   ```shell
   python DTW_5W_multiprocess.py
   ```

4. Pre-training TR model:

   ```shell
   python Pre-training-TR.py
   ```

5. Finetune TS model:

   ```shell
   python Finetune-TS.py
   ```

Citation

Thanks for dataset from https://github.com/Agribigdata/Field_road_dataset
