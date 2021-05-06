## 3D Dense Convolutional Neural Networks Utilizing Molecular Topological Features for Accurate Atomization Energy Predictions

### Requirements
- python 3.8
- numpy
- [PyTorch Lightning (>= 1.2.0rc0)](https://www.pytorchlightning.ai/)
- [Multiwfn](http://sobereva.com/multiwfn/download.html)
- Gaussian 16

### Data
The QM9-G4MP2 dataset is publicly available through [Materials Data Facility](https://petreldata.net/mdf/detail/wardlogan_machine_learning_calculations_v1.1/). The geometries (xyz files) and energies of the molecules were extracted from the dataset, which were then processed using the scripts in this repository. 
### Data Preparation
1. Generate a unique molecular orientation and prepare a corresponding Gaussian input file.\
`python gen_unique_xyz.py "/wfx_file_directory_path/"`\
The user defined string argument represents the path of the directory, where the *wfx* files would be stored.
2. Run the input files generated in step 1 using Gaussian to obtain the *wfx* files.
3. Generate requisite volumetric properties from *wfx* files using Multiwfn.\
`./calc_3dprop.sh`
4. Prepare training and test datasets.\
`python make_inp.py "train"`\
`python make_inp.py "test"`\
5. Place the training and test datasets generated in step-4 in a directory of your choice. Also, place the target labels (g4mp2_b3lyp_diff_labels.pickle) provided in the data folder in the same directory.

*Note: For the sake of convenience, all the processing steps need to be carried out separately for training and test set molecules.*

### Model Training
Train the model using the following command.\
`python train_pl.py --data_path "/datasets_directory_path/" --channel 2 --grid_length 14 --batch_size 32 --epochs 250 --dense1 16 --dense2 16 > results.txt 2> errors.txt`\
\
Arguments:\
`data_path`: path of the directory where input and output data are stored\
`channel`: one of `0`, `1`, `2`, or `3`\
           `0`: Nuclear Electrostatic potential (NEP)\
           `1`: Electron Localization Function (ELF)\
           `2`: Localized Orbital Locator (LOL)\
           `3`: Electrostatic Potential (ESP)\
`grid_length`: Voxel grid length of the cubic volume\
`dense1`: Depth of the first dense block\
`dense2`: Depth of the second dense block

### References
