## 3D Convolutional Neural Networks Utilizing Molecular Topological Features for Accurate Atomization Energy Predictions

### Requirements
- python 3.8
- numpy
- [natsort](https://pypi.org/project/natsort/)
- [torch-summary](https://pypi.org/project/torch-summary/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [PyTorch Lightning (version 1.2.0rc0)](https://pypi.org/project/pytorch-lightning/1.2.0rc0/)
- [Multiwfn](http://sobereva.com/multiwfn/download.html)
- Gaussian 16

*Note: We recommend using Anaconda for easy installation of libraries.*

### Data
The QM9-G4MP2 dataset is publicly available through [Materials Data Facility](https://petreldata.net/mdf/detail/wardlogan_machine_learning_calculations_v1.1/). The geometries (*xyz* files) and energies of the molecules were extracted from the dataset, which were then processed using the scripts in this repository. 
### Data Preparation
1. Extract *xyz* files of the molecules from the QM9-G4MP2 dataset. Create two directories to store training and test set data. Move the training and test set coordinate files to their respective directories. Now, carry out the following steps inside the training and test set directories.
2. Generate a unique molecular orientation and prepare a corresponding Gaussian input file.\
`python gen_unique_xyz.py --wfx_path "/wfx_file_directory_path/"`\
The user defined string argument represents the path of the directory, where the *wfx* files would be stored. For the sake of convenience, keep the wfx directory path the same as in step-1 for all the training/test set molecules. 
3. Run the input files generated in step-2 using Gaussian to obtain the *wfx* files.
4. Generate requisite volumetric properties from *wfx* files using Multiwfn.\
`./calc_3dprop.sh`
5. Prepare training and test datasets using the property files generated in step-4.\
`python make_inp.py --data_split "train"`\
`python make_inp.py --data_split "test"`
6. Place the training and test datasets (*pickle* files) generated in step-5 in a directory of your choice. Also, place the target labels (*g4mp2_b3lyp_diff_labels.pickle*) provided in the *data* folder in the same directory.

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

*Note: We recommend using GPUs for model training.*

### References
1. DenseNet model adapted from https://github.com/gpleiss/efficient_densenet_pytorch
