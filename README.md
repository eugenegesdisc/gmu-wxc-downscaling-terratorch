# gmu-wxc-downscaling-terratorch
Programs and configurations to fine-tune Prithvi WxC for downscaling climate variables

## Set up

The project used conda (miniforge) environment to set up the environment.

The following are the steps to manually configure the environment under Ubuntu server.

$ conda create -n devterratorch -c conda-forge python=3.13
$ conda activate devterratorch
# install pytorch
$ conda install pytorch -c conda-forge
$ conda install -c conda-forge gdal lightning torchgeo libgdal-hdf5 libgdal-netcdf


Following the step to install Prithvi-WxC:
Installing Prithvi-WxC ( https://github.com/NASA-IMPACT/Prithvi-WxC )
$ cd ..
$ git clone https://github.com/NASA-IMPACT/Prithvi-WxC
$ cd Prithvi-WxC

Install the Prithvi-WxC

# Do the following to add g++, if needed:
$ sudo apt install g++

$ python -m pip install '.[examples]'


$ python -m pip install -U 'jsonargparse[signatures]'

$ conda install pytest -c conda-forge
$ conda install tensorboard -c conda-forge


## Configure Cridentials for downloading from Earthdata

See examples/config for an template. The default directory and configuration file is at the following:
../cfg/.gmu_downscaling_earthdata_login.cfg


## Run the programs

### Preparare the data - train, validate, test, or predict

The following examples show how to run programs from the cloned program directory:

Make necessary directorys for logs, lighntning logs.

The following example show to prepare training dateset using 10 day data:
$ time python -m gmudownscalingterratorch cmd prepdataset --input-times "2020-01-01; 2020-01-10 23:59:59" --root-dir "../exp" --name-of-sample merra10day --download --extract --convert

### Train (fit), test or predict
To train (activated the conda environment):

$ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$ nohup bash -c 'time python -m gmudownscalingterratorch fit --trainer.default_root_dir "../exs/merra2day/tmean/e10" --trainer.strategy=ddp_find_unused_parameters_true --trainer.log_every_n_steps=10 --trainer.max_epochs=10 -c "tests/configs/gmu_downscaling_cnn_pixelshuffle_2day.yaml"' > ../logs/merra2day/train_log_tmean_e10.txt 2>../logs/merra2day/train_log_tmean_e10.err &

This program uses LightningCLI with torchgeo as the base. So, you can override certain parameters to carry out different experiments.

To test with the latest version, example run can be as follows:

$ python -m gmudownscalingterratorch test -c "tests/configs/gmu_downscaling_cnn_pixelshuffle_2day.yaml"

To predict with the latest version, example run can be as follows:

$ python -m gmudownscalingterratorch predict -c "tests/configs/gmu_downscaling_cnn_pixelshuffle_2day.yaml"

Severeral other experimental configurations can be found under tests/configs.


