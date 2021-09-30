confignet = {
        'net_id': 'test_pc-ae',                # ID of the network (the same name is utilized for the network directory)
        'dataset': ['<example_directory_0>',], # List of the directories that contain the geometric data
        'probsamp': [None,],                   # List of the directories that contain the files (*.dat) with the point sampling probability
        'shapelist': [500,],                   # List with the number/names of the shapes to be sampled
        'out_data':  None,                     # Output directory to save the network files. A folder <net_id> will be created in <out_data>
        'training_batch_size': 50,             # Batch size utilized for training the model
        'test_batch_size': 50,                 # Batch size utilized for testing the model
        'pc_size': 6146,                       # Point cloud size
        'latent_layer': 128,                   # Size of the latent layer
        'encoder_layers': [64, 128, 128, 256], # List containing the number of feature per convolutional layer, **apart from the last layer**
        'decoder_layers': [256, 256],          # List containing the number of feature per fully connected layer, **apart from the last layer**
        'l_rate': 5e-4,                        # Learning rate for the AdamOptimizer algorithm
        'epochs_max': 700,                     # Maximum number of epochs
        'stop_training': 1e-06,                # Convergence criteria for the mean loss value
        'frac_training': 0.9,                  # Fraction of the data set utilized for training
        'autosave_rate': 10,                   # Interval (epochs) for saving the network files
        'alpha1': 1e3,                         # Scalar multiplier applied to the shape reconstruction loss (PC-VAE)
        'alpha2': 1e-3,                        # Scalar multiplier applied to the shape Kullback-Leibler Divergence (PC-VAE)
        'dpout': 1.0                           # Dropout ratio utilized for training the PC-VAE
        }