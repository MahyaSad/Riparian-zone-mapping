class Config:
    # Data paths
    TRAIN_PATH = '/path/to/train/data'
    VAL_PATH = '/path/to/val/data'
    TEST_PATH = '/path/to/test/data'
    
    # Tile IDs
    TRAIN_TILES = ['tile1', 'tile2', 'tile3']
    VAL_TILES = ['tile4', 'tile5']
    TEST_TILES = ['tile6', 'tile7']
    
    # Variables
    S1_VARIABLES = ['VH', 'VV', 'incidence_angle', 'DEM']
    S2_VARIABLES = ['B02', 'B03', 'B04', 'B08', 'B11', 
                    'NDVI', 'NDWI_veg', 'NDWI_water', 'DEM']
    
    # Model parameters
    NUM_CLASSES = 2
    BASE_FILTERS = 64
    DEPTH = 4
    
    # Training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    EPOCHS = 100
    PATIENCE = 10
    
    # Dataset parameters
    PATCH_SIZE = 64
    SAMPLES_PER_TILE = 100
    NO_DATA_VALUE = -9999
    
    # Hardware parameters
    NUM_WORKERS = 4
    
    # Hyperparameter search space
    SEARCH_SPACE = {
        'patch_size': [32, 64, 128, 256],
        'batch_size': [4, 8, 16, 32],
        'learning_rate': (1e-5, 1e-2),
        'base_filters': (32, 128),
        'depth': (3, 5),
        'fusion_type': ['concat', 'attention']
    }