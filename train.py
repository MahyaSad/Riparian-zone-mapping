# train.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import optuna
from tqdm import tqdm
from model import DualStreamUNET3D
from dataset import EnhancedZarrTileDataset

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def train_single_stream(data_type, config, train_dataset, val_dataset):
    """Train a single encoder (S1 or S2) and save the weights"""
    world_size = torch.cuda.device_count()
    
    def train_worker(rank, world_size):
        setup_distributed(rank, world_size)
        
        # Create data samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=train_sampler,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            sampler=val_sampler,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        # Initialize model
        if data_type == 'sentinel1':
            in_channels = len(config['s1_variables'])
        else:
            in_channels = len(config['s2_variables'])
            
        model = Encoder3D(in_channels, config['base_filters'], config['depth'])
        temp_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(model.bottleneck_channels, config['num_classes'])
        )
        
        model = model.to(rank)
        temp_head = temp_head.to(rank)
        model = DDP(model, device_ids=[rank])
        temp_head = DDP(temp_head, device_ids=[rank])
        
        # Setup training
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = optim.AdamW([
            {'params': model.parameters()},
            {'params': temp_head.parameters()}
        ], lr=config['learning_rate'], weight_decay=config['weight_decay'])
        
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            # Training
            model.train()
            temp_head.train()
            train_loss = 0.0
            
            for data_batch, _, labels in train_loader:
                data_batch = data_batch.to(rank)
                labels = labels.to(rank)
                
                optimizer.zero_grad()
                features, _ = model(data_batch)
                outputs = temp_head(features)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                torch.nn.utils.clip_grad_norm_(temp_head.parameters(), config['grad_clip'])
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            temp_head.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for data_batch, _, labels in val_loader:
                    data_batch = data_batch.to(rank)
                    labels = labels.to(rank)
                    
                    features, _ = model(data_batch)
                    outputs = temp_head(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            scheduler.step()
            
            # Average losses across processes
            train_loss_tensor = torch.tensor(train_loss / len(train_loader)).to(rank)
            val_loss_tensor = torch.tensor(val_loss / len(val_loader)).to(rank)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            
            train_loss_avg = train_loss_tensor.item() / world_size
            val_loss_avg = val_loss_tensor.item() / world_size
            
            if rank == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss_avg:.4f}, Val Loss = {val_loss_avg:.4f}")
                
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    patience_counter = 0
                    # Save only encoder weights
                    torch.save({
                        'encoder_state_dict': model.module.state_dict(),
                        'config': config
                    }, f'{data_type}_encoder.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= config['patience']:
                        print(f'Early stopping at epoch {epoch}')
                        break
        
        cleanup_distributed()
    
    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

def train_distributed(rank, world_size, model, train_dataset, val_dataset, config):
    """Train the full dual-stream model"""
    setup_distributed(rank, world_size)
    
    # Create data samplers and loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Setup model
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        for s1_data, s2_data, labels in train_loader:
            s1_data = s1_data.to(rank)
            s2_data = s2_data.to(rank)
            labels = labels.to(rank)
            
            optimizer.zero_grad()
            outputs = model(s1_data, s2_data)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s1_data, s2_data, labels in val_loader:
                s1_data = s1_data.to(rank)
                s2_data = s2_data.to(rank)
                labels = labels.to(rank)
                
                outputs = model(s1_data, s2_data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        scheduler.step()
        
        # Average losses across processes
        train_loss_tensor = torch.tensor(train_loss / len(train_loader)).to(rank)
        val_loss_tensor = torch.tensor(val_loss / len(val_loader)).to(rank)
        dist.all_reduce(train_loss_tensor, op=dist.Reduc
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        train_loss_avg = train_loss_tensor.item() / world_size
        val_loss_avg = val_loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss_avg:.4f}, Val Loss = {val_loss_avg:.4f}")
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'config': config
                }, 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f'Early stopping at epoch {epoch}')
                    break
    
    cleanup_distributed()
    return best_val_loss

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    config = {
        # Model architecture params
        'num_layers': trial.suggest_int('num_layers', 2, 5),
        'depth': trial.suggest_int('depth', 3, 5),
        'base_filters': trial.suggest_int('base_filters', 32, 128),
        'fusion_type': trial.suggest_categorical('fusion_type', ['concat', 'attention']),
        'use_bn': trial.suggest_categorical('use_bn', [True, False]),
        
        # Patch and data params
        'patch_size': trial.suggest_categorical('patch_size', [32, 64, 128, 256]),
        'samples_per_tile': trial.suggest_int('samples_per_tile', 50, 200),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
        
        # Training params
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'min_lr': trial.suggest_loguniform('min_lr', 1e-7, 1e-4),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
        'grad_clip': trial.suggest_uniform('grad_clip', 0.1, 3.0),
        'epochs': 100,
        'patience': trial.suggest_int('patience', 5, 15),
        'num_workers': 4,
        
        # Variables definition
        's1_variables': ['VH', 'VV', 'incidence_angle', 'DEM'],
        's2_variables': ['B02', 'B03', 'B04', 'B08', 'B11', 'NDVI', 'NDWI_veg', 'NDWI_water', 'DEM'],
        'num_classes': 2
    }
    
    # Initialize datasets with current patch size
    train_dataset = EnhancedZarrTileDataset(
        base_path=config['train_path'],
        tile_ids=config['train_tiles'],
        patch_size=config['patch_size'],
        label_variable='label',
        stats_file='stats.json',
        calculate_stats=True,
        gpu_id=0,
        balanced_sampling=True,
        samples_per_tile=config['samples_per_tile']
    )
    
    val_dataset = EnhancedZarrTileDataset(
        base_path=config['val_path'],
        tile_ids=config['val_tiles'],
        patch_size=config['patch_size'],
        label_variable='label',
        stats_file='stats.json',
        calculate_stats=False,
        gpu_id=0,
        balanced_sampling=True,
        samples_per_tile=config['samples_per_tile']
    )
    
    # First train individual encoders
    print("Training Sentinel-1 encoder...")
    train_single_stream('sentinel1', config, train_dataset, val_dataset)
    
    print("Training Sentinel-2 encoder...")
    train_single_stream('sentinel2', config, train_dataset, val_dataset)
    
    # Initialize and train full model
    world_size = torch.cuda.device_count()
    model = DualStreamUNET3D(config)
    
    # Load pretrained encoder weights
    model.load_pretrained_encoders('sentinel1_encoder.pth', 'sentinel2_encoder.pth')
    
    # Train full model
    best_val_loss = mp.spawn(
        train_distributed,
        args=(world_size, model, train_dataset, val_dataset, config),
        nprocs=world_size,
        join=True
    )
    
    return best_val_loss

def main():
    """Main training pipeline"""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available for training")
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    # Base configuration
    base_config = {
        'train_path': '/path/to/train/data',
        'val_path': '/path/to/val/data',
        'test_path': '/path/to/test/data',
        'train_tiles': ['tile1', 'tile2', 'tile3'],
        'val_tiles': ['tile4', 'tile5'],
        'test_tiles': ['tile6', 'tile7'],
        'num_classes': 2,
        'num_workers': 4 * num_gpus
    }
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Initialize Optuna study
    study = optuna.create_study(
        direction='minimize',
        study_name='satellite_segmentation',
        storage='sqlite:///study.db',
        load_if_exists=True
    )
    
    # Optional callback to save study state
    def save_study_callback(study, trial):
        import joblib
        joblib.dump(study, 'study.pkl')
    
    try:
        # Optimize hyperparameters
        study.optimize(
            objective,
            n_trials=100,
            callbacks=[save_study_callback],
            gc_after_trial=True,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_value = study.best_value
        print("\nBest trial:")
        print(f"Value: {best_value}")
        print("Params:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")
        
        # Train final model with best hyperparameters
        final_config = {**base_config, **best_params}
        
        # Initialize datasets for final training
        train_dataset = EnhancedZarrTileDataset(
            base_path=final_config['train_path'],
            tile_ids=final_config['train_tiles'],
            patch_size=final_config['patch_size'],
            label_variable='label',
            stats_file='stats.json',
            calculate_stats=True,
            gpu_id=0,
            balanced_sampling=True,
            samples_per_tile=final_config['samples_per_tile']
        )
        
        val_dataset = EnhancedZarrTileDataset(
            base_path=final_config['val_path'],
            tile_ids=final_config['val_tiles'],
            patch_size=final_config['patch_size'],
            label_variable='label',
            stats_file='stats.json',
            calculate_stats=False,
            gpu_id=0,
            balanced_sampling=True,
            samples_per_tile=final_config['samples_per_tile']
        )
        
        # Train final model
        model = DualStreamUNET3D(final_config)
        
        # First train encoders
        train_single_stream('sentinel1', final_config, train_dataset, val_dataset)
        train_single_stream('sentinel2', final_config, train_dataset, val_dataset)
        
        # Load pretrained encoders and train full model
        model.load_pretrained_encoders('sentinel1_encoder.pth', 'sentinel2_encoder.pth')
        
        mp.spawn(
            train_distributed,
            args=(num_gpus, model, train_dataset, val_dataset, final_config),
            nprocs=num_gpus,
            join=True
        )
        
        print("\nFinal training completed!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nCleaning up...")
        torch.cuda.empty_cache()
        if os.path.exists('best_model.pth'):
            print("Best model saved as 'best_model.pth'")
        if os.path.exists('results.json'):
            print("Results saved as 'results.json'")

if __name__ == "__main__":
    main()