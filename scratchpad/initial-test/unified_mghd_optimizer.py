"""
Comprehensive MGHD Hyperparameter Optimizer
Combines basic Optuna, advanced optimization strategies, and direct training
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

# Quantum error correction imports
from panqec.codes import surface_2d
from panqec.error_models import PauliErrorModel
from torch.utils.data import DataLoader

# Local imports
from panq_functions import (GNNDecoder, collate, fraction_of_solved_puzzles, 
                           logical_error_rate, surface_code_edges, 
                           generate_syndrome_error_volume, adapt_trainset)
from ldpc.mod2 import nullspace
from poc_my_models import MGHD

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
    use_amp = True
    amp_data_type = torch.float16
else:
    device = torch.device('cpu')
    use_amp = True
    amp_data_type = torch.bfloat16

@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies"""
    strategy: str = 'basic'  # 'basic', 'advanced', 'ensemble', 'curriculum', 'direct'
    n_trials: int = 30
    ensemble_size: int = 5
    epochs: int = 20
    use_full_dataset: bool = False  # True for final training
    seed: int = 42

class ComprehensiveMGHDOptimizer:
    """
    All-in-one MGHD optimization: basic Optuna, advanced strategies, and direct training
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.setup_quantum_environment()
        
    def setup_quantum_environment(self):
        """Setup quantum error correction environment"""
        self.d = 3
        self.error_model = PauliErrorModel(0.34, 0.32, 0.34)  # DP error model
        self.code = surface_2d.RotatedPlanar2DCode(self.d)
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
        
        # Setup graph structure
        src, tgt = surface_code_edges(self.code)
        src_tensor = torch.LongTensor(src)
        tgt_tensor = torch.LongTensor(tgt)
        GNNDecoder.surface_code_edges = (src_tensor, tgt_tensor)
        
        # Setup nullspaces
        hxperp = torch.FloatTensor(nullspace(self.code.Hx.toarray())).to(device)
        hzperp = torch.FloatTensor(nullspace(self.code.Hz.toarray())).to(device)
        GNNDecoder.hxperp = hxperp
        GNNDecoder.hzperp = hzperp
        GNNDecoder.device = device
        
    def get_dataset_params(self):
        """Get dataset parameters based on configuration"""
        if self.config.use_full_dataset or self.config.strategy == 'direct':
            return {
                'len_train_set': 20000,
                'len_test_set': 5000,
                'batch_size': 128
            }
        else:
            # For search phase, use same dataset size as final to ensure consistency
            return {
                'len_train_set': 20000,  
                'len_test_set': 5000,    
                'batch_size': 128
            }
    
    def basic_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Basic hyperparameter search space (focused around proven best parameters)"""
        return {
            # Training parameters - narrow range around your best
            'lr': trial.suggest_float('lr', 5e-5, 2e-4, log=True),  # Around 1e-4
            'weight_decay': trial.suggest_float('weight_decay', 5e-5, 2e-4, log=True),  # Around 1e-4
            
            # GNN Architecture - focused on working ranges
            'n_iters': trial.suggest_int('n_iters', 6, 10),  # Around 8
            'n_node_features': trial.suggest_categorical('n_node_features', [128, 256, 384]),  # Around 256
            'n_edge_features': trial.suggest_categorical('n_edge_features', [128, 256, 384]),  # Around 256
            'msg_net_size': trial.suggest_categorical('msg_net_size', [96, 128, 160]),  # Around 128
            'msg_net_dropout_p': trial.suggest_float('msg_net_dropout_p', 0.02, 0.06),  # Around 0.037
            'gru_dropout_p': trial.suggest_float('gru_dropout_p', 0.08, 0.15),  # Around 0.11
            
            # Mamba parameters - focused around your best
            'mamba_d_model': trial.suggest_categorical('mamba_d_model', [192, 256, 320]),  # Around 256
            'mamba_d_state': trial.suggest_int('mamba_d_state', 40, 70),  # Around 55
            'mamba_d_conv': trial.suggest_int('mamba_d_conv', 2, 4),  # Keep constraint
            'mamba_expand': trial.suggest_int('mamba_expand', 3, 5),  # Around 4
        }
    
    def advanced_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Advanced hyperparameter search space with novel parameters"""
        base_params = self.basic_search_space(trial)
        
        # Sample attention mechanism once
        attn = trial.suggest_categorical('attention_mechanism', [
            'none', 'channel_attention', 'cross_attention'
        ])
        
        # Add advanced parameters
        advanced_params = {
            # Learning rate scheduling
            'lr_schedule': trial.suggest_categorical('lr_schedule', ['constant', 'cosine', 'exponential']),
            'warmup_steps': trial.suggest_int('warmup_steps', 0, 500),
            
            # Advanced regularization
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.15),
            'gradient_clip': trial.suggest_float('gradient_clip', 0.5, 5.0),
            
            # Architecture innovations
            'residual_connections': trial.suggest_int('residual_connections', 0, 2),
            'attention_mechanism': attn,
            'attention_heads': 1,  # kept for back-compat, unused
            'attention_dim': base_params['n_node_features'],  # unused in SE/FiLM
            
            # Training stability
            'noise_injection': trial.suggest_float('noise_injection', 0.0, 0.03),
            'accumulation_steps': trial.suggest_int('accumulation_steps', 1, 4),
            
            # Multiple Mamba layers - LATENCY CONSTRAINED
            'mamba_layers': trial.suggest_int('mamba_layers', 1, 2),
        }
        
        # Conditional parameters - only when needed
        if attn == 'channel_attention':
            advanced_params['se_reduction'] = trial.suggest_categorical('se_reduction', [2, 4, 8])
        if attn == 'cross_attention':
            advanced_params['film_hidden'] = trial.suggest_categorical('film_hidden', [16, 32, 64])
        
        return {**base_params, **advanced_params}
    
    def _violates_latency_budget(self, params: Dict[str, Any]) -> bool:
        """Check if hyperparameter combination exceeds latency budget"""
        # Simple MAC-based proxy for latency
        mamba_cost = params['mamba_d_model'] * params.get('mamba_layers', 1) 
        gnn_cost = params['n_node_features'] * params['n_iters']
        total_budget = mamba_cost + gnn_cost
        
        # Very relaxed threshold to allow fair comparison of all attention mechanisms
        return total_budget > 2500
    
    def objective_function(self, trial: optuna.Trial) -> float:
        """Unified objective function for all optimization strategies"""
        
        # Get hyperparameters based on strategy
        if self.config.strategy in ['basic', 'direct']:
            params = self.basic_search_space(trial)
        else:
            params = self.advanced_search_space(trial)
        
        # Early pruning for latency budget (except direct strategy)
        if self.config.strategy != 'direct' and self._violates_latency_budget(params):
            raise optuna.exceptions.TrialPruned()
        
        dataset_params = self.get_dataset_params()
        
        print(f"\\nTrial {trial.number}: lr={params['lr']:.6f}, "
              f"n_node_features={params['n_node_features']}, "
              f"mamba_d_model={params['mamba_d_model']}")
        
        try:
            # Create model
            gnn_params = {
                'dist': self.d, 'n_node_inputs': 4, 'n_node_outputs': 4,
                'n_iters': params['n_iters'],
                'n_node_features': params['n_node_features'],
                'n_edge_features': params['n_edge_features'],
                'msg_net_size': params['msg_net_size'],
                'msg_net_dropout_p': params['msg_net_dropout_p'],
                'gru_dropout_p': params['gru_dropout_p']
            }
            
            mamba_params = {
                'd_model': params['mamba_d_model'],
                'd_state': params['mamba_d_state'],
                'd_conv': params['mamba_d_conv'],
                'expand': params['mamba_expand'],
                'attention_mechanism': params.get('attention_mechanism', 'none'),
                'se_reduction': params.get('se_reduction', 4),
                'film_hidden': params.get('film_hidden', None),
                'mamba_layers': params.get('mamba_layers', 1)
            }
            
            model = MGHD(gnn_params=gnn_params, mamba_params=mamba_params).to(device)
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=params['lr'], 
                                  weight_decay=params['weight_decay'])
            
            # Setup scheduler with warmup
            scheduler = None
            if params.get('lr_schedule', 'constant') != 'constant' or params.get('warmup_steps', 0) > 0:
                total_epochs = self.config.epochs
                base_sched = None
                if params.get('lr_schedule') == 'cosine':
                    base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
                elif params.get('lr_schedule') == 'exponential':
                    base_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
                
                # Wrap with warmup
                warmup = params.get('warmup_steps', 0)
                if warmup > 0:
                    def lr_lambda(epoch):
                        if epoch < warmup:
                            return (epoch + 1) / float(max(1, warmup))
                        return 1.0
                    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                    # Simple compose: step warmup first, then base
                    scheduler = (warmup_sched, base_sched)
                else:
                    scheduler = base_sched
            
            # Generate data
            trainset = adapt_trainset(
                generate_syndrome_error_volume(
                    self.code, self.error_model, p=0.15, 
                    batch_size=dataset_params['len_train_set']
                ), self.code, num_classes=4
            )
            trainloader = DataLoader(trainset, batch_size=dataset_params['batch_size'], 
                                   collate_fn=collate, shuffle=True)
            
            testset = adapt_trainset(
                generate_syndrome_error_volume(
                    self.code, self.error_model, p=0.05, 
                    batch_size=dataset_params['len_test_set'], for_training=False
                ), self.code, num_classes=4, for_training=False
            )
            testloader = DataLoader(testset, batch_size=512, collate_fn=collate, shuffle=False)
            
            # Training loop
            criterion = nn.CrossEntropyLoss(label_smoothing=params.get('label_smoothing', 0.0))
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
            best_ler = float('inf')
            accum = params.get('accumulation_steps', 1)
            grad_clip = params.get('gradient_clip', 0.0)
            
            for epoch in range(self.config.epochs):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                epoch_losses = []
                
                for batch_idx, (inputs, targets, src_ids, dst_ids) in enumerate(trainloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
                    
                    with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
                        outputs = model(inputs, src_ids, dst_ids)  # Correct call signature
                        loss = criterion(outputs[-1], targets) / accum
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % accum == 0:
                        if grad_clip and grad_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    
                    epoch_losses.append(loss.item() * accum)
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    _, _, ler_total = logical_error_rate(model, testloader, self.code)
                    if ler_total < best_ler:
                        best_ler = ler_total
                
                # Step scheduler(s)
                if scheduler is not None:
                    if isinstance(scheduler, tuple):
                        warmup_sched, base_sched = scheduler
                        warmup_sched.step()
                        if base_sched is not None and epoch >= params.get('warmup_steps', 0):
                            base_sched.step()
                    else:
                        scheduler.step()
                
                # Report for pruning
                trial.report(ler_total, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            # Apply tiny latency penalty to encourage efficient models
            latency_penalty = 1e-3 * (params['mamba_d_model']/256.0) * params.get('mamba_layers', 1)
            return best_ler + latency_penalty
            
        except Exception as e:
            if "causal_conv1d only supports width between 2 and 4" in str(e):
                return float('inf')
            else:
                raise
    
    def run_basic_optimization(self) -> Dict[str, Any]:
        """Run basic Optuna optimization"""
        print("🔬 Running Basic Optuna Optimization...")
        
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
        )
        
        study.optimize(self.objective_function, n_trials=self.config.n_trials)
        
        print(f"\\nBest LER: {study.best_value:.6f}")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        return study.best_params
    
    def run_ensemble_optimization(self) -> Tuple[List[Dict], List[float]]:
        """Run ensemble optimization"""
        print("🎪 Running Ensemble Optimization...")
        
        # Generate diverse hyperparameter sets
        diverse_params = self._generate_diverse_params(self.config.ensemble_size)
        
        ensemble_results = []
        for i, params in enumerate(diverse_params):
            print(f"Training ensemble member {i+1}/{len(diverse_params)}")
            
            # Create a mock trial for compatibility
            class MockTrial:
                def __init__(self, params):
                    self.params = params
                    self.number = i
                def suggest_float(self, name, *args, **kwargs):
                    return self.params[name]
                def suggest_int(self, name, *args, **kwargs):
                    return self.params[name]
                def suggest_categorical(self, name, *args, **kwargs):
                    return self.params[name]
                def report(self, *args): pass
                def should_prune(self): return False
            
            mock_trial = MockTrial(params)
            ler = self.objective_function(mock_trial)
            ensemble_results.append((params, ler))
        
        # Sort by performance
        ensemble_results.sort(key=lambda x: x[1])
        
        print("\\nEnsemble Results:")
        for i, (params, ler) in enumerate(ensemble_results):
            print(f"  Member {i+1}: LER = {ler:.6f}")
        
        best_params = ensemble_results[0][0]
        best_lers = [result[1] for result in ensemble_results]
        
        return [result[0] for result in ensemble_results], best_lers
    
    def _generate_diverse_params(self, n_models: int) -> List[Dict]:
        """Generate diverse hyperparameter sets"""
        # Load best known parameters as baseline
        try:
            with open('best_hyperparameters.json', 'r') as f:
                best_params = json.load(f)
                base_lr = best_params['training_parameters']['lr']
                base_features = best_params['model_architecture']['n_node_features']
        except:
            # Fallback defaults
            base_lr = 0.0001
            base_features = 256
        
        diverse_params = []
        for i in range(n_models):
            # Create variations
            scale_factors = [0.5, 0.75, 1.0, 1.25, 1.5]
            lr_factors = [0.3, 0.5, 1.0, 1.5, 2.0]
            
            scale = scale_factors[i % len(scale_factors)]
            lr_scale = lr_factors[i % len(lr_factors)]
            
            params = {
                'lr': base_lr * lr_scale,
                'weight_decay': 0.0001 * lr_scale,
                'n_iters': 6 + (i % 4),
                'n_node_features': int(base_features * scale),
                'n_edge_features': int(base_features * scale),
                'msg_net_size': 128,
                'msg_net_dropout_p': 0.05 + (i % 3) * 0.02,
                'gru_dropout_p': 0.1 + (i % 3) * 0.03,
                'mamba_d_model': int(256 * scale),
                'mamba_d_state': 16 + (i % 3) * 16,
                'mamba_d_conv': 2 + (i % 3),
                'mamba_expand': 2 + (i % 4),
            }
            diverse_params.append(params)
        
        return diverse_params
    
    def train_with_best_params(self, best_params: Dict[str, Any], 
                             compare_with_baseline: bool = True) -> Dict[str, Any]:
        """Train final model with best parameters and compare with baseline"""
        print("🏆 Training with Best Parameters...")
        
        # Use full dataset for final training
        self.config.use_full_dataset = True
        dataset_params = self.get_dataset_params()
        
        # Create optimized model
        gnn_params = {
            'dist': self.d, 'n_node_inputs': 4, 'n_node_outputs': 4,
            'n_iters': best_params['n_iters'],
            'n_node_features': best_params['n_node_features'],
            'n_edge_features': best_params['n_edge_features'],
            'msg_net_size': best_params['msg_net_size'],
            'msg_net_dropout_p': best_params['msg_net_dropout_p'],
            'gru_dropout_p': best_params['gru_dropout_p']
        }
        
        mamba_params = {
            'd_model': best_params['mamba_d_model'],
            'd_state': best_params['mamba_d_state'],
            'd_conv': best_params['mamba_d_conv'],
            'expand': best_params['mamba_expand']
        }
        
        optimized_model = MGHD(gnn_params=gnn_params, mamba_params=mamba_params).to(device)
        optimizer_opt = optim.AdamW(optimized_model.parameters(), 
                                  lr=best_params['lr'], 
                                  weight_decay=best_params['weight_decay'])
        
        print(f"Optimized model parameters: {sum(p.numel() for p in optimized_model.parameters()):,}")
        print(f"Using LR: {best_params['lr']:.6f}, Weight Decay: {best_params['weight_decay']:.6f}")
        
        # Add ReduceLROnPlateau scheduler (from your best config)
        scheduler_opt = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_opt, mode='min', factor=0.5, patience=3
        )
        
        models_to_train = [('Optimized MGHD', optimized_model, optimizer_opt, scheduler_opt)]
        
        # Add baseline if requested
        if compare_with_baseline:
            # Use ORIGINAL unoptimized parameters for true comparison
            baseline_gnn_params = {
                'dist': self.d, 'n_node_inputs': 4, 'n_node_outputs': 4,
                'n_iters': 5, 'n_node_features': 64, 'n_edge_features': 64,  # Original values
                'msg_net_size': 128, 'msg_net_dropout_p': 0.05, 'gru_dropout_p': 0.05  # Original values
            }
            baseline_mamba_params = {
                'd_model': 64, 'd_state': 16, 'd_conv': 4, 'expand': 2  # Original values
            }
            
            baseline_model = MGHD(gnn_params=baseline_gnn_params, mamba_params=baseline_mamba_params).to(device)
            optimizer_base = optim.AdamW(baseline_model.parameters(), lr=0.0001, weight_decay=0.0001)
            
            print(f"Baseline model parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
            print(f"Baseline using original unoptimized architecture")
            
            # Add baseline scheduler too
            scheduler_base = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_base, mode='min', factor=0.5, patience=3
            )
            
            models_to_train.append(('Baseline MGHD', baseline_model, optimizer_base, scheduler_base))
        
        # Generate full dataset
        trainset = adapt_trainset(
            generate_syndrome_error_volume(
                self.code, self.error_model, p=0.15, 
                batch_size=dataset_params['len_train_set']
            ), self.code, num_classes=4
        )
        trainloader = DataLoader(trainset, batch_size=dataset_params['batch_size'], 
                               collate_fn=collate, shuffle=True)  # Match search conditions
        
        testset = adapt_trainset(
            generate_syndrome_error_volume(
                self.code, self.error_model, p=0.05, 
                batch_size=dataset_params['len_test_set'], for_training=False
            ), self.code, num_classes=4, for_training=False
        )
        testloader = DataLoader(testset, batch_size=512, collate_fn=collate, shuffle=False)
        
        # Training
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        results = {}
        epochs = 25 if self.config.strategy == 'direct' else 30  # Use 25 epochs for direct to match search conditions
        
        print(f"\\nTraining for {epochs} epochs...")
        print("epoch, " + ", ".join([f"{name.replace(' ', '_').lower()}_loss, {name.replace(' ', '_').lower()}_LER" for name, _, _, _ in models_to_train]))
        
        for epoch in range(epochs):
            epoch_results = []
            
            # Train all models
            for name, model, optimizer, scheduler in models_to_train:
                model.train()
                epoch_losses = []
                
                for inputs, targets, src_ids, dst_ids in trainloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)
                    
                    optimizer.zero_grad()
                    with torch.autocast(device_type=device.type, dtype=amp_data_type, enabled=use_amp):
                        outputs = model(inputs, src_ids, dst_ids)
                        loss = criterion(outputs[-1], targets)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_losses.append(loss.item())
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    _, _, ler_total = logical_error_rate(model, testloader, self.code)
                
                avg_loss = np.mean(epoch_losses)
                epoch_results.extend([avg_loss, ler_total])
                
                # Step scheduler (ReduceLROnPlateau needs LER)
                scheduler.step(ler_total)
                
                # Store results
                if name not in results:
                    results[name] = {'losses': [], 'lers': [], 'best_ler': float('inf')}
                results[name]['losses'].append(avg_loss)
                results[name]['lers'].append(ler_total)
                
                # Track best LER like in search phase
                if ler_total < results[name]['best_ler']:
                    results[name]['best_ler'] = ler_total
            
            print(f"{epoch+1}, " + ", ".join([f"{val:.6f}" for val in epoch_results]))
        
        # Final comparison
        print("\\n=== FINAL COMPARISON ===")
        for name in results:
            final_ler = results[name]['lers'][-1]
            best_ler = results[name]['best_ler']
            print(f"{name} - Final LER: {final_ler:.6f}, Best LER: {best_ler:.6f}")
        
        if len(results) > 1:
            opt_best = results['Optimized MGHD']['best_ler']
            base_best = results['Baseline MGHD']['best_ler']
            improvement = (base_best - opt_best) / base_best * 100
            print(f"Optimization improved performance by {improvement:.2f}% (comparing best LERs)!")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"optimization_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'best_params': best_params,
                'training_results': {name: {'final_ler': results[name]['lers'][-1]} 
                                   for name in results},
                'strategy': self.config.strategy,
                'n_trials': self.config.n_trials
            }, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        return results
    
    def run_optimization(self) -> Dict[str, Any]:
        """Run optimization based on configured strategy"""
        
        print(f"🚀 Starting {self.config.strategy.upper()} Optimization Strategy")
        print(f"Trials: {self.config.n_trials}, Epochs: {self.config.epochs}")
        
        if self.config.strategy == 'basic':
            best_params = self.run_basic_optimization()
            
        elif self.config.strategy == 'ensemble':
            ensemble_params, ensemble_lers = self.run_ensemble_optimization()
            best_params = ensemble_params[0]  # Best performing member
            
        elif self.config.strategy == 'advanced':
            best_params = self.run_basic_optimization()  # Uses advanced search space
            
        elif self.config.strategy == 'direct':
            # Use latency-optimized defaults with channel attention
            best_params = {
                'lr': 1e-4, 'weight_decay': 1e-4,
                'n_iters': 8, 'n_node_features': 256, 'n_edge_features': 256,
                'msg_net_size': 128, 'msg_net_dropout_p': 0.04, 'gru_dropout_p': 0.11,
                'mamba_d_model': 256, 'mamba_d_state': 48, 'mamba_d_conv': 2, 'mamba_expand': 4,
                'attention_mechanism': 'channel_attention', 'se_reduction': 4, 'mamba_layers': 1,
                'lr_schedule': 'cosine', 'warmup_steps': 200,
                'label_smoothing': 0.05, 'gradient_clip': 1.0,
                'residual_connections': 1, 'noise_injection': 0.01, 'accumulation_steps': 1
            }
            print("Using latency-optimized defaults with channel attention")
            
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
        
        # Train final model with best parameters
        results = self.train_with_best_params(best_params)
        
        return {
            'best_params': best_params,
            'training_results': results,
            'strategy': self.config.strategy
        }

def main():
    parser = argparse.ArgumentParser(description='Comprehensive MGHD Optimization')
    parser.add_argument('--strategy', choices=['basic', 'advanced', 'ensemble', 'direct'], 
                       default='basic', help='Optimization strategy')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs per trial')
    parser.add_argument('--ensemble-size', type=int, default=5, help='Ensemble size')
    
    args = parser.parse_args()
    
    config = OptimizationConfig(
        strategy=args.strategy,
        n_trials=args.trials,
        epochs=args.epochs,
        ensemble_size=args.ensemble_size
    )
    
    optimizer = ComprehensiveMGHDOptimizer(config)
    results = optimizer.run_optimization()
    
    print("\\n🏆 Optimization Complete!")
    print(f"Strategy: {results['strategy']}")
    print(f"Best LER achieved: {min([res['lers'][-1] for res in results['training_results'].values()]):.6f}")

if __name__ == "__main__":
    main()
