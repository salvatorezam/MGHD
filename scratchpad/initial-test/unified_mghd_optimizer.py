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
import os
import sys
import subprocess
from pathlib import Path

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
            # AMP scaler: enable only on CUDA; use a no-op scaler on CPU
            if device.type == 'cuda':
                from torch.cuda.amp import GradScaler as _GradScaler
                scaler = _GradScaler(enabled=use_amp)
            else:
                class _NoopScaler:
                    def scale(self, x): return x
                    def step(self, opt): opt.step()
                    def update(self): pass
                    def unscale_(self, opt): pass
                scaler = _NoopScaler()
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
                    
                    with torch.autocast(device_type='cuda', dtype=amp_data_type, enabled=(use_amp and device.type == 'cuda')):
                        outputs = model(inputs, src_ids, dst_ids)  # Correct call signature
                        loss = criterion(outputs[-1], targets) / accum
                    
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % accum == 0:
                        if grad_clip and grad_clip > 0:
                            try:
                                scaler.unscale_(optimizer)
                            except Exception:
                                pass
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
        # AMP scaler: enable only on CUDA; use a no-op scaler on CPU
        if device.type == 'cuda':
            from torch.cuda.amp import GradScaler as _GradScaler
            scaler = _GradScaler(enabled=use_amp)
        else:
            class _NoopScaler:
                def scale(self, x): return x
                def step(self, opt): opt.step()
                def update(self): pass
                def unscale_(self, opt): pass
            scaler = _NoopScaler()
        
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
                    with torch.autocast(device_type='cuda', dtype=amp_data_type, enabled=(use_amp and device.type == 'cuda')):
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
    parser = argparse.ArgumentParser(description='Comprehensive MGHD Optimization + Step-11')
    # Legacy optimization path
    parser.add_argument('--strategy', choices=['basic', 'advanced', 'ensemble', 'direct'], 
                       default='basic', help='Optimization strategy')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs per trial')
    parser.add_argument('--ensemble-size', type=int, default=5, help='Ensemble size')

    # Foundation training (GPU-only)
    # Back-compat: keep --step11-train as alias for --foundation-train
    parser.add_argument('--foundation-train', action='store_true', help='Run Garnet foundation training (formerly Step-11)')
    parser.add_argument('--step11-train', action='store_true', help='Alias for --foundation-train (deprecated)')
    parser.add_argument('--profile', choices=['S','M','L','XL','Lplus'], default='S')
    parser.add_argument('--garnet-mode', choices=['foundation','student'], default='foundation')
    parser.add_argument('--teacher-ensemble', default='mwpf+mwpm')
    parser.add_argument('--teacher', choices=['mwpf','mwpm','lut','ensemble','mwpf+mwpm'], default='mwpm',
                        help='Teacher for labels during training/validation (default: mwpm for robust d=3)')
    # Coset regularizer and p-aware smoothing
    parser.add_argument('--coset-reg', type=float, default=0.01,
                        help='Small weight for coset/teacher consistency (|sigmoid(logits)-y|)')
    parser.add_argument('--smoothing-base', type=float, default=0.09,
                        help='Base label smoothing (overrides --label-smoothing)')
    parser.add_argument('--smoothing-highp-threshold', type=float, default=0.05,
                        help='p threshold above which to reduce smoothing')
    parser.add_argument('--smoothing-highp-factor', type=float, default=0.6,
                        help='Multiply smoothing by this factor when p_train >= threshold')
    parser.add_argument('--steps-per-epoch', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--amp', choices=['bf16','fp16','off'], default='bf16')
    parser.add_argument('--outdir', type=str, default='results/step11')
    parser.add_argument('--seed', type=int, default=42)
    # Fine-tune / eval controls
    parser.add_argument('--train-p', type=float, default=None, help='Fix training p (e.g., 0.05); default uses curriculum')
    parser.add_argument('--val-N', type=int, default=1024, help='Validation shots per epoch (default: 1024)')
    parser.add_argument('--init-ckpt', type=str, default=None, help='Optional checkpoint to initialize model weights')
    parser.add_argument('--val-p', type=float, default=0.05, help='Validation physical error rate p (default: 0.05)')
    # Custom training p-grid (overrides built-in curriculum when provided)
    parser.add_argument('--train-p-grid', type=str, default=None,
                        help='Comma-separated training p grid (e.g., "0.002,0.003,0.005,0.008"). Overrides fixed --train-p and curriculum.')
    parser.add_argument('--train-p-weights', type=str, default=None,
                        help='Comma-separated weights aligned with --train-p-grid (e.g., "0.5,0.3,0.15,0.05"). Optional; uniform if omitted.')
    # Curriculum preset for circuit-level p-schedule
    parser.add_argument('--curriculum', choices=['none','circuit_v1'], default='none',
                        help='Enable circuit-level p scheduling over epochs')
    # S-arch overrides and improvements
    parser.add_argument('--ov-n-iters', type=int, default=None)
    parser.add_argument('--ov-node-feats', type=int, default=None)
    parser.add_argument('--ov-edge-feats', type=int, default=None)
    parser.add_argument('--ov-msg-size', type=int, default=None)
    parser.add_argument('--ov-msg-drop', type=float, default=None)
    parser.add_argument('--ov-gru-drop', type=float, default=None)
    parser.add_argument('--ov-mamba-d-model', type=int, default=None)
    parser.add_argument('--ov-mamba-d-state', type=int, default=None)
    parser.add_argument('--ov-mamba-expand', type=int, default=None)
    parser.add_argument('--post-mamba-ln', action='store_true')
    parser.add_argument('--ema-decay', type=float, default=0.0)
    parser.add_argument('--parity-lambda', type=float, default=0.0)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--lr-schedule', choices=['cosine','constant'], default='cosine')
    # S-arch overrides and improvements
    

    args = parser.parse_args()

    if getattr(args, 'foundation_train', False) or getattr(args, 'step11_train', False):
        return run_foundation_train(args)

    # Legacy path
    config = OptimizationConfig(
        strategy=args.strategy,
        n_trials=args.trials,
        epochs=args.epochs,
        ensemble_size=args.ensemble_size
    )

    optimizer = ComprehensiveMGHDOptimizer(config)
    results = optimizer.run_optimization()

    print("\n🏆 Optimization Complete!")
    print(f"Strategy: {results['strategy']}")
    print(f"Best LER achieved: {min([res['lers'][-1] for res in results['training_results'].values()]):.6f}")


def run_foundation_train(args):
    """Step-11 training loop with CUDA-Q sampler (fallback: numpy+LUT).

    - Streams synthetic syndrome/label batches from tools/cudaq_sampler.py
    - Binary head via logit difference and BCEWithLogits
    - Cosine schedule w/ warmup, grad clip, bf16 AMP
    - Saves best checkpoint by coset-validated val metric
    - Calls tools/eval_ler.py at the end; writes small handoff JSON
    """
    import torch
    import numpy as np
    import signal, tempfile, os as _os
    from tools.cudaq_sampler import CudaqGarnetSampler, get_code_mats
    from tools.eval_ler import _coset_success  # reuse logic locally
    from poc_my_models import MGHD

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # H matrices for validation/coset checks (authoritative)
    Hx, Hz, meta = get_code_mats()
    Hx_t = torch.from_numpy(Hx.astype(np.uint8))
    Hz_t = torch.from_numpy(Hz.astype(np.uint8))
    # Masks for parity loss (shape [4,9])
    _mask_x = (Hx_t.to(torch.float32) > 0)
    _mask_z = (Hz_t.to(torch.float32) > 0)

    # Model profiles
    profiles = {
        'S': dict(n_iters=7, n_node_features=128, n_edge_features=128, msg_net=96, d_model=192, d_state=32),
        'M': dict(n_iters=8, n_node_features=192, n_edge_features=192, msg_net=128, d_model=256, d_state=48),
        'L': dict(n_iters=9, n_node_features=256, n_edge_features=256, msg_net=160, d_model=320, d_state=64),
        # XL (L+) modest capacity bump: +1 iter, wider Mamba + head
        'XL': dict(n_iters=10, n_node_features=256, n_edge_features=256, msg_net=192, d_model=384, d_state=64),
        'Lplus': dict(n_iters=10, n_node_features=256, n_edge_features=256, msg_net=192, d_model=384, d_state=64),
    }
    pf = profiles[args.profile]

    gnn_params = dict(
        dist=3, n_node_inputs=9, n_node_outputs=9,  # MGHD will adapt for rotated
        n_iters=pf['n_iters'], n_node_features=pf['n_node_features'], n_edge_features=pf['n_edge_features'],
        msg_net_size=pf['msg_net'], msg_net_dropout_p=0.04, gru_dropout_p=0.11,
    )
    # Apply overrides if provided (for S tuning)
    if getattr(args, 'ov_n_iters', None) is not None:
        gnn_params['n_iters'] = args.ov_n_iters
    if getattr(args, 'ov_node_feats', None) is not None:
        gnn_params['n_node_features'] = args.ov_node_feats
    if getattr(args, 'ov_edge_feats', None) is not None:
        gnn_params['n_edge_features'] = args.ov_edge_feats
    if getattr(args, 'ov_msg_size', None) is not None:
        gnn_params['msg_net_size'] = args.ov_msg_size
    mamba_params = dict(d_model=pf['d_model'], d_state=pf['d_state'], d_conv=2, expand=3,
                        attention_mechanism='channel_attention', se_reduction=4,
                        post_mamba_ln=bool(getattr(args, 'post_mamba_ln', False)))
    if getattr(args, 'ov_msg_drop', None) is not None:
        gnn_params['msg_net_dropout_p'] = args.ov_msg_drop
    if getattr(args, 'ov_gru_drop', None) is not None:
        gnn_params['gru_dropout_p'] = args.ov_gru_drop
    if getattr(args, 'ov_mamba_d_model', None) is not None:
        mamba_params['d_model'] = args.ov_mamba_d_model
    if getattr(args, 'ov_mamba_d_state', None) is not None:
        mamba_params['d_state'] = args.ov_mamba_d_state
    if getattr(args, 'ov_mamba_expand', None) is not None:
        mamba_params['expand'] = args.ov_mamba_expand
    model = MGHD(gnn_params=gnn_params, mamba_params=mamba_params).to(device)
    # Optional init from checkpoint for fine-tuning
    if getattr(args, 'init_ckpt', None):
        try:
            state = torch.load(args.init_ckpt, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"[Foundation] Loaded init checkpoint: {args.init_ckpt}")
        except Exception as e:
            print(f"[Foundation] Warning: failed to load init checkpoint: {e}")
    # Enforce rotated layout behavior (8 checks + 9 qubits; binary head)
    try:
        model.set_rotated_layout()
        # Align MGHD graph indices with authoritative H matrices to ensure parity alignment with sampler/teacher
        try:
            import numpy as _np
            model.set_authoritative_mats(Hx, Hz, device=device)
        except Exception:
            pass
    except Exception:
        pass
    model.train()
    # Ensure graph indices are built once
    try:
        model._ensure_static_indices(device)
    except Exception:
        pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # LR schedule
    total_epochs = args.epochs
    if getattr(args, 'lr_schedule', 'cosine') == 'cosine':
        warmup = max(1, int(0.05 * total_epochs))
        base_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        def lr_lambda(epoch):
            if epoch < warmup:
                return (epoch + 1) / float(max(1, warmup))
            return 1.0
        warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        base_sched = None
        warmup_sched = None

    # AMP settings
    use_amp = args.amp != 'off'
    amp_dtype = torch.bfloat16 if args.amp == 'bf16' else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')

    if args.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    sampler = CudaqGarnetSampler(args.garnet_mode)
    # EMA setup
    ema_decay = float(getattr(args, 'ema_decay', 0.0))
    use_ema = ema_decay > 0.0
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point} if use_ema else None

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / f"step11_garnet_{args.profile}_best.pt"
    ckpt_last = outdir / f"step11_garnet_{args.profile}_last.pt"
    # Write run manifest (repro + paper assets)
    try:
        # Command line
        (outdir / 'cmd.txt').write_text(' '.join(map(str, sys.argv)))
        # Args JSON
        import json as _json
        (outdir / 'args.json').write_text(_json.dumps(vars(args), indent=2, default=str))
        # Env summary
        import torch
        env = {
            'python': sys.version,
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'),
            'amp': args.amp,
            'seed': args.seed,
        }
        (outdir / 'env.json').write_text(_json.dumps(env, indent=2))
        # Metrics CSV header
        import csv as _csv
        with open(outdir / 'metrics.csv', 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['epoch','train_loss_mean','val_ler','samples_epoch','mwpf_shots_cum','mwpm_shots_cum'])
    except Exception:
        pass

    best_val = 1.0
    history = []
    # Graceful stop flag for SIGTERM/SIGINT
    _stop_flag = {"stop": False}

    def _signal_handler(sig, frame):
        try:
            print(f"[Foundation] Received signal {sig}; will save last checkpoint and stop after current step.")
        except Exception:
            pass
        _stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    def _save_last_ckpt(epoch_idx: int):
        try:
            payload = {
                'epoch': int(epoch_idx),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'warmup_sched': getattr(warmup_sched, 'state_dict', lambda: {})(),
                'base_sched': getattr(base_sched, 'state_dict', lambda: {})(),
                'best_val_ler': float(best_val),
                'args': vars(args),
                'teacher_stats': sampler.stats_snapshot(),
                'rng_numpy': np.random.get_state()[1].tolist(),
                'rng_torch': torch.random.get_rng_state().cpu().tolist(),
            }
            with tempfile.NamedTemporaryFile(delete=False, dir=str(outdir), suffix='.pt.tmp') as tf:
                tmp_path = Path(tf.name)
            torch.save(payload, tmp_path)
            _os.replace(tmp_path, ckpt_last)
        except Exception:
            pass

    # Curriculum p-schedule for training
    rng = np.random.default_rng(args.seed)
    # Pre-parse training p-grid if provided
    train_grid = None
    train_w = None
    if getattr(args, 'train_p_grid', None):
        try:
            train_grid = [float(x) for x in str(args.train_p_grid).split(',') if x]
            if getattr(args, 'train_p_weights', None):
                train_w = [float(x) for x in str(args.train_p_weights).split(',') if x]
                s = sum(train_w)
                if s > 0:
                    train_w = [w/s for w in train_w]
            # If weights missing or mismatched, fall back to uniform
            if (not train_w) or (len(train_w) != len(train_grid)):
                train_w = [1.0/len(train_grid)]*len(train_grid)
        except Exception:
            train_grid, train_w = None, None

    # Circuit-level curriculum default grid and buckets
    CIRCUIT_GRID = [0.001,0.002,0.003,0.004,0.005,0.006,0.008,0.010,0.012,0.015]
    def _bucket_weight(p: float) -> float:
        if 0.001 <= p <= 0.004: return 0.40/4.0  # spread uniformly within bucket
        if 0.004 < p <= 0.008: return 0.45/4.0
        if 0.008 < p <= 0.015: return 0.15/3.0
        return 0.0
    def _stage_boost(p: float, epoch: int, total_epochs: int) -> float:
        # Stage windows: warm (epochs 1-2): 0.008–0.010; mid (next ~5): 0.005–0.007; final: 0.003–0.006
        if epoch < 2:
            return 3.0 if (0.008 <= p <= 0.010) else 1.0
        elif epoch < 7:
            return 2.0 if (0.005 <= p <= 0.007) else 1.0
        else:
            return 2.0 if (0.003 <= p <= 0.006) else 1.0
    for epoch in range(args.epochs):
        epoch_losses = []
        samples_epoch = 0
        nan_batches = 0
        for step in range(args.steps_per_epoch):
            # Choose p with precedence: explicit grid > fixed --train-p > curriculum preset > built-in
            if train_grid is not None:
                p_train = float(rng.choice(train_grid, p=train_w))
            elif getattr(args, 'train_p', None) is not None:
                p_train = float(args.train_p)
            elif getattr(args, 'curriculum', 'none') == 'circuit_v1':
                weights = np.array([_bucket_weight(p) * _stage_boost(p, epoch, args.epochs) for p in CIRCUIT_GRID], dtype=np.float64)
                if weights.sum() <= 0:
                    weights = np.ones(len(CIRCUIT_GRID), dtype=np.float64)
                weights /= weights.sum()
                p_train = float(rng.choice(CIRCUIT_GRID, p=weights))
            else:
                if epoch < 10:
                    ps, ws = [0.08, 0.05, 0.03, 0.02], [0.5, 0.3, 0.15, 0.05]
                elif epoch < 20:
                    ps, ws = [0.05, 0.03, 0.08, 0.02], [0.5, 0.3, 0.15, 0.05]
                else:
                    ps, ws = [0.03, 0.02, 0.05, 0.08], [0.5, 0.3, 0.15, 0.05]
                p_train = float(rng.choice(ps, p=ws))
            s_bin, labels_x, labels_z = sampler.sample_batch(args.batch_size, p=p_train, teacher=getattr(args,'teacher','mwpm'), rng=rng)
            samples_epoch += int(s_bin.shape[0])
            # Targets: use labels_x (d=3 symmetric)
            y = torch.from_numpy(labels_x.astype(np.float32)).to(device)

            # Build node_inputs from syndrome bits: place into first feature of check nodes
            B = s_bin.shape[0]
            num_check_nodes = 8
            num_qubit_nodes = 9
            nodes_per_graph = num_check_nodes + num_qubit_nodes
            node_inputs = torch.zeros(B, nodes_per_graph, 9, device=device, dtype=torch.float32)
            node_inputs[:, :num_check_nodes, 0] = torch.from_numpy(s_bin.astype(np.float32)).to(device)
            flat_inputs = node_inputs.view(-1, 9)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                outs = model(flat_inputs, model._src_ids, model._dst_ids)  # [n_iters, B*n_nodes, n_node_outputs]
                final = outs[-1]
                # Slice qubit node outputs: last 9 of each graph
                final = final.view(B, nodes_per_graph, -1)[:, num_check_nodes:, :]  # [B,9,2 or 9]
                # Binary head: if 2 logits, condense to bitlogits; if >2 treat last axis as pre-bitlogits
                if final.shape[-1] == 2:
                    bitlogits = (final[..., 1] - final[..., 0])  # [B,9]
                else:
                    # Fallback: assume single logit already
                    bitlogits = final.squeeze(-1)
                # Guard against NaNs/Infs before BCE
                bitlogits = torch.nan_to_num(bitlogits, nan=0.0, posinf=30.0, neginf=-30.0)
                # p-aware label smoothing: y' = (1-s)*y + 0.5*s
                s_base = float(getattr(args, 'smoothing_base', getattr(args, 'label_smoothing', 0.0)))
                highp_thr = float(getattr(args, 'smoothing_highp_threshold', 0.05))
                highp_fac = float(getattr(args, 'smoothing_highp_factor', 0.6))
                s_eff = s_base * (highp_fac if p_train >= highp_thr else 1.0)
                y_eff = y * (1.0 - s_eff) + 0.5 * s_eff
                loss_main = torch.nn.functional.binary_cross_entropy_with_logits(bitlogits, y_eff, reduction='mean')
                # Parity auxiliary loss (differentiable XOR expectation)
                if float(getattr(args, 'parity_lambda', 0.0)) > 0.0:
                    p = bitlogits.sigmoid()
                    q = 1.0 - 2.0 * p  # [B,9]
                    q_exp = q.unsqueeze(1)  # [B,1,9]
                    mz = _mask_z.to(q_exp.device).unsqueeze(0)
                    mx = _mask_x.to(q_exp.device).unsqueeze(0)
                    z_prod = torch.where(mz, q_exp, torch.ones_like(q_exp)).prod(dim=2)  # [B,4]
                    x_prod = torch.where(mx, q_exp, torch.ones_like(q_exp)).prod(dim=2)  # [B,4]
                    z_par = 0.5 * (1.0 - z_prod)
                    x_par = 0.5 * (1.0 - x_prod)
                    sZ_t = torch.from_numpy(s_bin[:, :4].astype(np.float32)).to(device)
                    sX_t = torch.from_numpy(s_bin[:, 4:8].astype(np.float32)).to(device)
                    # Compute BCE for parity outside autocast (FP32) to avoid unsafe autocast path
                    with torch.cuda.amp.autocast(enabled=False):
                        loss_par = torch.nn.functional.binary_cross_entropy(z_par.float(), sZ_t.float()) + \
                                   torch.nn.functional.binary_cross_entropy(x_par.float(), sX_t.float())
                    loss = loss_main + float(getattr(args, 'parity_lambda', 0.0)) * loss_par
                else:
                    loss = loss_main
                # Coset/teacher consistency (small weight): encourage matching teacher representative
                coset_w = float(getattr(args, 'coset_reg', 0.0))
                if coset_w > 0.0:
                    with torch.cuda.amp.autocast(enabled=False):
                        pprob = bitlogits.sigmoid().float()
                        l1 = torch.abs(pprob - y.float()).mean()
                        loss = loss + coset_w * l1

            # Skip update on non-finite loss to avoid poisoning epoch mean
            if not torch.isfinite(loss):
                nan_batches += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if use_ema:
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point and k in ema_state:
                            ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=(1.0 - ema_decay))
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if use_ema:
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point and k in ema_state:
                            ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=(1.0 - ema_decay))
            try:
                lv = float(loss.detach().cpu().item())
                if not (lv == lv):  # NaN check
                    nan_batches += 1
                else:
                    epoch_losses.append(lv)
            except Exception:
                nan_batches += 1

            if _stop_flag["stop"]:
                break

        # Step LR schedules if enabled
        if warmup_sched is not None:
            warmup_sched.step(epoch)
        if base_sched is not None:
            base_sched.step()

        # Validation (small batch for parity/coset)
        with torch.no_grad():
            vN = int(getattr(args, 'val_N', 1024))
            vP = float(getattr(args, 'val_p', 0.05))
            v_s, v_lx, v_lz = sampler.sample_batch(vN, p=vP, teacher=getattr(args,'teacher','mwpm'), rng=rng)
            # If EMA enabled, temporarily evaluate with EMA weights
            _saved = {}
            if use_ema:
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point and k in ema_state:
                        _saved[k] = v.detach().clone()
                        v.data.copy_(ema_state[k])
            # Use model to predict
            B = v_s.shape[0]
            node_inputs = torch.zeros(B, 17, 9, device=device, dtype=torch.float32)
            node_inputs[:, :8, 0] = torch.from_numpy(v_s.astype(np.float32)).to(device)
            final = model(node_inputs.view(-1, 9), model._src_ids, model._dst_ids)[-1]
            final = final.view(B, 17, -1)[:, 8:, :]
            if final.shape[-1] == 2:
                bits = (final[..., 1] - final[..., 0]).sigmoid() > 0.5
            else:
                bits = (final.squeeze(-1).sigmoid() > 0.5)
            y_pred = bits.to(torch.uint8).cpu().numpy()
            succ = _coset_success(Hz, Hx, v_s, y_pred, v_lx)
            # Debug metrics: parity accuracy and teacher parity consistency
            sZ = v_s[:, :Hz.shape[0]]; sX = v_s[:, Hz.shape[0]:Hz.shape[0]+Hx.shape[0]]
            sZ_pred = (Hz @ y_pred.T) % 2; sX_pred = (Hx @ y_pred.T) % 2
            parity_ok = ((sZ_pred.T == sZ) & (sX_pred.T == sX)).all(axis=1)
            par_acc = float(parity_ok.mean())
            sZ_lab = (Hz @ v_lx.T) % 2; sX_lab = (Hx @ v_lx.T) % 2
            teach_par_ok = ((sZ_lab.T == sZ) & (sX_lab.T == sX)).all(axis=1)
            teach_par_acc = float(teach_par_ok.mean())
            val_ler = 1.0 - float(succ.mean())
            if use_ema and _saved:
                for k, v in model.state_dict().items():
                    if k in _saved:
                        v.data.copy_(_saved[k])

        # Compute finite mean only
        train_loss_mean = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else float('nan')
        history.append(dict(epoch=epoch+1, loss=train_loss_mean, val_ler=val_ler, nan_batches=nan_batches))
        nan_note = f" (nan_batches={nan_batches})" if nan_batches else ""
        print(f"[Foundation] epoch {epoch+1}/{args.epochs} loss={train_loss_mean:.4f} val_LER={val_ler:.4f} par_acc={par_acc:.3f} teach_par={teach_par_acc:.3f}{nan_note}")
        # Append metrics row
        try:
            import csv as _csv
            st = sampler.stats_snapshot()
            with open(outdir / 'metrics.csv', 'a', newline='') as f:
                w = _csv.writer(f)
                w.writerow([epoch+1, f"{train_loss_mean:.6f}", f"{val_ler:.6f}", samples_epoch, st.get('mwpf_shots',0), st.get('mwpm_shots',0)])
                try:
                    f.flush(); _os.fsync(f.fileno())
                except Exception:
                    pass
        except Exception:
            pass

        # Save best
        if val_ler < best_val:
            best_val = val_ler
            torch.save(model.state_dict(), ckpt_path)

        # Always update last checkpoint for resume safety
        _save_last_ckpt(epoch+1)

        if _stop_flag["stop"]:
            print("[Foundation] Stopping early due to signal; last checkpoint saved.")
            break

    # Auto-evaluate LER with harness
    ler_json = outdir / f"ler_{args.profile}_foundation.json"
    try:
        cmd = [sys.executable, str(Path('tools')/ 'eval_ler.py'),
               '--decoder','mghd','--checkpoint', str(ckpt_path),
               '--metric','coset','--N-per-p','10000',
               '--p-grid','0.02,0.03,0.05,0.08','--out', str(ler_json)]
        subprocess.run(cmd, cwd=Path(__file__).parent, check=False)
    except Exception:
        pass

    # Handoff summary
    handoff = dict(
        profile=args.profile,
        ckpt=str(ckpt_path),
        best_val_ler=best_val,
        history=history[-5:],
        ler_json=str(ler_json)
    )
    with open(outdir / f"handoff_step11_{args.profile}.json", 'w') as f:
        json.dump(handoff, f, indent=2)
    print(f"Saved checkpoint to {ckpt_path}")
    # Teacher stats snapshot
    try:
        with open(outdir / 'teacher_stats.json', 'w') as f:
            json.dump(sampler.stats_snapshot(), f, indent=2)
    except Exception:
        pass
    return 0

# Back-compat entry for older scripts
def run_step11_garnet_train(args):
    return run_foundation_train(args)

if __name__ == "__main__":
    main()
