import sys
import os
# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import argparse
import torch
import torch.backends
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import json
from datetime import datetime

# Fixed arguments that don't change across trials
FIXED_ARGS = {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "root_path": "./dataset/ETT-small/",
    "data_path": "ETTh2.csv",
    "model": "CIDC",
    "data": "ETTh2",
    "features": "M",
    "seq_len": 96,
    "label_len": 96,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "hnet_attn_rotary_emb_dim": [0, 0],
    "hnet_attn_window_size": [-1, -1],
    "hnet_ssm_chunk_size": 64,
}

# Horizon lengths to evaluate
HORIZON_LENGTHS = [96, 192, 336, 720]

# Global parser (created once)
PARSER = None

def create_parser():
    """Create the argument parser (same as run.py)"""
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh2', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # HNet parameters
    parser.add_argument('--hnet_arch_layout', type=str, default='["m1", ["m1", ["T1"], "m1"], "m1"]',
                        help='JSON list describing hierarchy, e.g., ["m1", ["m1", ["T1"], "m1"], "m1"]')
    parser.add_argument('--hnet_d_model', type=int, nargs='+', default=[32, 64, 128],
                        help='Per-stage model widths (non-decreasing); first must match input embedding dim')
    parser.add_argument('--hnet_d_intermediate', type=int, nargs='+', default=[0, 128, 192],
                        help='Per-stage MLP hidden dims (used by uppercase blocks T/M); 0 disables MLP')
    parser.add_argument('--hnet_moe_loss_weight', type=float, default=0.1, help='weight of ratio / boundary loss')
    parser.add_argument('--hnet_num_experts', type=int, default=16, help='number of experts, aka the compression ratio')
    # SSM (Mamba2) parameters
    parser.add_argument('--hnet_ssm_chunk_size', type=int, default=256,
                        help='Mamba kernel tile length (performance knob)')
    parser.add_argument('--hnet_ssm_d_conv', type=int, default=4,
                        help='Depthwise conv kernel size before SSM')
    parser.add_argument('--hnet_ssm_d_state', type=int, default=32,
                        help='SSM state dimension')
    parser.add_argument('--hnet_ssm_expand', type=int, default=2,
                        help='Channel expansion factor inside Mamba')
    # Attention parameters
    parser.add_argument('--hnet_attn_num_heads', type=int, nargs='+', default=[2, 4, 8],
                        help='Per-stage attention head counts')
    parser.add_argument('--hnet_attn_rotary_emb_dim', type=int, nargs='+', default=[8, 8, 8],
                        help='Per-stage RoPE dimensions (<= per-head dim)')
    parser.add_argument('--hnet_attn_window_size', type=int, nargs='+', default=[-1, -1, -1],
                        help='Per-stage causal window; -1 for full context')

    return parser


def objective(trial):
    """
    Optuna objective function.
    Samples hyperparameters, trains on all horizons, returns average MSE.
    """
    global PARSER
    
    print(f"\n{'#'*80}")
    print(f"# Trial {trial.number + 1}")
    print(f"{'#'*80}\n")
    
    # Sample hyperparameters using Optuna
    hparams = {
        'patch_len': trial.suggest_categorical('patch_len', [4, 8, 16]),
        'hnet_moe_loss_weight': trial.suggest_categorical('hnet_moe_loss_weight', [0.001,0.005,0.01]),
        'learning_rate': trial.suggest_categorical('learning_rate',  [0.0001, 0.0005, 0.001, 0.005]),
        'batch_size': trial.suggest_categorical('batch_size', [32,16,8]),
        'dropout': trial.suggest_categorical('dropout', [0.1, 0.25, 0.3]),
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
        'd_ff': trial.suggest_categorical('d_ff', [256, 512]),
        'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
        'd_layers': trial.suggest_categorical('d_layers', [1, 2, 3, 4, 8]),
        'lradj': trial.suggest_categorical('lradj', ['type1', 'type3', 'cosine']),
    }
    
    # Parse list parameters (stored as strings for categorical)
    # hparams['hnet_d_model'] = [int(x) for x in hparams['hnet_d_model'].split('_')]
    # hparams['hnet_d_intermediate'] = [int(x) for x in hparams['hnet_d_intermediate'].split('_')]
    # hparams['hnet_attn_num_heads'] = [int(x) for x in hparams['hnet_attn_num_heads'].split('_')]
    
    print("Sampled Hyperparameters:")
    for key, val in hparams.items():
        print(f"  {key}: {val}")
    print()
    
    # Evaluate across all horizons
    trial_mse_list = []
    trial_mae_list = []
    
    for horizon_idx, pred_len in enumerate(HORIZON_LENGTHS):
        print(f"\n{'='*80}")
        print(f"Training for horizon length: {pred_len} ({horizon_idx+1}/{len(HORIZON_LENGTHS)})")
        print(f"{'='*80}\n")
        
        # Build command line arguments
        cmd_args = []
        
        # Add fixed args
        for key, val in FIXED_ARGS.items():
            if isinstance(val, list):
                cmd_args.append(f'--{key}')
                cmd_args.extend([str(v) for v in val])
            else:
                cmd_args.extend([f'--{key}', str(val)])
        
        # Add pred_len and model_id
        cmd_args.extend(['--pred_len', str(pred_len)])
        cmd_args.extend(['--model_id', f'ETTh2_96_{pred_len}'])
        
        # Add sampled hyperparameters
        for key, val in hparams.items():
            if isinstance(val, list):
                cmd_args.append(f'--{key}')
                cmd_args.extend([str(v) for v in val])
            else:
                cmd_args.extend([f'--{key}', str(val)])
        
        # Add other necessary args
        cmd_args.extend(['--factor', '3'])
        cmd_args.extend(['--des', 'HPO'])
        cmd_args.extend(['--itr', '1'])
        
        # Parse arguments
        trial_args = PARSER.parse_args(cmd_args)
        
        # Set device
        if torch.cuda.is_available() and trial_args.use_gpu:
            trial_args.device = torch.device('cuda:{}'.format(trial_args.gpu))
        else:
            if hasattr(torch.backends, "mps"):
                trial_args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            else:
                trial_args.device = torch.device("cpu")
        
        if trial_args.use_gpu and trial_args.use_multi_gpu:
            trial_args.devices = trial_args.devices.replace(' ', '')
            device_ids = trial_args.devices.split(',')
            trial_args.device_ids = [int(id_) for id_ in device_ids]
            trial_args.gpu = trial_args.device_ids[0]
        
        # Run experiment
        exp = Exp_Long_Term_Forecast(trial_args)
        
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            trial_args.task_name,
            trial_args.model_id,
            trial_args.model,
            trial_args.data,
            trial_args.features,
            trial_args.seq_len,
            trial_args.label_len,
            trial_args.pred_len,
            trial_args.d_model,
            trial_args.n_heads,
            trial_args.e_layers,
            trial_args.d_layers,
            trial_args.d_ff,
            trial_args.expand,
            trial_args.d_conv,
            trial_args.factor,
            trial_args.embed,
            trial_args.distil,
            trial_args.des, 0)
        
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse, mae, dtw = exp.test_loss_only(setting)
        
        trial_mse_list.append(mse)
        trial_mae_list.append(mae)
        
        print(f"Horizon {pred_len}: MSE = {mse:.4f}, MAE = {mae:.4f}\n")
        
        # Clear cache
        if trial_args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif trial_args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
        
        del exp
        
        # Report intermediate value for pruning
        intermediate_avg_mse = np.mean(trial_mse_list)
        trial.report(intermediate_avg_mse, horizon_idx)
        
        # Check if trial should be pruned
        if trial.should_prune():
            print(f"\n*** Trial {trial.number + 1} pruned at horizon {pred_len} ***\n")
            raise optuna.TrialPruned()
    
    # Calculate average MSE across all horizons
    avg_mse = np.mean(trial_mse_list)
    avg_mae = np.mean(trial_mae_list)
    
    print(f"\n{'='*80}")
    print(f"Trial {trial.number + 1} Summary:")
    print(f"  Average MSE: {avg_mse:.4f}")
    print(f"  Average MAE: {avg_mae:.4f}")
    print(f"  MSE per horizon: {dict(zip(HORIZON_LENGTHS, [f'{m:.4f}' for m in trial_mse_list]))}")
    print(f"{'='*80}\n")
    
    # Store additional info in trial
    trial.set_user_attr('avg_mae', float(avg_mae))
    trial.set_user_attr('mse_per_horizon', {str(h): float(m) for h, m in zip(HORIZON_LENGTHS, trial_mse_list)})
    trial.set_user_attr('mae_per_horizon', {str(h): float(m) for h, m in zip(HORIZON_LENGTHS, trial_mae_list)})
    
    # Print current best so far (if exists)
    try:
        print(f"🏆 CURRENT BEST SO FAR:")
        print(f"   Best Trial: {trial.study.best_trial.number + 1}")
        print(f"   Best MSE: {trial.study.best_value:.4f}")
        print(f"   Best Hyperparameters:")
        for key, val in trial.study.best_params.items():
            print(f"      {key}: {val}")
        print(f"{'='*80}\n")
    except ValueError:
        # No best trial yet (first trial or all failed)
        print(f"🏆 No best trial yet (this was trial {trial.number + 1})")
        print(f"{'='*80}\n")
    
    return avg_mse


def save_study_results(study, output_file='hpo_results_etth2_optuna.json'):
    """Save Optuna study results to JSON"""
    results = {
        'best_trial': study.best_trial.number + 1,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_user_attrs': study.best_trial.user_attrs,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'all_trials': []
    }
    
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_result = {
                'number': trial.number + 1,
                'value': trial.value,
                'params': trial.params,
                'user_attrs': trial.user_attrs,
                'datetime_start': trial.datetime_start.strftime("%Y-%m-%d %H:%M:%S") if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.strftime("%Y-%m-%d %H:%M:%S") if trial.datetime_complete else None,
            }
            results['all_trials'].append(trial_result)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    # Set random seed
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # Create parser
    PARSER = create_parser()
    
    # Parse command line args for HPO settings
    import sys
    n_trials = 200  # Default number of trials
    study_name = "CIDC_etth2"
    
    # Save database in same directory as this script
    hpo_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(hpo_dir, 'hpo_etth2.db')
    storage_name = f"sqlite:///{db_path}"
    
    # Check if user specified number of trials
    if len(sys.argv) > 1 and sys.argv[1].startswith('--n_trials='):
        n_trials = int(sys.argv[1].split('=')[1])
    
    print(f"{'='*80}")
    print(f"Starting Optuna HPO for CIDC on ETTh2")
    print(f"{'='*80}")
    print(f"Number of trials: {n_trials}")
    print(f"Horizons: {HORIZON_LENGTHS}")
    print(f"Storage: {storage_name}")
    print(f"Study name: {study_name}")
    print(f"{'='*80}\n")
    
    # Create Optuna study
    sampler = TPESampler(seed=fix_seed)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=2)  # Prune after 2 horizons if doing badly
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,  # Resume if exists
        direction='minimize',
        sampler=sampler,
        pruner=pruner
    )
    
    # Optimize
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    
    # Print results
    print(f"\n{'#'*80}")
    print(f"# HPO COMPLETE")
    print(f"{'#'*80}\n")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number + 1}")
    print(f"Best average MSE: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")
    print("\nBest metrics per horizon:")
    for key, val in study.best_trial.user_attrs.get('mse_per_horizon', {}).items():
        print(f"  Horizon {key}: MSE = {val:.4f}")
    
    # Save results in same directory as script
    output_path = os.path.join(hpo_dir, 'hpo_results_etth2_optuna.json')
    save_study_results(study, output_path)
    
    # Print top 5 trials
    print(f"\n{'='*80}")
    print("Top 5 Trials:")
    print(f"{'='*80}")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for i, trial in enumerate(sorted_trials[:5], 1):
        if trial.value is not None:
            print(f"{i}. Trial {trial.number + 1}: MSE = {trial.value:.4f}")
    
    print(f"\n{'='*80}")
    print("You can visualize the results using:")
    print(f"  optuna-dashboard {storage_name}")
    print(f"{'='*80}\n")
