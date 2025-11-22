import numpy as np
import matplotlib.pyplot as plt
import os


def plot_multi_model_predictions(settings, plot_interval=20, batch_size=32, output_dir='./multi_model_plots/'):
    """
    Plot predictions from multiple models on the same plot.
    
    Args:
        settings: List of setting strings (model IDs from results folder)
        plot_interval: Plot every Nth sample (default: 20, same as in test)
        output_dir: Directory to save the plots
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load predictions and ground truth for all models
    model_data = {}
    
    print("Loading predictions from models...")
    for setting in settings:
        pred_path = f'./results/{setting}/pred.npy'
        true_path = f'./results/{setting}/true.npy'
        
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction file not found for {setting}")
            continue
        
        preds = np.load(pred_path)
        trues = np.load(true_path)
        
        # Extract model name from setting (format: ..._{model}_...)
        parts = setting.split('_')
        # Find the model name (it's after the data name)
        # Format: long_term_forecast_{dataset}_{seq}_{pred}_{MODEL}_...
        model_name = parts[6]  # This should be the model name
        
        model_data[setting] = {
            'preds': preds,
            'trues': trues,
            'model_name': model_name
        }
        print(f"  Loaded {model_name}: predictions shape {preds.shape}, true shape {trues.shape}")
    
    if len(model_data) == 0:
        print("No valid predictions found!")
        return
    
    # Get the ground truth from the first model (should be same for all)
    first_setting = list(model_data.keys())[0]
    trues = model_data[first_setting]['trues']
    
    print(f"\nTotal samples: {trues.shape[0]}")
    print(f"Prediction length: {trues.shape[1]}")
    print(f"Number of features: {trues.shape[2]}")
    
    # Generate plots for samples at plot_interval
    print(f"\nGenerating plots for samples at interval {plot_interval * batch_size}...")
    
    for i in range(0, trues.shape[0], plot_interval * batch_size):
        # Get ground truth for this sample (using last feature -1)
        gt = trues[i, :, -1]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot ground truth
        plt.plot(gt, label='GroundTruth', linewidth=2, color='black', linestyle='-')
        
        # Plot predictions from each model
        # Use discrete color indices for maximum distinction
        colors = [plt.cm.tab10(i % 10) for i in range(len(model_data))]
        
        
        for (setting, data), color in zip(model_data.items(), colors):
            preds = data['preds']
            model_name = data['model_name']
            
            # Get prediction for this sample
            pred = preds[i, :, -1]
            
            plt.plot(pred, label=f'{model_name}', linewidth=2, color=color, alpha=0.8)
        
        plt.legend(loc='best')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Sample {i // batch_size}')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{i // batch_size}.pdf')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot for sample {i // batch_size}")
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == '__main__':
    # Example usage
    settings = [
        'long_term_forecast_ETTh1_96_720_TimeMixer_ETTh1_ftM_sl96_ll0_pl720_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0',
        # 'long_term_forecast_ETTh1_96_96_Fuse_DC_Transformer_ETTh1_ftM_sl96_ll48_pl96_dm512_nh2_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0',
        'long_term_forecast_ETTh1_96_720_CIDC_ETTh1_ftM_sl96_ll96_pl720_dm256_nh8_el1_dl4_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0',
        # 'long_term_forecast_ETTh1_96_96_HNet_ETTh1_ftM_sl96_ll96_pl96_dm512_nh8_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0',
        # 'long_term_forecast_ETTh1_96_96_TransformerDecoder_ETTh1_ftM_sl96_ll96_pl96_dm512_nh8_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0'
    ]
    batch_size = 32

    
    plot_multi_model_predictions(settings, plot_interval=20, batch_size=batch_size)
