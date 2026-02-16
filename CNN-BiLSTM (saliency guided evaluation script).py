#!/usr/bin/env python3
"""
eval_only.py

Evaluates already-trained CNN-LSTM model on CODE-15 test split and saves 
the exact same outputs the training script would have produced (Excel/CSV 
metrics + saliency PNGs), without any further CNN training. Also fits 
XGBoost head on frozen CNN features for ensemble metrics.

Usage:
    python eval_only.py --data-dir "C:\path\to\data" --meta "C:\path\to\exams.csv" --ckpt "./results/best_model.h5"
"""

import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
import tensorflow as tf
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained CODE-15 model')
    parser.add_argument('--data-dir', required=True, 
                        help='Data directory containing HDF5 files (e.g., C:\\Users\\chaud\\OneDrive\\Desktop\\CODE 15 (data and models)\\Extracted data CODE 15)')
    parser.add_argument('--meta', required=True,
                        help='Path to exams.csv metadata file (e.g., same dir + \\exams.csv)')
    parser.add_argument('--out', default='./results_code15_TRAVELING_FIXES',
                        help='Output directory for results')
    parser.add_argument('--ckpt', required=True,
                        help='Path to saved model checkpoint (.h5) (e.g., ./results_code15_TRAVELING_FIXES/best_model_code15_artifact_aware.h5)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (must be >= 24)')
    parser.add_argument('--xgb-samples', type=int, default=5000,
                        help='Number of samples for XGBoost training (can lower to 3000 for speed)')
    parser.add_argument('--n-saliency', type=int, default=10,
                        help='Number of saliency maps to generate')
    
    args = parser.parse_args()
    
    # Validate batch size
    if args.batch < 24:
        print(f"ERROR: Batch size ({args.batch}) must be >= 24 to avoid imbalance deadlocks")
        sys.exit(1)
    
    print("="*80)
    print("CODE-15 MODEL EVALUATION (EVAL ONLY)")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Metadata: {args.meta}")
    print(f"Output directory: {args.out}")
    print(f"Model checkpoint: {args.ckpt}")
    print(f"Batch size: {args.batch}")
    print(f"XGBoost samples: {args.xgb_samples}")
    print(f"Saliency maps: {args.n_saliency}")
    print("="*80)
    
    # Dynamic import of the base script
    print("Dynamically importing base script...")
    try:
        # Try different possible paths for the base script
        possible_paths = [
            r"code15_traveling_fixes(11).py",  # Current directory
            os.path.abspath(r"code15_traveling_fixes(11).py"),  # Absolute path in current dir
        ]
        
        base_script_path = None
        for path in possible_paths:
            if os.path.exists(path):
                base_script_path = path
                break
                
        if base_script_path is None:
            print(f"ERROR: Base script not found. Tried:")
            for path in possible_paths:
                print(f"  - {path}")
            print("Please ensure code15_traveling_fixes(11).py is in the current directory")
            print("Or update the path in the script")
            sys.exit(1)
            
        mod = SourceFileLoader("ctf11", base_script_path).load_module()
        CODE15AgePredictorFinal = mod.CODE15AgePredictorFinal
        CNNLSTMModelFinal = mod.CNNLSTMModelFinal
        print(f"Successfully imported base script from: {base_script_path}")
    except Exception as e:
        print(f"ERROR: Failed to import base script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verify checkpoint exists
    if not os.path.exists(args.ckpt):
        print(f"ERROR: Model checkpoint not found: {args.ckpt}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Build dataset & splits (no training)
    print("\nBuilding dataset and preparing splits...")
    pred = CODE15AgePredictorFinal(args.data_dir, args.meta, args.out)
    train_df, val_df, test_df = pred.dataset.prepare_stratified_patient_splits()
    
    print(f"Train: {len(train_df)} records")
    print(f"Val: {len(val_df)} records") 
    print(f"Test: {len(test_df)} records")
    
    # Compute train normalization stats
    print("\nComputing normalization statistics from train data...")
    _ = pred.dataset.compute_train_normalization_stats(train_df, sample_size=1000)
    print(f"Train stats computed: mean range [{pred.dataset.train_stats['mean'].min():.4f}, {pred.dataset.train_stats['mean'].max():.4f}]")
    
    # Compute age weights from train data  
    print("Computing age weights from train data...")
    pred.dataset.compute_age_weights_from_train(train_df)
    print(f"Age weights computed for {len(pred.dataset.age_weight_by_exam_id)} train exam_ids")
    
    # Build model skeleton, then load saved weights
    print(f"\nBuilding model skeleton...")
    pred.cnn_lstm = CNNLSTMModelFinal(median_age=pred.dataset.train_median_age)
    pred.cnn_lstm.build_model(use_saliency_guidance=False)
    
    print(f"Loading saved model weights from: {args.ckpt}")
    try:
        # Load full saved model and rebuild the feature extractor
        loaded = tf.keras.models.load_model(args.ckpt, compile=False)
        pred.cnn_lstm.model = loaded
        
        # Rebuild feature extractor
        from tensorflow.keras import layers, models
        gap = None
        for layer in loaded.layers:
            if isinstance(layer, layers.GlobalAveragePooling1D):
                gap = layer
                break
        
        if gap is not None:
            pred.cnn_lstm.feature_extractor = models.Model(
                inputs=loaded.input, 
                outputs=gap.output
            )
            print("Feature extractor rebuilt successfully")
        else:
            print("WARNING: GlobalAveragePooling1D layer not found, feature extractor not created")
            # Create a basic feature extractor if none found
            try:
                # Find the last dense layer before outputs
                for layer in reversed(loaded.layers):
                    if hasattr(layer, 'output_shape') and len(layer.output_shape) == 2:
                        pred.cnn_lstm.feature_extractor = models.Model(
                            inputs=loaded.input, 
                            outputs=layer.output
                        )
                        print("Fallback feature extractor created")
                        break
            except:
                print("ERROR: Could not create feature extractor")
                sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create test dataset for standalone eval
    print(f"\nCreating test dataset for evaluation...")
    test_gen, n_valid_test = pred.dataset.create_weighted_data_generator(
        test_df, batch_size=args.batch, shuffle=False, is_train=False
    )
    
    test_ds = tf.data.Dataset.from_generator(
        test_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 1000, 12), dtype=tf.float32),
            {'age_output': tf.TensorSpec(shape=(None,), dtype=tf.float32),
             'bin_output': tf.TensorSpec(shape=(None,), dtype=tf.int32)},
            {'age_output': tf.TensorSpec(shape=(None,), dtype=tf.float32),
             'bin_output': tf.TensorSpec(shape=(None,), dtype=tf.float32)}
        )
    ).cache().prefetch(tf.data.AUTOTUNE)
    
    steps = max(1, n_valid_test // args.batch)
    print(f"Test dataset: {n_valid_test} samples, {steps} steps")
    
    # Evaluate standalone CNN-LSTM
    print("\n" + "="*40)
    print("STANDALONE CNN-LSTM EVALUATION")
    print("="*40)
    standalone = pred.cnn_lstm.evaluate_standalone(test_ds, steps=steps)
    print("Standalone results:")
    for key, value in standalone.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Fit XGBoost on frozen CNN features (fast)
    print("\n" + "="*40)
    print("XGBOOST TRAINING ON FROZEN FEATURES")
    print("="*40)
    
    print("Collecting training data for XGBoost...")
    Xtr, ytr = pred.collect_data_for_xgboost(
        train_df, max_samples=args.xgb_samples, saliency_mask=None
    )
    
    print("Collecting validation data for XGBoost...")
    Xva, yva = pred.collect_data_for_xgboost(
        val_df, max_samples=max(1000, args.xgb_samples//2), saliency_mask=None
    )
    
    print("Training XGBoost...")
    pred.cnn_lstm.train_xgboost(Xtr, ytr, Xva, yva, pred.output_dir)
    
    # Save XGBoost model and preprocessing stats for deployment
    print("Saving XGBoost model and preprocessing stats for deployment...")
    
    # 1) Save the XGBoost model
    xgb_model = pred.cnn_lstm.xgb_model  # this is your fitted xgboost.sklearn.XGBRegressor
    booster = xgb_model.get_booster()
    booster.save_model(os.path.join(args.out, "xgb_code15.json"))
    
    # 2) Save preprocessing stats and feature-extractor info
    np.savez(
        os.path.join(args.out, "preprocessing_stats_code15.npz"),
        mean=pred.dataset.train_stats["mean"],
        std=pred.dataset.train_stats["std"],
        sampling_rate=100,           # your model expects 100 Hz, 10 s → (1000, 12)
        input_length=1000,
        n_leads=12
    )
    
    # 3) Save ensemble metadata
    with open(os.path.join(args.out, "ensemble_meta.json"), "w") as f:
        import json
        json.dump(
            {
                "feature_layer": "GlobalAveragePooling1D",  # or penultimate layer if used
                "ensemble_rule": "0.6*age_cnn + 0.4*age_xgb",
                "cnn_checkpoint": os.path.abspath(args.ckpt),
                "xgb_model": "xgb_code15.json",
                "preproc_stats": "preprocessing_stats_code15.npz"
            },
            f, indent=2
        )
    
    print("Saved XGB model and preprocessing stats for deployment:")
    print(f"  - {os.path.join(args.out, 'xgb_code15.json')}")
    print(f"  - {os.path.join(args.out, 'preprocessing_stats_code15.npz')}")
    print(f"  - {os.path.join(args.out, 'ensemble_meta.json')}")
    
    # Modify the predictor to use custom saliency count
    original_generate_saliency_maps = pred.generate_saliency_maps
    
    def custom_generate_saliency_maps(X_sample, y_sample, n_samples=None):
        if n_samples is None:
            n_samples = args.n_saliency
        return original_generate_saliency_maps(X_sample, y_sample, n_samples)
    
    pred.generate_saliency_maps = custom_generate_saliency_maps
    
    # Run comprehensive eval + write outputs
    print("\n" + "="*40) 
    print("COMPREHENSIVE EVALUATION")
    print("="*40)
    
    print("Running comprehensive evaluation (this generates all outputs)...")
    print(f"  - Computing Ensemble, CNN-LSTM, and XGB metrics")
    print(f"  - Generating {args.n_saliency} saliency PNGs")
    print(f"  - Writing Excel (or CSV fallbacks + per-model decile CSVs)")
    
    results = pred.evaluate_comprehensive(test_df, batch_size=args.batch)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  MAE: {metrics['overall_mae']:.3f} years (primary metric)")
        print(f"  RMSE: {metrics['overall_rmse']:.3f} years")
        print(f"  R²: {metrics['overall_r2']:.3f}")
        print(f"  PSNR: {metrics['overall_psnr']:.1f} dB")
        print(f"  Pearson R: {metrics['pearson_r']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        print(f"  Samples: {metrics['n_samples']:,}")
    
    print(f"\n All results saved to: {args.out}")
    print("Expected files:")
    print("  - Excel file (or CSV fallbacks + per-model decile CSVs)")
    print(f"  - saliency_map_*.png (up to {args.n_saliency} maps)")
    print("  - xgb_code15.json (XGBoost model for deployment)")
    print("  - preprocessing_stats_code15.npz (normalization stats)")
    print("  - ensemble_meta.json (deployment metadata)")
    
    # List actual files created
    print(f"\nActual files created in {args.out}:")
    try:
        files_found = []
        for file in sorted(os.listdir(args.out)):
            file_path = os.path.join(args.out, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024*1024)
                files_found.append(f"  - {file} ({size_mb:.2f} MB)")
        
        if files_found:
            for file_info in files_found:
                print(file_info)
        else:
            print("  No files found in output directory")
            
    except Exception as e:
        print(f"  Could not list files: {e}")
    
    # Quality-of-life tips
    print(f"\n" + "="*60)
    print("QUALITY-OF-LIFE TIPS:")
    print("="*60)
    if not any("xlsx" in f for f in os.listdir(args.out)):
        print("TIP: Install xlsxwriter for single Excel file output:")
        print("pip install xlsxwriter")
    
    print(f"TIP: Batch size set to {args.batch} (>= 24 to avoid imbalance deadlocks)")
    print(f"TIP: XGBoost trained on {args.xgb_samples} samples (adjustable with --xgb-samples)")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("Standalone CNN-LSTM metrics computed")
    print("XGBoost fitted on frozen CNN features") 
    print("Ensemble metrics computed and saved")
    print("Saliency maps generated")
    print("Excel/CSV outputs written")
    print("XGBoost model saved for deployment (xgb_code15.json)")
    print("Preprocessing stats saved (preprocessing_stats_code15.npz)")
    print("Ensemble metadata saved (ensemble_meta.json)")
    print(f"All outputs saved to: {args.out}")


if __name__ == "__main__":
    main()