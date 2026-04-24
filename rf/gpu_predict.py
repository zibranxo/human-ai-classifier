"""
GPU-Optimized Prediction Script
Make predictions using trained GPU ensemble model
"""

import numpy as np
import pickle
import argparse
import torch
import sys
from gpu_ensemble import GPUEnsembleClassifier


def load_pipeline(model_dir='./models_gpu'):
    """Load trained GPU models and pipeline"""
    print("Loading GPU models...")
    
    # Load ensemble
    ensemble_path = f"{model_dir}/gpu_ensemble_model.pkl"
    ensemble = GPUEnsembleClassifier.load(ensemble_path)
    
    # Load pipeline
    pipeline_path = f"{model_dir}/gpu_pipeline.pkl"
    with open(pipeline_path, 'rb') as f:
        pipeline_data = pickle.load(f)
    
    extractor = pipeline_data['extractor']
    scaler = pipeline_data['scaler']
    selected_features = pipeline_data['selected_features']
    feature_indices = pipeline_data['feature_indices']
    device = pipeline_data.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set extractor device
    extractor.device = device
    
    print(f"✓ Models loaded successfully!")
    print(f"✓ Using device: {device.upper()}\n")
    
    return ensemble, extractor, scaler, selected_features, feature_indices


def predict_text(text, ensemble, extractor, scaler, feature_indices, show_features=False):
    """Predict single text"""
    # Extract features
    features = extractor.extract_features([text])
    
    # Select features
    features_selected = features[:, feature_indices]
    
    # Scale
    features_scaled = scaler.transform(features_selected)
    
    # Predict
    prediction = ensemble.predict(features_scaled)[0]
    proba = ensemble.predict_proba(features_scaled)[0]
    confidence = abs(proba[1] - 0.5) * 2  # Distance from 0.5
    
    # Get individual model predictions
    rf_pred = ensemble.rf_model.predict(features_scaled)[0]
    xgb_pred = ensemble.xgb_model.predict(features_scaled)[0]
    
    result = {
        'text': text[:100] + '...' if len(text) > 100 else text,
        'prediction': 'AI-Generated' if prediction == 1 else 'Human-Written',
        'confidence': confidence,
        'ai_probability': proba[1],
        'human_probability': proba[0],
        'rf_prediction': 'AI' if rf_pred == 1 else 'Human',
        'xgb_prediction': 'AI' if xgb_pred == 1 else 'Human',
        'agreement': 'Yes' if rf_pred == xgb_pred else 'No'
    }
    
    if show_features:
        result['features'] = features[0].tolist()
    
    return result


def print_result(result, detailed=False):
    """Pretty print prediction result"""
    print("="*70)
    print("  🔍 PREDICTION RESULT")
    print("="*70)
    print(f"\n📝 Text: {result['text']}")
    print(f"\n🎯 Prediction: {result['prediction']}")
    print(f"💯 Confidence: {result['confidence']*100:.2f}%")
    print(f"\n📊 Probabilities:")
    print(f"  👤 Human: {result['human_probability']*100:.2f}%")
    print(f"  🤖 AI   : {result['ai_probability']*100:.2f}%")
    
    if detailed:
        print(f"\n🔬 Individual Models:")
        print(f"  🌲 PyTorch RF: {result['rf_prediction']}")
        print(f"  ⚡ XGBoost   : {result['xgb_prediction']}")
        print(f"  🤝 Agreement : {result['agreement']}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized AI vs Human Text Prediction')
    parser.add_argument('--model-dir', type=str, default='./models_gpu', help='Directory with trained models')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--file', type=str, help='File with texts (one per line)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--features', action='store_true', help='Show extracted features')
    parser.add_argument('--detailed', action='store_true', help='Show detailed results')
    
    args = parser.parse_args()
    
    # Load models
    ensemble, extractor, scaler, selected_features, feature_indices = load_pipeline(args.model_dir)
    
    # Interactive mode
    if args.interactive:
        print("🎮 Interactive Mode - Enter text to classify (type 'quit' to exit)\n")
        while True:
            print("📝 Enter text: ", end='')
            text = input()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not text.strip():
                print("⚠️  Please enter some text.\n")
                continue
            
            result = predict_text(text, ensemble, extractor, scaler, feature_indices, args.features)
            print_result(result, args.detailed)
    
    # Single text
    elif args.text:
        result = predict_text(args.text, ensemble, extractor, scaler, feature_indices, args.features)
        print_result(result, args.detailed)
    
    # File with multiple texts
    elif args.file:
        print(f"📂 Processing file: {args.file}\n")
        
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = []
        for i, text in enumerate(texts, 1):
            print(f"Processing text {i}/{len(texts)}...", end='\r')
            result = predict_text(text, ensemble, extractor, scaler, feature_indices, args.features)
            results.append(result)
        
        print("\n")
        
        # Summary
        ai_count = sum(1 for r in results if r['prediction'] == 'AI-Generated')
        human_count = len(results) - ai_count
        
        print("="*70)
        print("  📊 BATCH PREDICTION SUMMARY")
        print("="*70)
        print(f"\n📈 Total texts: {len(results)}")
        print(f"🤖 AI-Generated: {ai_count} ({ai_count/len(results)*100:.1f}%)")
        print(f"👤 Human-Written: {human_count} ({human_count/len(results)*100:.1f}%)")
        print(f"\n💯 Average confidence: {np.mean([r['confidence'] for r in results])*100:.2f}%")
        print(f"🤝 Model agreement rate: {sum(1 for r in results if r['agreement'] == 'Yes')/len(results)*100:.1f}%")
        print("="*70 + "\n")
        
        # Individual results
        if args.detailed:
            for i, result in enumerate(results, 1):
                print(f"\n--- Text {i} ---")
                print_result(result, args.detailed)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
