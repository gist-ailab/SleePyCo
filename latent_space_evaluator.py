import torch
import os
import json
import argparse
import warnings
from latent_space_evaluation.test_script import evaluate_embeddings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, help='config file path')
    args = parser.parse_args()
    
    with open(args.config) as config_file:
        config = json.load(config_file)
        
    config['name'] = os.path.basename(args.config).replace('.json', '')
    ckpt_path = os.path.join('checkpoints', config['name'])
    embeddings_path = os.path.join(ckpt_path, 'embeddings.pt')
    
    data = torch.load(embeddings_path)
    embeddings = data["embeddings"].detach().cpu().numpy()
    labels = data["labels"].detach().cpu().numpy()
    
    results_path = os.path.join('results', config['name'])
    os.makedirs(results_path, exist_ok=True)
    evaluate_embeddings(embeddings, labels, output_dir=results_path)
    print("[INFO] Evaluation complete. Metrics and plots saved to 'results'.")

if __name__ == "__main__":
    main()