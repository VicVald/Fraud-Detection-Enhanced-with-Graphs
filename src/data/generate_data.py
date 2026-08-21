import argparse
from pathlib import Path
from gen_fraud_graph import Config, FraudGraphGenerator


def generate_synthetic_data(scale_factor: float, hardness: str, output_dir: str):
    print(f"Generating synthetic data (scale={scale_factor}, hardness={hardness}) to {output_dir}...")
    
    # Configure generator with low embedding dimension to avoid slow string conversion of long arrays
    config = Config(
        scale_factor=scale_factor,
        hardness=hardness,
        embedding_provider="fake",  # Fast, no API key or download needed
        embedding_dim=2,
        workers=2,
        output_dir=output_dir,
    )
    
    # Run generator
    generator = FraudGraphGenerator(config)
    generator.run()
    print("Synthetic data generation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic transaction graph data.")
    parser.add_argument("--scale", type=float, default=0.01, help="Scale factor (e.g., 0.01 = ~100k accounts)")
    parser.add_argument("--hardness", type=str, default="low", choices=["low", "medium", "high"], help="Difficulty level of fraud data")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    generate_synthetic_data(args.scale, args.hardness, args.output)
