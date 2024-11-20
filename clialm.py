import click
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import time
from datetime import datetime
from adaptive_predictor import AdaptivePredictor
import requests
import json

console = Console()

class PredictorCLI:
    def __init__(self):
        self.predictor = AdaptivePredictor()
        self.base_url = "https://allora-api.testnet.allora.network/emissions/v5"
    
    def get_network_inference(self, topic_id: int) -> float:
        """Fetch network inference from Allora API"""
        url = f"{self.base_url}/latest_network_inferences/{topic_id}"
        try:
            response = requests.get(url)
            data = response.json()
            return float(data["network_inferences"]["combined_value"])
        except Exception as e:
            console.print(f"[red]Error fetching network inference: {e}[/red]")
            return None

    def prepare_market_data(self) -> torch.Tensor:
        """Prepare market data for model input"""
        # Implement your market data collection logic here
        # This should return a tensor of the correct shape for your model
        pass

@click.group()
def cli():
    """Adaptive Predictor CLI"""
    pass

@cli.command()
@click.option('--topic-id', default=1, help='Topic ID (1 for ETH 10min)')
@click.option('--interval', default=300, help='Update interval in seconds')
def train(topic_id: int, interval: int):
    """Train the adaptive predictor using network consensus"""
    predictor_cli = PredictorCLI()
    console.print("[green]Starting adaptive training...[/green]")
    
    try:
        while True:
            # Get network inference
            network_inference = predictor_cli.get_network_inference(topic_id)
            if network_inference is None:
                time.sleep(10)
                continue
                
            # Prepare market data
            market_data = predictor_cli.prepare_market_data()
            if market_data is None:
                continue
                
            # Get local prediction
            inference = predictor_cli.predictor.get_inference(market_data)
            
            # Adapt to network
            predictor_cli.predictor.adapt_to_network(
                network_inference, 
                inference['prediction']
            )
            
            # Display status
            console.print(Panel(
                f"[cyan]Network Inference:[/cyan] {network_inference:.4f}\n"
                f"[green]Local Prediction:[/green] {inference['prediction']:.4f}\n"
                f"[yellow]Confidence:[/yellow] {inference['confidence']:.2f}",
                title=f"Training Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ))
            
            # Save model periodically
            predictor_cli.predictor.save_model()
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        console.print("[yellow]Stopping training...[/yellow]")
        predictor_cli.predictor.save_model()

@cli.command()
@click.option('--topic-id', default=1, help='Topic ID (1 for ETH 10min)')
def predict(topic_id: int):
    """Get current prediction from the model"""
    predictor_cli = PredictorCLI()
    
    # Load latest model state
    predictor_cli.predictor.load_model()
    
    # Get market data
    market_data = predictor_cli.prepare_market_data()
    if market_data is None:
        console.print("[red]Failed to get market data[/red]")
        return
        
    # Get prediction
    inference = predictor_cli.predictor.get_inference(market_data)
    
    # Get network inference for comparison
    network_inference = predictor_cli.get_network_inference(topic_id)
    
    # Display results
    table = Table(title="Prediction Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model Prediction", f"{inference['prediction']:.4f}")
    table.add_row("Model Confidence", f"{inference['confidence']:.2f}")
    if network_inference:
        table.add_row("Network Inference", f"{network_inference:.4f}")
        table.add_row("Deviation", f"{abs(inference['prediction'] - network_inference):.4f}")
    
    console.print(table)

@cli.command()
def status():
    """Show current model status and performance metrics"""
    predictor_cli = PredictorCLI()
    predictor_cli.predictor.load_model()
    
    metrics = predictor_cli.predictor.get_prediction_metrics()
    
    table = Table(title="Model Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in metrics.items():
        if isinstance(value, list):
            table.add_row(key, ", ".join(f"{v:.4f}" for v in value))
        elif isinstance(value, (int, float)):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)

if __name__ == '__main__':
    cli()