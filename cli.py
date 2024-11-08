import click
import requests
import json
from typing import Dict, List, Any
from enum import Enum
import os
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import tempfile
import webbrowser

console = Console()

ALLORA_API_BASE = "https://allora-api.testnet.allora.network/emissions/v5"

CRYPTO_TIMEFRAMES = {
    "ETH": ["5min", "10min", "20min", "24h"],
    "BTC": ["5min", "10min", "24h"],
    "SOL": ["10min", "24h"],
    "BNB": ["20min"],
    "ARB": ["20min"]
}

TOPIC_MAPPING = {
    ("ETH", "5min"): 13,
    ("ETH", "10min"): 1,
    ("ETH", "20min"): 7,
    ("ETH", "24h"): 2,
    ("BTC", "5min"): 14,
    ("BTC", "10min"): 3,
    ("BTC", "24h"): 4,
    ("SOL", "10min"): 5,
    ("SOL", "24h"): 6,
    ("BNB", "20min"): 8,
    ("ARB", "20min"): 9
}

class Category(Enum):
    CRYPTO_PRICE = "crypto_price"
    CTR = "ctr"
    VIEWABILITY = "viewability"
    BID_PRICE = "bid_price"

class CLI:
    def __init__(self):
        self.api_url = os.getenv("INFERENCE_API_ADDRESS", "http://localhost:8000")
        self.console = Console()

    def display_cryptocurrencies(self) -> str:
        """Display available cryptocurrencies and let user select one."""
        table = Table(title="Available Cryptocurrencies")
        table.add_column("Number", justify="right", style="cyan")
        table.add_column("Cryptocurrency", style="magenta")
        table.add_column("Available Timeframes", style="green")

        cryptos = list(CRYPTO_TIMEFRAMES.keys())
        for idx, crypto in enumerate(cryptos, 1):
            table.add_row(
                str(idx),
                crypto,
                ", ".join(CRYPTO_TIMEFRAMES[crypto])
            )

        console.print(table)
        
        choice = Prompt.ask(
            "Select cryptocurrency number",
            choices=[str(i) for i in range(1, len(cryptos) + 1)]
        )
        return cryptos[int(choice) - 1]
            
    def display_timeframes(self, crypto: str) -> str:
        """Display available timeframes for selected cryptocurrency."""
        table = Table(title=f"Available Timeframes for {crypto}")
        table.add_column("Number", justify="right", style="cyan")
        table.add_column("Timeframe", style="magenta")

        timeframes = CRYPTO_TIMEFRAMES[crypto]
        for idx, timeframe in enumerate(timeframes, 1):
            table.add_row(str(idx), timeframe)

        console.print(table)
        
        choice = Prompt.ask(
            "Select timeframe number",
            choices=[str(i) for i in range(1, len(timeframes) + 1)]
        )
        return timeframes[int(choice) - 1]

    def get_target_variable(self, category: Category) -> str:
        examples = {
            Category.CRYPTO_PRICE: "ETH, BTC, SOL",
            Category.CTR: "campaign_123",
            Category.VIEWABILITY: "publisher_abc",
            Category.BID_PRICE: "placement_xyz"
        }
        
        console.print(Panel(f"Example targets for {category.value}: {examples[category]}"))
        return Prompt.ask("Enter target variable")

    def plot_confidence_intervals(self, confidence_intervals: List[float], percentiles: List[str]):
        """Create an interactive visualization of confidence intervals."""
        # Create the violin plot
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=("Confidence Interval Distribution", "Price Range"))

        # Add violin plot
        fig.add_trace(
            go.Violin(
                y=confidence_intervals,
                box_visible=True,
                line_color='blue',
                meanline_visible=True,
                fillcolor='lightblue',
                opacity=0.6,
                name="Distribution"
            ),
            row=1, col=1
        )

        # Add range plot with error bars
        median_idx = len(confidence_intervals) // 2
        median = confidence_intervals[median_idx]
        
        fig.add_trace(
            go.Scatter(
                x=['Price Range'],
                y=[median],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[confidence_intervals[-1] - median],
                    arrayminus=[median - confidence_intervals[0]],
                    color='red'
                ),
                mode='markers',
                marker=dict(size=10, color='blue'),
                name="Price Range"
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title="Market Confidence Intervals Analysis",
            showlegend=False,
            height=800,
            template="plotly_white"
        )

        # Add percentile annotations
        for i, (percentile, value) in enumerate(zip(percentiles, confidence_intervals)):
            fig.add_annotation(
                x=0,
                y=value,
                text=f"{percentile} ({value:.2f})",
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=2,
                row=1, col=1
            )

        # Create temporary HTML file and open in browser
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
            fig.write_html(f.name)
            webbrowser.open(f'file://{f.name}')

    def plot_market_analysis(self, data: Dict, confidence_intervals: List[float], percentiles: List[str]):
        """Create an enhanced market analysis visualization."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Price Distribution Analysis",
                "Statistical Price Range (±3σ)",
                "Top Market Contributors"
            ),
            vertical_spacing=0.2,
            row_heights=[0.4, 0.3, 0.3]
        )

        # 1. Improved Distribution Plot
        fig.add_trace(
            go.Violin(
                y=confidence_intervals,
                box_visible=True,
                line_color='rgba(41, 128, 185, 0.8)',
                fillcolor='rgba(52, 152, 219, 0.3)',
                meanline_visible=True,
                name="Distribution",
                showlegend=False,
                points=False,  # Hide individual points
                width=0.8,    # Adjust width of violin plot
                meanline=dict(
                    color="rgba(41, 128, 185, 1.0)",
                    width=2
                ),
                box=dict(
                    line=dict(
                        color="rgba(41, 128, 185, 1.0)",
                        width=2
                    )
                )
            ),
            row=1, col=1
        )

        # Add cleaner percentile annotations on both sides
        for i, (percentile, value) in enumerate(zip(percentiles, confidence_intervals)):
            # Left side annotation
            fig.add_annotation(
                x=-0.3,
                y=value,
                text=f"${value:.2f}",
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='rgba(41, 128, 185, 0.6)',
                font=dict(size=10, color='rgba(41, 128, 185, 1.0)'),
                align='right',
                row=1, col=1
            )
            
            # Right side annotation
            fig.add_annotation(
                x=0.3,
                y=value,
                text=f"{percentile}",
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='rgba(41, 128, 185, 0.6)',
                font=dict(size=10, color='rgba(41, 128, 185, 1.0)'),
                align='left',
                row=1, col=1
            )

        # Update layout
        fig.update_layout(
            title={
                'text': "Cryptocurrency Price Analysis Dashboard",
                'y':0.98,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24, color='rgba(41, 128, 185, 1.0)')
            },
            showlegend=False,
            height=1200,
            template="plotly_white",
            margin=dict(t=100, b=50, l=120, r=120),  # Increased margins for labels
            paper_bgcolor='white',
            plot_bgcolor='rgba(240, 247, 255, 0.5)'
        )

        # Update axes
        fig.update_yaxes(
            title_text="Price (USD)",
            title_font=dict(size=12),
            gridcolor='rgba(189, 195, 199, 0.4)',
            row=1, col=1
        )

        fig.update_xaxes(
            showticklabels=False,  # Hide x-axis labels
            showgrid=False,        # Hide x-axis grid
            zeroline=False,        # Hide zero line
            row=1, col=1
        )

        # 2. Improved Standard Deviation Plot
        median = confidence_intervals[2]
        std = (confidence_intervals[3] - confidence_intervals[1]) / 2

        x_range = np.linspace(median - 3.5*std, median + 3.5*std, 100)
        y_range = np.exp(-0.5 * ((x_range - median) / std) ** 2) / (std * np.sqrt(2 * np.pi))

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode='lines',
                line=dict(color='rgba(41, 128, 185, 0.8)', width=3),
                fill='tonexty',
                fillcolor='rgba(52, 152, 219, 0.1)',
                name='Normal Distribution',
                showlegend=False
            ),
            row=2, col=1
        )

        # Add styled standard deviation markers
        for i, sigma in enumerate([1, 2, 3]):
            color = f'rgba(41, 128, 185, {0.8 - i*0.2})'  # Decreasing opacity
            fig.add_vline(
                x=median + sigma*std,
                line_dash="dash",
                line_color=color,
                line_width=2,
                row=2, col=1
            )
            fig.add_vline(
                x=median - sigma*std,
                line_dash="dash",
                line_color=color,
                line_width=2,
                row=2, col=1
            )
            fig.add_annotation(
                x=median + sigma*std,
                y=max(y_range) * 1.1,
                text=f"+{sigma}σ",
                font=dict(size=12, color='rgba(41, 128, 185, 0.8)'),
                showarrow=False,
                row=2, col=1
            )
            fig.add_annotation(
                x=median - sigma*std,
                y=max(y_range) * 1.1,
                text=f"-{sigma}σ",
                font=dict(size=12, color='rgba(41, 128, 185, 0.8)'),
                showarrow=False,
                row=2, col=1
            )

        # 3. Enhanced Top Workers Plot
        workers_data = self._process_workers_data(data)
        top_workers = workers_data[:5]

        fig.add_trace(
            go.Bar(
                x=[w["worker"][-8:] + "..." for w in top_workers],
                y=[w["weight"] for w in top_workers],
                text=[f'Value: {w["value"]:.2f}' for w in top_workers],
                textposition='auto',
                marker=dict(
                    color='rgba(41, 128, 185, 0.8)',
                    line=dict(color='rgba(41, 128, 185, 1.0)', width=1)
                ),
                name="Worker Weights"
            ),
            row=3, col=1
        )

        # Add styled percentile annotations
        for i, (percentile, value) in enumerate(zip(percentiles, confidence_intervals)):
            fig.add_annotation(
                x=0.1,
                y=value,
                text=f"{percentile} ({value:.2f})",
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=2,
                arrowcolor='rgba(41, 128, 185, 0.8)',
                font=dict(size=11),
                row=1, col=1
            )

        # Update axes styling
        fig.update_xaxes(
            title_text="Price Range",
            title_font=dict(size=14),
            gridcolor='rgba(189, 195, 199, 0.4)',
            row=2, col=1
        )
        fig.update_xaxes(
            title_text="Contributors",
            title_font=dict(size=14),
            gridcolor='rgba(189, 195, 199, 0.4)',
            row=3, col=1
        )
        fig.update_yaxes(
            title_text="Weight",
            title_font=dict(size=14),
            gridcolor='rgba(189, 195, 199, 0.4)',
            row=3, col=1
        )

        # Create temporary HTML file and open in browser
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
            fig.write_html(f.name)
            webbrowser.open(f'file://{f.name}')

    def _process_workers_data(self, data: Dict) -> List[Dict]:
        """Helper method to process workers data."""
        workers_data = []
        network_inferences = data["network_inferences"]
        
        for weight_data in data["inferer_weights"]:
            worker = weight_data["worker"]
            weight = float(weight_data["weight"])
            value = next(
                (float(v["value"]) for v in network_inferences["inferer_values"] 
                 if v["worker"] == worker),
                None
            )
            if value:
                workers_data.append({"worker": worker, "weight": weight, "value": value})
        
        return sorted(workers_data, key=lambda x: x["weight"], reverse=True)

    def get_market_insights(self, topic_id: int) -> Dict:
        """Fetch market insights from Allora API."""
        url = f"{ALLORA_API_BASE}/latest_network_inferences/{topic_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Debug print
            console.print("[cyan]Received data structure:[/cyan]")
            console.print(json.dumps(data, indent=2))
            
            # Extract relevant metrics with safety checks
            network_inferences = data.get("network_inferences", {})
            if not network_inferences:
                console.print("[red]No network inferences found in response[/red]")
                return None
                
            combined_value = float(network_inferences.get("combined_value", 0))
            confidence_intervals = data.get("confidence_interval_values", [])
            
            if not confidence_intervals:
                console.print("[red]No confidence intervals found in response[/red]")
                return None
                
            # Convert confidence intervals to float with safety check
            confidence_intervals = [float(x) for x in confidence_intervals if x is not None]
            
            if len(confidence_intervals) < 5:
                console.print(f"[yellow]Warning: Expected 5 confidence intervals, got {len(confidence_intervals)}[/yellow]")
                # Pad with zeros if necessary
                confidence_intervals.extend([0.0] * (5 - len(confidence_intervals)))
                
            percentiles = ["2.28%", "15.87%", "50%", "84.13%", "97.72%"]
            
            # Display tables
            self.display_market_tables(data)
            
            # Generate and display interactive plot
            console.print("\nGenerating interactive visualization...")
            self.plot_market_analysis(data, confidence_intervals, percentiles)
            
            return data
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error making request: {str(e)}[/red]")
            return None
        except (KeyError, ValueError, IndexError) as e:
            console.print(f"[red]Error processing data: {str(e)}[/red]")
            console.print("[yellow]Raw response data:[/yellow]")
            console.print(json.dumps(data, indent=2))
            return None
        except Exception as e:
            console.print(f"[red]Unexpected error: {str(e)}[/red]")
            return None

    def submit_inference(self, category: Category, target_variable: str):
        url = f"{self.api_url}/inference"
        data = {
            "category": category.value,
            "target_variable": target_variable
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            result = response.json()
            
            console.print(Panel(f"Inference submitted successfully!", style="green"))
            console.print(json.dumps(result, indent=2))
        except Exception as e:
            console.print(f"[red]Error submitting inference: {str(e)}[/red]")

    def submit_ground_truth(self, category: Category, target_variable: str):
        value = Prompt.ask("Enter ground truth value", default="0.0")
        
        url = f"{self.api_url}/ground-truth"
        data = {
            "category": category.value,
            "target_variable": target_variable,
            "value": float(value)
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            console.print(Panel("Ground truth submitted successfully!", style="green"))
        except Exception as e:
            console.print(f"[red]Error submitting ground truth: {str(e)}[/red]")

    def display_market_tables(self, data: Dict):
        """Display market data in formatted tables."""
        # Network Statistics Table
        stats_table = Table(title="Network Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        network_inferences = data["network_inferences"]
        stats_table.add_row("Combined Value", f"{float(network_inferences['combined_value']):.2f}")
        stats_table.add_row("Naive Value", f"{float(network_inferences['naive_value']):.2f}")
        stats_table.add_row("Block Height", data['inference_block_height'])  # Changed this line
        
        console.print(stats_table)
        console.print()

        # Top Workers Table
        workers_table = Table(title="Top Workers by Weight")
        workers_table.add_column("Worker", style="cyan")
        workers_table.add_column("Weight", style="magenta")
        workers_table.add_column("Value", style="green")
        workers_table.add_column("Deviation", style="yellow")
        
        # Combine and sort workers by weight
        workers_data = []
        combined_value = float(network_inferences['combined_value'])
        
        # Process inferer weights
        for weight_data in data["inferer_weights"]:
            worker = weight_data["worker"]
            weight = float(weight_data["weight"])
            value = next(
                (float(v["value"]) for v in network_inferences["inferer_values"] 
                 if v["worker"] == worker),
                None
            )
            if value:
                deviation = ((value - combined_value) / combined_value) * 100
                workers_data.append({
                    "worker": worker,
                    "weight": weight,
                    "value": value,
                    "deviation": deviation
                })
        
        # Sort by weight and display top 10
        workers_data.sort(key=lambda x: x["weight"], reverse=True)
        for worker in workers_data[:10]:
            workers_table.add_row(
                worker["worker"][-8:] + "...",  # Show last 8 chars of address
                f"{worker['weight']:.4f}",
                f"{worker['value']:.2f}",
                f"{worker['deviation']:+.2f}%"
            )
        
        console.print(workers_table)
        console.print()

        # Confidence Intervals Table
        ci_table = Table(title="Confidence Intervals")
        ci_table.add_column("Percentile", style="cyan")
        ci_table.add_column("Value", style="green")
        ci_table.add_column("Range", style="yellow")
        
        percentiles = data["confidence_interval_raw_percentiles"]
        values = data["confidence_interval_values"]
        
        median = float(values[2])  # 50th percentile
        for p, v in zip(percentiles, values):
            value = float(v)
            range_pct = ((value - median) / median) * 100
            ci_table.add_row(
                f"{p}%", 
                f"{value:.2f}",
                f"{range_pct:+.2f}%"
            )
        
        console.print(ci_table)

        # Summary Panel
        summary = Panel(
            f"[bold]Market Summary[/bold]\n"
            f"Current Value: ${combined_value:.2f}\n"
            f"Confidence Range: ${float(values[0]):.2f} to ${float(values[-1]):.2f}\n"
            f"Total Active Workers: {len(workers_data)}",
            title="Summary",
            style="green"
        )
        console.print(summary)

@click.group()
def cli():
    """Allora Data Provider CLI"""
    pass

@cli.command()
def configure():
    """Configure API endpoint and other settings"""
    api_url = Prompt.ask(
        "Enter API endpoint",
        default="http://localhost:8000"
    )
    os.environ["INFERENCE_API_ADDRESS"] = api_url
    console.print(Panel("Configuration saved!", style="green"))

@cli.command()
def inference():
    """Submit inference data"""
    cli_handler = CLI()
    category = cli_handler.display_categories()
    target = cli_handler.get_target_variable(category)
    cli_handler.submit_inference(category, target)

@cli.command()
def ground_truth():
    """Submit ground truth data"""
    cli_handler = CLI()
    category = cli_handler.display_categories()
    target = cli_handler.get_target_variable(category)
    cli_handler.submit_ground_truth(category, target)

@cli.command()
def insights():
    """Get market insights for cryptocurrency predictions"""
    cli_handler = CLI()
    crypto = cli_handler.display_cryptocurrencies()
    timeframe = cli_handler.display_timeframes(crypto)
    
    topic_id = TOPIC_MAPPING.get((crypto, timeframe))
    if topic_id:
        cli_handler.get_market_insights(topic_id)
    else:
        console.print("[red]Topic not found for selected combination[/red]")

if __name__ == "__main__":
    cli()
