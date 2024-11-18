import click
import requests
from typing import Dict, Tuple, List
from rich.console import Console
from rich.table import Table
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import pandas as pd
import openai
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

console = Console()

ALLORA_API_BASE = "https://allora-api.testnet.allora.network/emissions/v5"

TOPICS = {
    "ETH": {
        "price": 13,    # ETH 5min Price Prediction
        "volatility": 15  # ETH 5min Volatility Prediction
    },
    "BTC": {
        "price": 14,    # BTC 5min Price Prediction
        "volatility": 16  # BTC 5min Volatility Prediction
    }
}

class AlloraLiquidityManager:
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
    def get_historical_metrics(self, asset: str) -> Dict:
        """Get historical price data and metrics for analysis"""
        # Get 1 year of daily data
        symbol = f"{asset}-USD"
        data = yf.download(symbol, period="1y", interval="1d")
        
        # Calculate basic metrics
        metrics = {
            "volatility": float(data['Close'].pct_change().std() * np.sqrt(252)),
            "max_drawdown": self._calculate_max_drawdown(data['Close']),
            "daily_returns": data['Close'].pct_change().dropna().values.tolist()[-30:],  # Last 30 days
            "current_price": float(data['Close'].iloc[-1]),
            "price_30d_ago": float(data['Close'].iloc[-31])
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate the maximum drawdown from peak"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return float(drawdown.min())

    def get_llm_analysis(self, asset: str, metrics: Dict) -> Dict:
        """Get LLM analysis of historical data and protocol recommendations"""
        
        # Use triple quotes with raw string to avoid f-string interpretation of JSON structure
        prompt = f"""
        You are a DeFi expert analyzing liquidity provision strategies. Given the following metrics for {asset}:
        
        - 30-day price change: {((metrics['current_price'] / metrics['price_30d_ago']) - 1) * 100:.2f}%
        - Annualized volatility: {metrics['volatility'] * 100:.2f}%
        - Maximum drawdown: {metrics['max_drawdown'] * 100:.2f}%
        
        For the following protocols:
        1. Curve (known for stable pairs, low IL risk, conservative)
        2. Uniswap (balanced risk, medium IL exposure)
        3. Balancer (flexible weights, medium-high IL risk)
        
        Please provide:
        1. A risk assessment for each protocol given current market conditions
        2. Recommended allocation percentages across these protocols for a balanced strategy
        3. Brief explanation for each allocation
        4. Specific recommendations for:
           - Optimal pool selection within each protocol
           - Target price ranges for concentrated liquidity (if applicable)
           - Rebalancing frequency suggestions
        
        Format your response as JSON with the following structure:
        {{
            "protocol_risks": {{
                "curve": {{"risk_level": "", "risk_factors": []}},
                "uniswap": {{"risk_level": "", "risk_factors": []}},
                "balancer": {{"risk_level": "", "risk_factors": []}}
            }},
            "allocations": {{
                "curve": {{
                    "percentage": 0,
                    "reasoning": "",
                    "pool_recommendations": "",
                    "rebalancing_frequency": ""
                }},
                "uniswap": {{
                    "percentage": 0,
                    "reasoning": "",
                    "pool_recommendations": "",
                    "price_range": "",
                    "rebalancing_frequency": ""
                }},
                "balancer": {{
                    "percentage": 0,
                    "reasoning": "",
                    "pool_recommendations": "",
                    "rebalancing_frequency": ""
                }}
            }},
            "market_outlook": "",
            "general_recommendations": []
        }}
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a DeFi expert analyzing liquidity provision strategies."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        return response.choices[0].message.content

    def determine_allocation_strategy(self, 
                                   asset: str,
                                   total_amount: float) -> Dict:
        """Determine allocation strategy using LLM analysis"""
        
        # Get historical metrics
        metrics = self.get_historical_metrics(asset)
        
        # Get LLM analysis and parse JSON
        analysis = json.loads(self.get_llm_analysis(asset, metrics))
        
        # Calculate allocations based on analysis
        allocations = {}
        for protocol, details in analysis["allocations"].items():
            amount = (details["percentage"] / 100) * total_amount
            risk_details = analysis["protocol_risks"][protocol]
            
            allocations[protocol] = {
                "amount": amount,
                "allocation_percentage": details["percentage"],
                "reasoning": details["reasoning"],
                "risk_level": risk_details["risk_level"],
                "risk_factors": risk_details["risk_factors"]
            }
        
        return allocations

    def get_allora_predictions(self, asset: str) -> Dict:
        """Get predictions from Allora API"""
        try:
            # Get price prediction
            price_response = requests.get(
                f"{ALLORA_API_BASE}/latest_network_inferences/{TOPICS[asset]['price']}"
            )
            price_data = price_response.json()
            predicted_price = float(price_data["network_inferences"]["combined_value"])

            # Get volatility prediction
            volatility_response = requests.get(
                f"{ALLORA_API_BASE}/latest_network_inferences/{TOPICS[asset]['volatility']}"
            )
            volatility_data = volatility_response.json()
            predicted_volatility = float(volatility_data["network_inferences"]["combined_value"])

            # If predicted volatility is over 100%, recalculate using predicted price
            if predicted_volatility > 1.0:  # 100% = 1.0
                # Get current price from historical metrics
                current_price = self.get_historical_metrics(asset)['current_price']
                # Calculate volatility as absolute percentage change
                predicted_volatility = abs((predicted_price - current_price) / current_price)

            return {
                "predicted_price": predicted_price,
                "predicted_volatility": predicted_volatility
            }
        except Exception as e:
            console.print(f"[red]Error fetching Allora predictions: {str(e)}[/red]")
            return {
                "predicted_price": 0,
                "predicted_volatility": 0
            }

    def display_market_analysis(self, 
                              asset: str,
                              metrics: Dict,
                              allocations: Dict):
        """Display comprehensive market analysis and allocations"""
        
        # Get Allora predictions
        predictions = self.get_allora_predictions(asset)
        
        # Predictions Comparison Table
        predictions_table = Table(title=f"Price and Volatility Analysis for {asset}")
        predictions_table.add_column("Metric", style="cyan")
        predictions_table.add_column("Current", style="green", justify="right")
        predictions_table.add_column("Predicted (5min)", style="yellow", justify="right")
        
        predictions_table.add_row(
            "Price",
            f"${metrics['current_price']:,.2f}",
            f"${predictions['predicted_price']:,.2f}"
        )
        predictions_table.add_row(
            "Volatility (Annual)",
            f"{metrics['volatility']*100:.2f}%",
            f"{predictions['predicted_volatility']*100:.2f}%"
        )
        
        console.print(predictions_table)
        console.print()
        
        # Market Metrics Table
        metrics_table = Table(title=f"Market Analysis for {asset}")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green", justify="right")
        
        metrics_table.add_row("Current Price", f"${metrics['current_price']:,.2f}")
        metrics_table.add_row("Annual Volatility", f"{metrics['volatility']*100:.2f}%")
        metrics_table.add_row("Maximum Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        
        console.print(metrics_table)
        console.print()
        
        # Allocations Table
        alloc_table = Table(title="Recommended Allocations")
        alloc_table.add_column("Protocol", style="cyan")
        alloc_table.add_column("Amount", style="green", justify="right")
        alloc_table.add_column("Allocation %", style="yellow", justify="right")
        alloc_table.add_column("Risk Level", style="red")
        
        for protocol, details in allocations.items():
            alloc_table.add_row(
                protocol.capitalize(),
                f"${details['amount']:,.2f}",
                f"{details['allocation_percentage']:.1f}%",
                details['risk_level']
            )
        
        console.print(alloc_table)
        console.print()
        
        # Rationale Table
        rationale_table = Table(title="Allocation Rationale")
        rationale_table.add_column("Protocol", style="cyan")
        rationale_table.add_column("Reasoning", style="white", width=60)
        rationale_table.add_column("Risk Factors", style="red", width=40)
        
        for protocol, details in allocations.items():
            rationale_table.add_row(
                protocol.capitalize(),
                details['reasoning'],
                "\n".join(details['risk_factors'])
            )
        
        console.print(rationale_table)
        
        # Market Outlook
        if "market_outlook" in allocations:
            console.print("[bold cyan]Market Outlook:[/bold cyan]")
            console.print(allocations["market_outlook"])
            console.print()
        
        # General Recommendations
        if "general_recommendations" in allocations:
            console.print("[bold cyan]General Recommendations:[/bold cyan]")
            for rec in allocations.get("general_recommendations", []):
                console.print(f"• {rec}")

@click.command()
@click.option('--asset', type=click.Choice(['ETH', 'BTC']), required=True, help='Asset to analyze')
@click.option('--amount', type=float, required=True, help='Total amount to allocate')
@click.option('--openai-api-key', help='OpenAI API Key (optional if set in .env file)', envvar='OPENAI_API_KEY')
def main(asset: str, amount: float, openai_api_key: str):
    """Automated Liquidity Management using LLM analysis"""
    
    if not openai_api_key:
        console.print("[red]Error: OpenAI API key not provided. Set it in .env file or pass via --openai-api-key[/red]")
        return

    manager = AlloraLiquidityManager(openai_api_key)
    
    # Get historical metrics
    console.print(f"\n[cyan]Analyzing historical data for {asset}...[/cyan]")
    metrics = manager.get_historical_metrics(asset)
    
    # Determine allocation strategy
    allocations = manager.determine_allocation_strategy(asset, amount)
    
    # Display analysis and recommendations
    manager.display_market_analysis(asset, metrics, allocations)

if __name__ == "__main__":
    main()