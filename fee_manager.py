import click
import requests
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import pandas as pd

console = Console()

# Define exchange API endpoints
EXCHANGE_APIS = {
    "okx": {
        "funding_rate": "https://www.okx.com/api/v5/public/funding-rate",
        "ticker": "https://www.okx.com/api/v5/market/ticker"
    },
    "dydx": {
        "funding_rate": "https://api.dydx.exchange/v3/markets",
        "ticker": "https://api.dydx.exchange/v3/markets"
    },
    "vertex": {
        "funding_rate": "https://prod.vertexprotocol.com/v1/query/funding_rates",
        "ticker": "https://prod.vertexprotocol.com/v1/query/market_summary"
    },
    "bitget": {
        "funding_rate": "https://api.bitget.com/api/mix/v1/market/current-fundRate",
        "ticker": "https://api.bitget.com/api/mix/v1/market/ticker"
    },
    "coinex": {
        "funding_rate": "https://api.coinex.com/v1/contract/funding_rate",
        "ticker": "https://api.coinex.com/v1/market/ticker"
    },
    "bingx": {
        "funding_rate": "https://open-api.bingx.com/openApi/swap/v2/quote/fundingRate",
        "ticker": "https://open-api.bingx.com/openApi/swap/v2/quote/price"
    },
    "cryptocom": {
        "funding_rate": "https://api.crypto.com/v2/public/get-funding-rate",
        "ticker": "https://api.crypto.com/v2/public/get-ticker"
    }
}

# Add these constants at the top after the existing EXCHANGE_APIS
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

class FeeManager:
    def __init__(self):
        self.console = Console()
        
    def get_funding_rates(self, asset: str) -> Dict[str, float]:
        """Fetch funding rates from different exchanges"""
        funding_rates = {}
        
        for exchange, endpoints in EXCHANGE_APIS.items():
            try:
                if exchange == "okx":
                    okx_symbol = f"{asset}-USDT-SWAP"
                    response = requests.get(
                        endpoints["funding_rate"],
                        params={"instId": okx_symbol},
                        timeout=5
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("data") and len(data["data"]) > 0:
                            funding_rates[exchange] = float(data["data"][0]["fundingRate"]) * 100

                elif exchange == "dydx":
                    symbol = f"{asset}-USD"
                    response = requests.get(endpoints["funding_rate"])
                    if response.status_code == 200:
                        data = response.json()
                        for market in data["markets"]:
                            if market == symbol:
                                funding_rates[exchange] = float(data["markets"][market]["nextFundingRate"]) * 100

                elif exchange == "vertex":
                    response = requests.get(endpoints["funding_rate"])
                    if response.status_code == 200:
                        data = response.json()
                        for rate in data["funding_rates"]:
                            if rate["product"] == f"{asset}_USD":
                                funding_rates[exchange] = float(rate["rate"]) * 100

                elif exchange == "bitget":
                    symbol = f"{asset}USDT_UMCBL"
                    response = requests.get(
                        endpoints["funding_rate"],
                        params={"symbol": symbol},
                        timeout=5
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("data"):
                            funding_rates[exchange] = float(data["data"]["fundingRate"]) * 100

                elif exchange == "coinex":
                    symbol = f"{asset}USDT"
                    response = requests.get(
                        endpoints["funding_rate"],
                        params={"contract": symbol},
                        timeout=5
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("data"):
                            funding_rates[exchange] = float(data["data"]["funding_rate"]) * 100

                elif exchange == "bingx":
                    symbol = f"{asset}USDT"
                    response = requests.get(
                        endpoints["funding_rate"],
                        params={"symbol": symbol},
                        timeout=5
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("data"):
                            funding_rates[exchange] = float(data["data"]["fundingRate"]) * 100

                elif exchange == "cryptocom":
                    symbol = f"{asset}_USDT"
                    response = requests.get(
                        endpoints["funding_rate"],
                        params={"instrument_name": symbol},
                        timeout=5
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("result"):
                            funding_rates[exchange] = float(data["result"]["funding_rate"]) * 100

                # Print debug info
                console.print(f"[yellow]Debug {exchange}:[/yellow]")
                console.print(f"URL: {response.url}")
                console.print(f"Status: {response.status_code}")
                console.print(f"Response: {response.text[:200]}...")  # First 200 chars
                
            except Exception as e:
                console.print(f"[red]Error fetching {exchange} funding rate: {str(e)}[/red]")
                funding_rates[exchange] = None
        
        # Print the collected rates
        console.print("\n[cyan]Collected Funding Rates:[/cyan]")
        for exchange, rate in funding_rates.items():
            if rate is not None:
                console.print(f"{exchange}: {rate:.4f}%")
            else:
                console.print(f"{exchange}: N/A")
        
        return funding_rates

    def get_market_data(self, asset: str) -> Dict:
        """Get current market data including price and volatility"""
        try:
            symbol = f"{asset}-USD"
            data = yf.download(symbol, period="1mo", interval="1h")
            
            if data.empty:
                raise ValueError(f"No data retrieved for {symbol}")
            
            # Calculate volatility (24h)
            hourly_returns = data['Close'].pct_change()
            volatility_24h = hourly_returns.tail(24).std().iloc[0] * np.sqrt(24) * 100
            
            # Calculate price metrics
            current_price = data['Close'].iloc[-1].iloc[0]
            price_24h_ago = data['Close'].iloc[-24].iloc[0]
            price_change_24h = ((current_price / price_24h_ago) - 1) * 100
            
            return {
                "asset": asset,
                "current_price": current_price,
                "price_change_24h": price_change_24h,
                "volatility_24h": volatility_24h
            }
            
        except Exception as e:
            console.print(f"[red]Error fetching market data: {str(e)}[/red]")
            return None

    def analyze_opportunity(self, 
                           funding_rates: Dict[str, float], 
                           market_data: Dict,
                           amount: float) -> Dict:
        """Analyze opportunities based on funding rates and market conditions"""
        
        # Get Allora predictions
        predictions = self.get_allora_predictions(market_data['asset'])
        
        # Initialize strategy with predictions
        strategy = {
            "position": "neutral",
            "allocation": 0,
            "reasoning": [],
            "exchange_recommendations": [],
            "risk_level": "medium",
            "predicted_price": predictions['predicted_price'],
            "predicted_volatility": predictions['predicted_volatility'] * 100,  # Convert to percentage
            "predicted_price_change": ((predictions['predicted_price'] - market_data['current_price']) 
                                     / market_data['current_price'] * 100)
        }
        
        # Calculate average funding rate
        valid_rates = [rate for rate in funding_rates.values() if rate is not None]
        avg_funding_rate = sum(valid_rates) / len(valid_rates) if valid_rates else 0
        
        # Define strategy thresholds
        HIGH_FUNDING_THRESHOLD = 0.1  # 0.1% per 8 hours
        HIGH_VOLATILITY_THRESHOLD = 50  # 50% annualized
        
        # Analyze funding rates and predicted price movement
        if abs(avg_funding_rate) > HIGH_FUNDING_THRESHOLD:
            # Consider both funding rate and predicted price movement
            if avg_funding_rate > 0 and strategy["predicted_price_change"] < 0:
                strategy["position"] = "short"
                strategy["reasoning"].append(
                    f"High positive funding rate ({avg_funding_rate:.3f}%) and predicted price decrease "
                    f"({strategy['predicted_price_change']:.2f}%) suggests shorting"
                )
            elif avg_funding_rate < 0 and strategy["predicted_price_change"] > 0:
                strategy["position"] = "long"
                strategy["reasoning"].append(
                    f"High negative funding rate ({avg_funding_rate:.3f}%) and predicted price increase "
                    f"({strategy['predicted_price_change']:.2f}%) suggests longing"
                )
        
        # Adjust for volatility
        if market_data["volatility_24h"] > HIGH_VOLATILITY_THRESHOLD:
            strategy["risk_level"] = "high"
            strategy["allocation"] = amount * 0.5  # Reduce allocation in high volatility
            strategy["reasoning"].append(f"High volatility ({market_data['volatility_24h']:.1f}%) suggests reduced position size")
        else:
            strategy["allocation"] = amount * 0.8
            
        # Find best exchanges
        sorted_exchanges = sorted(
            funding_rates.items(),
            key=lambda x: abs(x[1]) if x[1] is not None else 0,
            reverse=True
        )
        
        for exchange, rate in sorted_exchanges:
            if rate is not None:
                strategy["exchange_recommendations"].append({
                    "exchange": exchange,
                    "funding_rate": rate,
                    "suggested_allocation": strategy["allocation"] / len(sorted_exchanges)
                })
        
        return strategy

    def display_analysis(self, 
                        asset: str,
                        funding_rates: Dict[str, float],
                        market_data: Dict,
                        strategy: Dict):
        """Display comprehensive analysis and recommendations"""
        
        # Funding Rates Table
        funding_table = Table(title=f"Funding Rates for {asset}")
        funding_table.add_column("Exchange", style="cyan")
        funding_table.add_column("Rate (8h)", style="green")
        
        for exchange, rate in funding_rates.items():
            rate_str = f"{rate:.4f}%" if rate is not None else "N/A"
            funding_table.add_row(exchange, rate_str)
        
        console.print(funding_table)
        console.print()
        
        # Market Data Table
        market_table = Table(title="Market Conditions")
        market_table.add_column("Metric", style="cyan")
        market_table.add_column("Value", style="green")
        
        market_table.add_row("Current Price", f"${market_data['current_price']:,.2f}")
        market_table.add_row("24h Change", f"{market_data['price_change_24h']:+.2f}%")
        market_table.add_row("24h Volatility", f"{market_data['volatility_24h']:.2f}%")
        market_table.add_row("Predicted Price", f"${strategy['predicted_price']:,.2f}")
        market_table.add_row("Predicted Price Change", f"{strategy['predicted_price_change']:+.2f}%")
        market_table.add_row("Predicted Volatility", f"{strategy.get('predicted_volatility', 0):,.2f}%")
        
        console.print(market_table)
        console.print()
        
        # Strategy Table
        strategy_table = Table(title="Trading Strategy")
        strategy_table.add_column("Component", style="cyan")
        strategy_table.add_column("Details", style="green")
        
        strategy_table.add_row("Position", strategy["position"].upper())
        strategy_table.add_row("Risk Level", strategy["risk_level"].upper())
        strategy_table.add_row("Total Allocation", f"${strategy['allocation']:,.2f}")
        
        console.print(strategy_table)
        console.print()
        
        # Exchange Recommendations
        exchange_table = Table(title="Exchange Allocations")
        exchange_table.add_column("Exchange", style="cyan")
        exchange_table.add_column("Funding Rate", style="yellow")
        exchange_table.add_column("Suggested Allocation", style="green")
        
        for rec in strategy["exchange_recommendations"]:
            exchange_table.add_row(
                rec["exchange"],
                f"{rec['funding_rate']:.4f}%",
                f"${rec['suggested_allocation']:,.2f}"
            )
        
        console.print(exchange_table)
        console.print()
        
        # Reasoning
        console.print("[bold cyan]Strategy Reasoning:[/bold cyan]")
        for reason in strategy["reasoning"]:
            console.print(f"• {reason}")

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
                current_price = self.get_market_data(asset)['current_price']
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

@click.command()
@click.option('--asset', type=click.Choice(['ETH', 'BTC']), required=True, help='Asset to analyze')
@click.option('--amount', type=float, required=True, help='Total amount to allocate')
def main(asset: str, amount: float):
    """Funding Rate Analysis and Trading Strategy Generator"""
    
    manager = FeeManager()
    
    # Get funding rates
    console.print(f"\n[cyan]Fetching funding rates for {asset}...[/cyan]")
    funding_rates = manager.get_funding_rates(asset)
    
    # Get market data
    console.print(f"[cyan]Analyzing market conditions...[/cyan]")
    market_data = manager.get_market_data(asset)
    
    if market_data:
        # Analyze opportunity
        strategy = manager.analyze_opportunity(funding_rates, market_data, amount)
        
        # Display analysis
        manager.display_analysis(asset, funding_rates, market_data, strategy)
    else:
        console.print("[red]Unable to complete analysis due to missing market data[/red]")

if __name__ == "__main__":
    main()