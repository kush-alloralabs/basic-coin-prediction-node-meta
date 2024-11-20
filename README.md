# Basic Coin Prediction Node

## Quick Start

### Market Insights
To view market insights for cryptocurrency predictions:

```bash
python cli.py insights
```

This will:
1. Show you a list of available cryptocurrencies (ETH, BTC, SOL, BNB, ARB)
2. Let you select a cryptocurrency
3. Show available timeframes for that cryptocurrency
4. Display detailed market analysis including:
   - Price distribution analysis
   - Statistical price ranges
   - Network contributor information
   - Confidence intervals

### Liquidity Management
To get AI-powered liquidity management recommendations:

```bash
python liquidity_manager.py --asset [ETH, BTC, SOL, BNB, ARB] --amount [amount in chosen currency]
```

This will:
1. Analyze historical data for the selected asset
2. Generate optimal liquidity allocation strategies
3. Provide detailed market analysis including:
   - Risk assessment
   - Position sizing recommendations
   - Market condition analysis
   - Allocation breakdowns

## Configuration

### Environment Variables
Create a `.env` file in the project root with the following configurations

```plaintext
# Required for basic functionality
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```


Here's a detailed README focused on the liquidity and fee management components:

# Cryptocurrency Liquidity & Fee Management System Deep Dive

## Overview
This system provides AI-powered liquidity management and fee analysis for cryptocurrency trading, leveraging the Allora Network's predictive capabilities. It combines real-time market data, funding rates across multiple exchanges, and machine learning predictions to optimize liquidity allocation and fee strategies.

## Key Components

### 1. Liquidity Manager
The Liquidity Manager (`liquidity_manager.py`) provides sophisticated liquidity allocation strategies using historical data analysis and AI-powered predictions.

#### Features:
- Historical metrics analysis
- AI-powered strategy generation using GPT-4
- Risk assessment across multiple protocols
- Dynamic allocation recommendations
- Market condition analysis

#### Data Sources:

```37:53:basic-coin-prediction-node-meta/liquidity_manager.py
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
    
```


#### GPT-4 Prompt Construction:
The system uses a carefully crafted prompt to generate liquidity strategies:

```63:116:basic-coin-prediction-node-meta/liquidity_manager.py
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
```


### 2. Fee Manager
The Fee Manager (`fee_manager.py`) analyzes funding rates and fees across multiple exchanges to identify arbitrage opportunities and optimize trading costs.

#### Supported Exchanges:

```14:43:basic-coin-prediction-node-meta/fee_manager.py
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
```


#### Features:
- Real-time funding rate analysis
- Cross-exchange fee comparison
- Historical fee pattern analysis
- Arbitrage opportunity detection
- Risk-adjusted fee optimization

## Allora Network Integration

### API Configuration
The system integrates with Allora Network's prediction API for enhanced market insights:

```python
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
```

### Prediction Integration
The system uses Allora's predictions for:
- Price movement forecasting
- Volatility predictions
- Market sentiment analysis
- Risk assessment

## Usage

### Liquidity Management
```bash
python liquidity_manager.py --asset [ETH/BTC] --amount [amount]
```

This will:
1. Analyze historical market data
2. Generate AI-powered allocation strategies
3. Provide protocol-specific recommendations
4. Display risk assessments and confidence metrics

### Fee Analysis
```bash
python fee_manager.py --asset [ETH/BTC]
```

Outputs:
- Current funding rates across exchanges
- Fee arbitrage opportunities
- Historical fee patterns
- Risk-adjusted recommendations

## Data Processing Pipeline

1. **Market Data Collection**
   - Historical price data from yfinance
   - Real-time exchange data
   - Allora Network predictions

2. **Analysis Layer**
   - Volatility calculations
   - Risk metrics computation
   - Cross-exchange fee analysis
   - Protocol-specific risk assessment

3. **Strategy Generation**
   - GPT-4 analysis of market conditions
   - Protocol-specific recommendations
   - Risk-adjusted allocation strategies
   - Fee optimization suggestions

## Visualization and Reporting

The system provides rich visualization of analysis results:

```175:304:basic-coin-prediction-node-meta/cli.py
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
```


## Configuration

### Environment Variables
Required in `.env`:
```plaintext
OPENAI_API_KEY=your_api_key_here
```

### Exchange API Configuration
The system supports multiple exchanges with configurable endpoints and parameters:

```14:43:basic-coin-prediction-node-meta/fee_manager.py
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
```


## Technical Details

### Market Analysis Components
- Price distribution analysis
- Volatility modeling
- Funding rate analysis
- Protocol risk assessment
- Cross-exchange arbitrage detection

### AI Integration
- GPT-4 for strategy generation
- Allora Network for price predictions
- Custom ML models for risk assessment

### Data Sources
- Exchange APIs for real-time data
- Historical price data from yfinance
- Protocol-specific metrics
- Network consensus data from Allora

This system provides a comprehensive solution for cryptocurrency liquidity management and fee optimization, combining traditional market analysis with cutting-edge AI predictions and cross-exchange analysis.

</rewritten_file>
