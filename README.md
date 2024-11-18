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
python liquidity_manager.py --asset ETH --amount 1000
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

</rewritten_file>
