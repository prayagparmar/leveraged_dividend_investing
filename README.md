# Leveraged DRIP Strategy Backtesting Tool

A comprehensive Python backtesting system for evaluating leveraged dividend reinvestment plans (DRIP) with margin trading, dynamic interest rates, and income generation capabilities.

## Overview

This tool simulates a leveraged dividend reinvestment strategy where:
- Initial capital is leveraged using margin to purchase dividend-paying securities
- All dividends are automatically reinvested (with optional leverage)
- Interest costs are paid from dividends or accumulated as margin debt
- Optional income withdrawal feature for generating cash flow
- Comprehensive risk management with margin call simulations
- Uses dynamic Federal Funds Rate data for realistic interest cost modeling

## Features

- **Dynamic Interest Rates**: Uses Federal Reserve data (FEDFUNDS.csv) for accurate historical interest costs
- **Flexible Leverage**: Configurable target leverage ratios with margin requirement enforcement
- **Income Generation**: Optional dividend income withdrawal with customizable rates and hold-off periods
- **Tax Modeling**: Dividend taxation with configurable tax rates
- **Risk Management**: Automatic margin call handling and position liquidation
- **Wind-down Strategy**: Automatic deleveraging when equity thresholds are reached
- **Comprehensive Analytics**: Greeks calculation, performance metrics, and benchmark comparisons
- **Visualization**: Interactive plots showing portfolio performance, leverage ratios, and income streams

## Dependencies

Install the required Python packages:

```bash
pip install yfinance pandas numpy matplotlib
```

### Required Packages:
- `yfinance` - Yahoo Finance data retrieval
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `datetime` - Date/time handling (built-in)
- `typing` - Type hints (built-in)
- `argparse` - Command-line argument parsing (built-in)
- `warnings` - Warning management (built-in)

## Required Data Files

**FEDFUNDS.csv**: Federal Funds Rate data from the Federal Reserve Economic Data (FRED)
- Download from: https://fred.stlouisfed.org/series/FEDFUNDS
- Format: CSV with columns `observation_date` and `FEDFUNDS`
- Place in the same directory as the script
- If missing, the script will use a default 2% rate with a warning

## Usage

### Basic Usage

```bash
python3 leverage_drip_income.py --ticker QYLD --start_date 2014-01-01 --leverage_ratio 2 --plot
```

### Advanced Usage with Income Generation

```bash
python leverage_drip_income.py \
    --ticker SCHD \
    --start_date 2012-01-01 \
    --end_date 2024-01-01 \
    --initial_investment 250000 \
    --leverage_ratio 1.5 \
    --income_withdrawal_rate 0.8 \
    --income_hold_off_years 10 \
    --tax_rate 0.15 \
    --plot \
    --export_greeks
```

## Command Line Parameters

### Core Strategy Parameters
- `--ticker` (default: QYLD): Stock ticker symbol to backtest
- `--start_date` (default: 2001-01-01): Start date in YYYY-MM-DD format
- `--end_date` (default: current date): End date in YYYY-MM-DD format
- `--initial_investment` (default: 100000): Initial investment amount in dollars
- `--leverage_ratio` (default: 2.0): Target leverage ratio (e.g., 2.0 = 2x leverage)

### Risk Management Parameters
- `--margin_requirement` (default: 0.25): Minimum equity ratio for margin maintenance (25%)
- `--broker_spread` (default: 0.02): Spread over Fed rate charged by broker (200 basis points)
- `--wind_down_threshold` (default: 100000000): Equity threshold to trigger deleveraging
- `--wind_down_rate` (default: 1.0): Fraction of dividends used for debt repayment during wind-down

### Tax and Income Parameters
- `--tax_rate` (default: 0.2): Tax rate on dividends (20%)
- `--income_withdrawal_rate` (default: 0.5): Percentage of net dividends withdrawn as income (50%)
- `--income_hold_off_years` (default: 20.0): Years to wait before starting income withdrawals
- `--pay_interest_from_dividends` (default: True): Pay interest from dividends vs. accumulating as debt

### Output Options
- `--plot`: Display interactive performance charts
- `--export_greeks`: Export Greeks analysis to CSV file

## Strategy Mechanics

### Portfolio Initialization
1. Uses initial capital plus borrowed funds to purchase shares
2. Calculates maximum allowable leverage based on margin requirements
3. Maintains target leverage ratio throughout the investment period

### Dividend Processing
1. Receives gross dividends on all shares held
2. Applies dividend tax at specified rate
3. Uses net dividends to pay accrued interest (if enabled)
4. Withdraws specified percentage as income (after hold-off period)
5. Reinvests remaining dividends with leverage to maintain target ratio

### Interest Cost Management
- Uses dynamic Federal Funds Rate + broker spread
- Interest accrues daily, charged monthly
- Can be paid from dividends or accumulated as additional debt
- Realistic interest rate environment based on historical Fed policy

### Risk Management
- Continuous monitoring of equity-to-portfolio ratio
- Automatic position liquidation if below maintenance margin
- Margin calls trigger immediate asset sales to restore compliance
- Wind-down feature for systematic deleveraging at high equity levels

## Output Analysis

### Performance Metrics
- **Equity CAGR**: Compound annual growth rate of net equity
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return metric
- **Volatility**: Annualized standard deviation of returns

### Greeks Analysis
- **Delta**: Position sensitivity to underlying price changes
- **Gamma**: Rate of change of delta
- **Theta**: Daily cost of carry (interest - dividends)
- **Vega**: Sensitivity to volatility changes
- **Rho**: Sensitivity to interest rate changes

### Cash Flow Analysis
- Total dividends received and taxes paid
- Interest costs and loan repayments
- Income withdrawn vs. reinvested
- Margin call frequency and amounts

### Benchmark Comparison
- Unleveraged buy-and-hold performance
- Strategy outperformance/underperformance
- Risk-adjusted comparison metrics

## Example Scenarios

### Conservative Income Strategy
```bash
python leverage_drip_income.py \
    --ticker VYM \
    --leverage_ratio 1.3 \
    --income_withdrawal_rate 0.9 \
    --income_hold_off_years 5 \
    --tax_rate 0.15
```

### Aggressive Growth Strategy
```bash
python leverage_drip_income.py \
    --ticker JEPI \
    --leverage_ratio 2.5 \
    --income_withdrawal_rate 0.1 \
    --income_hold_off_years 15 \
    --wind_down_threshold 500000
```

### Risk Assessment Mode
```bash
python leverage_drip_income.py \
    --ticker QYLD \
    --leverage_ratio 3.0 \
    --margin_requirement 0.30 \
    --broker_spread 0.035 \
    --plot \
    --export_greeks
```

## File Structure

```
project/
├── leverage_drip_income.py    # Main script
├── FEDFUNDS.csv              # Federal Reserve interest rate data
├── README.md                 # This documentation
└── output/                   # Generated files (optional)
    ├── {TICKER}_greeks_summary.csv
    └── performance_charts.png
```

## Risk Warnings

⚠️ **This tool is for educational and research purposes only**
- Historical backtesting does not guarantee future performance
- Leveraged investing involves significant risk of loss
- Margin calls can result in forced liquidation at unfavorable prices
- Interest rate changes can dramatically impact strategy performance
- Dividend cuts or suspensions can cause severe losses in leveraged positions

## Technical Notes

### Data Source
- Uses Yahoo Finance via yfinance library
- Fetches split-adjusted prices without dividend adjustments
- Handles corporate actions (splits, special dividends) automatically
- Requires active internet connection for data retrieval

### Performance Considerations
- Memory usage scales with backtest period length
- Large datasets may require increased timeout values
- Plotting large datasets may be slow on older systems

### Limitations
- Does not account for bid-ask spreads
- Assumes perfect dividend reinvestment timing
- No consideration for early exercise of margin calls
- Tax implications may vary by jurisdiction

## Contributing

This is a research tool. Contributions welcome for:
- Additional risk metrics
- Enhanced visualization options
- Alternative interest rate models
- Performance optimizations

## License

Educational use only. Not for commercial trading purposes.
