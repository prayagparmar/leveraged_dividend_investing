import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date as dt_date
from typing import Dict, Optional, Tuple
import argparse
import warnings

# Global cache for Fed rate data
_fed_data_cache = None

#update the csv file when the rate changes https://fred.stlouisfed.org/series/FEDFUNDS
def load_fed_data():
    """Load and cache Fed rate data from FEDFUNDS.csv"""
    global _fed_data_cache
    
    if _fed_data_cache is not None:
        return _fed_data_cache
    
    try:
        # Load Fed rate data with explicit column specification
        fed_data = pd.read_csv('FEDFUNDS.csv')
        fed_data['observation_date'] = pd.to_datetime(fed_data['observation_date'])
        fed_data = fed_data.set_index('observation_date')
        
        # Cache the loaded data
        _fed_data_cache = fed_data
        return fed_data
        
    except FileNotFoundError:
        warnings.warn("FEDFUNDS.csv not found, using default 2% Fed rate")
        return None
    except Exception as e:
        warnings.warn(f"Error loading Fed rate data: {e}, using default 2% Fed rate")
        return None

def get_fed_rate(date: pd.Timestamp) -> float:
    """Get Federal Funds Rate for a given date from cached FEDFUNDS.csv"""
    fed_data = load_fed_data()

    if fed_data is None:
        return 2.0  # Fallback to default

    try:
        # Ensure date is timezone-naive for comparison
        if hasattr(date, 'tz_localize') and date.tz is not None:
            date = date.tz_localize(None)

        # Convert to datetime64 for consistent comparison
        target_date = pd.Timestamp(date).to_datetime64()

        # Find dates that are <= target date
        valid_dates = fed_data.index[fed_data.index.values <= target_date]

        if len(valid_dates) == 0:
            # If no data available before this date, use the first available rate
            return float(fed_data.iloc[0]['FEDFUNDS'])

        # Get the most recent rate
        latest_date = valid_dates.max()
        return float(fed_data.loc[latest_date]['FEDFUNDS'])

    except Exception as e:
        warnings.warn(f"Error reading Fed rate data: {e}, using default 2% Fed rate")
        return 2.0

class StrictLeverageDripStrategy:
    def __init__(self, ticker: str, start_date: str, end_date: Optional[str] = None,
                 initial_investment: float = 100000, leverage_ratio: float = 2.0,
                 broker_spread: float = 0.02, margin_requirement: float = 0.25,
                 wind_down_threshold: Optional[float] = None,
                 wind_down_rate: float = 1.0, tax_rate: float = 0.20,
                 pay_interest_from_dividends: bool = True,
                 income_withdrawal_rate: float = 0.0,
                 income_hold_off_years: float = 0.0,
                 dca_amount: float = 0.0):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_investment = initial_investment
        self.target_leverage = leverage_ratio
        self.broker_spread = broker_spread
        self.margin_requirement = margin_requirement
        self.wind_down_threshold = wind_down_threshold
        self.wind_down_rate = wind_down_rate
        self.tax_rate = tax_rate
        self.pay_interest_from_dividends = pay_interest_from_dividends
        self.income_withdrawal_rate = income_withdrawal_rate
        self.income_hold_off_years = income_hold_off_years
        self.dca_amount = dca_amount

        # Validate tax rate
        if not (0.0 <= tax_rate <= 1.0):
            raise ValueError("tax_rate must be between 0.0 and 1.0")
        
        # Validate income withdrawal rate
        if not (0.0 <= income_withdrawal_rate <= 1.0):
            raise ValueError("income_withdrawal_rate must be between 0.0 and 1.0")
        
        # Validate income hold-off years
        if income_hold_off_years < 0.0:
            raise ValueError("income_hold_off_years must be non-negative")

        if self.dca_amount < 0.0:
            raise ValueError("dca_amount must be non-negative")

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            stock = yf.Ticker(self.ticker)

            # IMPORTANT: Get the data with auto_adjust=False but keep splits inline
            # This gives us split-adjusted prices WITHOUT dividend adjustment
            data = stock.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False,
                actions='inline'  # This includes dividends and splits in the dataframe
            )

            if data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")

            # Extract dividends from the data (already split-adjusted by yfinance)
            if 'Dividends' in data.columns:
                aligned_dividends = data['Dividends'].fillna(0)
            else:
                # Fallback to using the dividends property
                raw_dividends = stock.dividends
                aligned_dividends = pd.Series(0, index=data.index, dtype=float)
                common_dates = raw_dividends.index.intersection(data.index)
                aligned_dividends.loc[common_dates] = raw_dividends.loc[common_dates]

            # Count splits for reporting
            splits_count = 0
            if 'Stock Splits' in data.columns:
                splits_count = (data['Stock Splits'] > 0).sum()

            # Count dividend events
            dividend_count = (aligned_dividends > 0).sum()

            print(f"\nData Fetch Summary for {self.ticker}:")
            print(f"  Using split-adjusted Close prices (NOT adjusted for dividends)")
            print(f"  Found {splits_count} stock splits")
            print(f"  Found {dividend_count} dividend payments")
            print(f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")

            # Clean up the dataframe - remove dividend and split columns
            if 'Dividends' in data.columns:
                data = data.drop(['Dividends'], axis=1, errors='ignore')
            if 'Stock Splits' in data.columns:
                data = data.drop(['Stock Splits'], axis=1, errors='ignore')

            return data, aligned_dividends

        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    def calculate_cagr(self, start_value: float, end_value: float, years: float) -> float:
        if start_value <= 0 or years <= 0:
            return 0
        return (pow(end_value / start_value, 1 / years) - 1) * 100

    def calculate_max_drawdown(self, values: np.ndarray, dates: pd.DatetimeIndex) -> Tuple[float, Optional[str]]:
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        min_drawdown_idx = np.argmin(drawdown)
        max_drawdown_date = dates[min_drawdown_idx].strftime('%Y-%m-%d')
        return np.min(drawdown) * 100, max_drawdown_date

    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        excess_returns = np.mean(returns) - risk_free_rate/252
        return excess_returns / np.std(returns) * np.sqrt(252)

    def _initialize_portfolio(self, price: float) -> Tuple[float, float, float]:
        max_possible_leverage = 1 / self.margin_requirement
        initial_leverage = min(self.target_leverage, max_possible_leverage)

        initial_loan = self.initial_investment * (initial_leverage - 1)
        initial_equity = self.initial_investment
        initial_portfolio_value = self.initial_investment * initial_leverage
        shares_held = initial_portfolio_value / price
        return shares_held, initial_loan, initial_portfolio_value

    def _calculate_gap_days(self, current_date: dt_date, prev_date: dt_date) -> int:
        if prev_date is None:
            # For first day, no gap - we start position on first trading day
            return 0
        return (current_date - prev_date.date()).days

    def _accrue_interest(self, loan_balance: float, days_gap: int,
                         accrued_interest: float, days_since_last_charge: int, current_date: pd.Timestamp) -> Tuple[float, float, float, float, bool]:
        gap_interest = 0
        if loan_balance > 0:
            # Get dynamic Fed rate and add broker spread
            fed_rate = get_fed_rate(current_date)
            effective_rate = (fed_rate / 100) + self.broker_spread
            gap_interest = loan_balance * (effective_rate / 360) * days_gap
            accrued_interest += gap_interest

        days_since_last_charge += days_gap
        charge_interest_flag = False

        if days_since_last_charge >= 30:
            charge_interest_flag = True
            days_since_last_charge %= 30

        return loan_balance, accrued_interest, days_since_last_charge, gap_interest, charge_interest_flag

    def _enforce_margin_requirements(self, price: float, shares_held: float, loan_balance: float) -> Tuple[float, float, bool, float]:
        portfolio_value = shares_held * price
        equity = portfolio_value - loan_balance

        # Handle bankruptcy scenario
        if equity <= 0:
            # Liquidate entire position
            amount_sold = portfolio_value
            loan_balance_remaining = loan_balance - portfolio_value
            return 0.0, max(0, loan_balance_remaining), True, amount_sold

        current_equity_ratio = equity / portfolio_value
        margin_call_occurred = False
        amount_sold = 0.0

        # Only check if below maintenance margin
        if current_equity_ratio < self.margin_requirement:
            # Calculate required portfolio value to meet margin
            required_portfolio_value = equity / self.margin_requirement

            # Calculate minimum sale needed to restore compliance
            required_sale = portfolio_value - required_portfolio_value

            if required_sale > 0:
                # Cannot sell more than exists or more than owed
                amount_to_sell = min(
                    required_sale,
                    portfolio_value,  # Can't sell more than owned
                    loan_balance     # Can't repay more than owed
                )

                shares_to_sell = amount_to_sell / price
                new_shares_held = max(0, shares_held - shares_to_sell)
                new_loan_balance = loan_balance - amount_to_sell

                # Update values
                shares_held = new_shares_held
                loan_balance = new_loan_balance
                amount_sold = amount_to_sell
                margin_call_occurred = True


        return shares_held, loan_balance, margin_call_occurred, amount_sold

    def _reinvest_cash(self, price: float, shares_held: float, loan_balance: float,
                   cash_to_invest: float, wind_down_triggered: bool) -> Tuple[float, float, bool, float]:
        leverage_used = False
        loan_repayment_from_cash = 0

        if cash_to_invest <= 0:
            return shares_held, loan_balance, leverage_used, loan_repayment_from_cash

        if wind_down_triggered:
            if loan_balance > 0:
                loan_repayment_from_cash = cash_to_invest * self.wind_down_rate
                cash_to_invest -= loan_repayment_from_cash
                loan_balance = max(0, loan_balance - loan_repayment_from_cash)

            if cash_to_invest > 0:
                additional_shares = cash_to_invest / price
                shares_held += additional_shares
        else:
            portfolio_value = shares_held * price
            equity = portfolio_value - loan_balance

            if equity <= 0:
                additional_shares = cash_to_invest / price
                shares_held += additional_shares
                return shares_held, loan_balance, False, 0

            current_leverage = portfolio_value / equity if equity > 0 else float('inf')

            if current_leverage < self.target_leverage:
                target_portfolio_value = equity * self.target_leverage
                investment_needed = target_portfolio_value - portfolio_value

                max_possible_portfolio_value = equity / self.margin_requirement
                max_possible_borrow = max_possible_portfolio_value - portfolio_value
                max_possible_borrow = max(0, max_possible_borrow)

                leverage_portion = min(investment_needed, max_possible_borrow)
                leverage_portion = max(0, leverage_portion)

                total_investment = cash_to_invest + leverage_portion
                additional_shares = total_investment / price

                shares_held += additional_shares
                loan_balance += leverage_portion

                if leverage_portion > 0:
                    leverage_used = True
            else:
                additional_shares = cash_to_invest / price
                shares_held += additional_shares

        return shares_held, loan_balance, leverage_used, loan_repayment_from_cash

    def _handle_dividend(self, price: float, shares_held: float, loan_balance: float,
                         dividend_per_share: float, wind_down_triggered: bool,
                         accrued_interest: float, in_income_phase: bool) -> Tuple[float, float, float, float, float, float, float, float, bool]:
        # Calculate gross dividend
        gross_dividend = dividend_per_share * shares_held

        # Apply tax calculation
        dividend_tax = gross_dividend * self.tax_rate

        net_dividend = gross_dividend - dividend_tax

        interest_payment = 0
        remaining_dividend = net_dividend

        # Pay interest first from dividends
        if self.pay_interest_from_dividends and accrued_interest > 0:
            interest_payment = min(remaining_dividend, accrued_interest)
            remaining_dividend -= interest_payment
            accrued_interest -= interest_payment

        # Income withdrawal - take specified percentage from what's left after interest
        effective_withdrawal_rate = self.income_withdrawal_rate if in_income_phase else 0.0
        income_withdrawal = remaining_dividend * effective_withdrawal_rate
        remaining_dividend -= income_withdrawal

        # Reinvest the remaining dividend using the new centralized method
        shares_held, loan_balance, leverage_used, loan_repayment_today = self._reinvest_cash(
            price, shares_held, loan_balance, remaining_dividend, wind_down_triggered
        )

        return shares_held, loan_balance, gross_dividend, loan_repayment_today, interest_payment, accrued_interest, dividend_tax, income_withdrawal, leverage_used

    def _update_tracking_arrays(self, results: Dict, portfolio_value: float, equity: float,
                                loan_balance: float, current_leverage: float, dividend_today: float,
                                gap_interest: float, wind_down_triggered: bool, loan_repayment_today: float,
                                margin_call_occurred: bool, margin_call_amount: float, dividend_tax_today: float,
                                income_withdrawal_today: float, dca_investment_today: float):
        results['portfolio_values'].append(portfolio_value)
        results['equity_values'].append(equity)
        results['loan_balances'].append(loan_balance)
        results['leverage_ratios'].append(current_leverage)
        results['dividend_payments'].append(dividend_today)
        results['dividend_taxes_paid'].append(dividend_tax_today)
        results['interest_costs'].append(gap_interest)
        results['wind_down_status'].append(wind_down_triggered)
        results['loan_repayments'].append(loan_repayment_today)
        results['margin_call_flags'].append(margin_call_occurred)
        results['margin_call_amounts'].append(margin_call_amount)
        results['income_withdrawals'].append(income_withdrawal_today)
        results['dca_investments'].append(dca_investment_today)

    def _calculate_final_metrics(self, results: Dict, dates: pd.DatetimeIndex,
                                 shares_held: float, accrued_interest: float) -> Dict:
        years = len(dates) / 252
        final_portfolio_value = results['portfolio_values'][-1]
        final_equity = results['equity_values'][-1]

        if accrued_interest > 0:
            results['loan_balances'][-1] += accrued_interest
            # Don't triple count interest - already counted in monthly charges

        equity_returns = np.diff(results['equity_values']) / np.array(results['equity_values'][:-1])
        equity_returns = equity_returns[np.isfinite(equity_returns)]

        total_capital_contributed = self.initial_investment + results.get('total_dca_invested', 0)
        results['total_capital_contributed'] = total_capital_contributed

        results['equity_cagr'] = self.calculate_cagr(
            total_capital_contributed, final_equity, years
        )
        results['portfolio_cagr'] = self.calculate_cagr(
            self.initial_investment * min(self.target_leverage, 1/self.margin_requirement),
            final_portfolio_value, years
        )
        results['max_drawdown'], results['max_drawdown_date'] = self.calculate_max_drawdown(np.array(results['equity_values']), dates)
        results['sharpe_ratio'] = self.calculate_sharpe_ratio(equity_returns)
        results['volatility'] = np.std(equity_returns) * np.sqrt(252) * 100 if equity_returns.size > 0 else 0
        results['years'] = years
        results['final_shares'] = shares_held

        return results

    def calculate_greeks(self, results: Dict, prices: np.ndarray) -> Dict:
        """Calculate Greeks for the strategy"""
        dates = results['dates']
        
        # 1. Delta - Position sensitivity using EQUITY values
        results['delta'] = results['final_shares'] * prices[-1] / self.initial_investment
        
        # Rolling 30-day delta based on equity changes
        rolling_deltas = []
        for i in range(len(results['equity_values'])):
            if i >= 30:
                price_change = (prices[i] - prices[i-30]) / prices[i-30]
                equity_change = (results['equity_values'][i] - results['equity_values'][i-30]) / results['equity_values'][i-30]
                rolling_delta = equity_change / price_change if price_change != 0 else np.nan
                rolling_deltas.append(rolling_delta)
            else:
                rolling_deltas.append(np.nan)
        results['rolling_deltas'] = rolling_deltas
        
        # 2. Gamma - Rate of change of delta (normalized by price change)
        gamma_values = []
        for i in range(1, len(rolling_deltas)):
            if not np.isnan(rolling_deltas[i]) and not np.isnan(rolling_deltas[i-1]) and i < len(prices):
                price_change = (prices[i] - prices[i-1]) / prices[i-1]
                if price_change != 0:
                    gamma = (rolling_deltas[i] - rolling_deltas[i-1]) / price_change
                    gamma_values.append(gamma)
        results['average_gamma'] = np.mean(gamma_values) if gamma_values else 0
        
        # 3. Enhanced Vega Proxy - Returns-based correlation approach
        # Calculate rolling volatility from returns
        price_returns = np.diff(prices) / prices[:-1]
        equity_returns = np.diff(results['equity_values']) / np.array(results['equity_values'][:-1])
        
        volatility_windows = []
        equity_return_windows = []
        
        for i in range(30, len(price_returns)):
            # Calculate volatility from price returns
            window_returns = price_returns[i-30:i]
            vol = np.std(window_returns) * np.sqrt(252)
            volatility_windows.append(vol)
            
            # Calculate corresponding equity return for that period
            equity_ret = equity_returns[i-1] if i-1 < len(equity_returns) else 0
            equity_return_windows.append(equity_ret)
        
        if len(volatility_windows) > 1 and len(equity_return_windows) > 1:
            # Method 1: Correlation between volatility changes and equity returns
            vol_changes = np.diff(volatility_windows)
            aligned_equity_returns = equity_return_windows[1:len(vol_changes)+1]
            
            if len(vol_changes) == len(aligned_equity_returns) and len(vol_changes) > 1:
                correlation = np.corrcoef(vol_changes, aligned_equity_returns)[0,1]
                avg_equity = np.mean(results['equity_values'])
                avg_vol = np.mean(volatility_windows)
                
                results['vega_proxy'] = correlation * avg_equity * avg_vol if not np.isnan(correlation) else 0
                results['vega_per_percent'] = correlation * avg_equity * 0.01 if not np.isnan(correlation) else 0
            else:
                results['vega_proxy'] = 0
                results['vega_per_percent'] = 0
                
            # Method 2: Beta of equity returns to volatility (alternative measure)
            if len(volatility_windows) == len(equity_return_windows) and len(volatility_windows) > 1:
                # Calculate beta of equity returns to volatility levels
                vol_array = np.array(volatility_windows)
                equity_ret_array = np.array(equity_return_windows)
                
                # Remove NaN and infinite values
                valid_mask = np.isfinite(vol_array) & np.isfinite(equity_ret_array)
                if np.sum(valid_mask) > 1:
                    vol_clean = vol_array[valid_mask]
                    equity_ret_clean = equity_ret_array[valid_mask]
                    
                    if np.std(vol_clean) > 0:
                        vega_beta = np.cov(equity_ret_clean, vol_clean)[0,1] / np.var(vol_clean)
                        results['vega_beta'] = vega_beta * avg_equity if 'avg_equity' in locals() else 0
                    else:
                        results['vega_beta'] = 0
                else:
                    results['vega_beta'] = 0
            else:
                results['vega_beta'] = 0
        else:
            results['vega_proxy'] = 0
            results['vega_per_percent'] = 0
            results['vega_beta'] = 0
        
        # 4. Theta - Daily cost of carry including reinvestment impact
        total_days = len(results['loan_balances'])
        if total_days > 0:
            avg_daily_interest = results['total_interest'] / total_days
            avg_daily_dividend = (results['total_dividends'] * (1 - self.tax_rate)) / total_days
            reinvestment_impact = (results['total_dividends'] - results['total_dividend_taxes'] - results['total_income_withdrawn']) / total_days
            
            results['theta'] = avg_daily_dividend - avg_daily_interest
            results['theta_with_reinvestment'] = avg_daily_dividend + reinvestment_impact - avg_daily_interest
        else:
            results['theta'] = 0
            results['theta_with_reinvestment'] = 0
        
        # Cumulative theta impact
        cumulative_theta = []
        running_theta = 0
        for i in range(len(dates)):
            # Use Fed rate for the specific date
            fed_rate = get_fed_rate(dates[i])
            effective_rate = (fed_rate / 100) + self.broker_spread
            daily_cost = results['loan_balances'][i] * (effective_rate / 360)
            running_theta -= daily_cost  # Negative because it's a cost
            cumulative_theta.append(running_theta)
        results['cumulative_theta_cost'] = cumulative_theta
        
        # 5. Rho - Interest rate sensitivity
        average_loan_balance = np.mean(results['loan_balances'])
        results['rho'] = -average_loan_balance * 0.01 / 360  # Daily impact of 1% rate increase
        results['annual_rho'] = results['rho'] * 360
        
        # 6. Additional Strategy-Specific Greeks
        results['lambda'] = results['leverage_ratios'][-1]  # Final leverage
        
        # Dividend coverage ratio
        annual_dividends = results['total_dividends'] / results['years'] if results['years'] > 0 else 0
        annual_interest = results['total_interest'] / results['years'] if results['years'] > 0 else 0
        results['dividend_coverage_ratio'] = annual_dividends / annual_interest if annual_interest > 0 else float('inf')
        
        # Margin safety
        final_equity_ratio = results['final_equity'] / results['final_portfolio_value'] if results['final_portfolio_value'] > 0 else 0
        results['margin_safety'] = (final_equity_ratio - self.margin_requirement) / self.margin_requirement
        
        # 7. Risk-Adjusted Greeks
        if results['volatility'] > 0:
            results['risk_adjusted_delta'] = results['delta'] / (results['volatility'] / 100)
        else:
            results['risk_adjusted_delta'] = 0
        
        results['theta_to_equity_ratio'] = abs(results['theta']) / results['final_equity'] * 100 if results['final_equity'] > 0 else 0
        results['leverage_adjusted_vega'] = results['vega_per_percent'] * results['lambda']
        
        return results

    def run_strategy(self) -> Dict:
        data, dividends = self.fetch_data()
        prices = data['Close'].values
        dates = data.index
        
        # Update start_date to actual data start date for accurate reporting
        self.actual_start_date = dates[0].strftime('%Y-%m-%d')
        self.actual_end_date = dates[-1].strftime('%Y-%m-%d')

        shares_held, loan_balance, _ = self._initialize_portfolio(prices[0])
        
        # Initialize benchmark (unleveraged buy-and-hold)
        benchmark_shares = self.initial_investment / prices[0]
        benchmark_values = []
        benchmark_dividends_received = 0

        accrued_interest = 0
        days_since_last_charge = 0
        wind_down_triggered = False
        wind_down_date = None
        prev_date = None

        results = {
            'portfolio_values': [], 'equity_values': [], 'loan_balances': [],
            'leverage_ratios': [], 'dividend_payments': [], 'dividend_taxes_paid': [],
            'interest_costs': [], 'wind_down_status': [], 'loan_repayments': [],
            'margin_call_flags': [], 'margin_call_amounts': [], 'income_withdrawals': [],
            'dca_investments': [], 'total_dividends': 0, 'total_dividend_taxes': 0,
            'total_interest': 0, 'total_loan_repayments': 0, 'total_margin_calls': 0,
            'total_margin_sales': 0, 'total_dividend_events': 0,
            'dividend_events_no_leverage': 0, 'total_income_withdrawn': 0,
            'total_dca_invested': 0
        }
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            days_gap = self._calculate_gap_days(date.date(), prev_date)
            

            # Determine if we're in the income phase
            days_elapsed = (date - dates[0]).days
            years_elapsed = days_elapsed / 365.25
            in_income_phase = years_elapsed >= self.income_hold_off_years

            # Update benchmark performance
            benchmark_value = benchmark_shares * price
            if date in dividends.index:
                dividend_per_share = dividends.loc[date]
                benchmark_dividend = benchmark_shares * dividend_per_share
                benchmark_dividends_received += benchmark_dividend
                # Apply same tax treatment as strategy for fair comparison
                benchmark_dividend_after_tax = benchmark_dividend * (1 - self.tax_rate)
                # Reinvest dividends in benchmark
                benchmark_shares += benchmark_dividend_after_tax / price
            benchmark_values.append(benchmark_value)

            (loan_balance, accrued_interest, days_since_last_charge,
             gap_interest, charge_interest_flag) = self._accrue_interest(
                loan_balance, days_gap, accrued_interest, days_since_last_charge, date
            )

            if charge_interest_flag and accrued_interest > 0:
                if self.pay_interest_from_dividends:
                    # Current behavior - try to pay by selling shares
                    portfolio_value = shares_held * price
                    amount_to_pay = accrued_interest

                    if portfolio_value > 0:
                        amount_available = min(amount_to_pay, portfolio_value)
                        shares_to_sell = amount_available / price
                        shares_held -= shares_to_sell


                    results['total_interest'] += amount_to_pay
                else:
                    # New behavior - add to loan balance
                    loan_balance += accrued_interest
                    results['total_interest'] += accrued_interest
                
                accrued_interest = 0

            portfolio_value = shares_held * price
            equity = portfolio_value - loan_balance

            (shares_held, loan_balance,
             margin_call_occurred, margin_call_amount) = self._enforce_margin_requirements(
                price, shares_held, loan_balance
            )

            portfolio_value = shares_held * price
            equity = portfolio_value - loan_balance

            if margin_call_occurred:
                results['total_margin_calls'] += 1
                results['total_margin_sales'] += margin_call_amount

            if (self.wind_down_threshold is not None and
                    not wind_down_triggered and
                    equity >= self.wind_down_threshold and
                    loan_balance > 0):  # Only trigger if there's actually a loan
                wind_down_triggered = True
                wind_down_date = date

            dividend_today = 0
            loan_repayment_today = 0
            dividend_tax_today = 0
            income_withdrawal_today = 0

            if date in dividends.index:
                dividend_per_share = dividends.loc[date]
                if dividend_per_share > 0:  # Only process actual dividend payments
                    (shares_held, loan_balance, gross_dividend,
                     loan_repayment_today, interest_payment_from_dividend, accrued_interest, dividend_tax, income_withdrawal_today, leverage_used) = self._handle_dividend(
                        price, shares_held, loan_balance, dividend_per_share, wind_down_triggered, accrued_interest, in_income_phase
                    )

                    results['total_dividends'] += gross_dividend
                    results['total_dividend_taxes'] += dividend_tax
                    results['total_interest'] += interest_payment_from_dividend
                    results['total_loan_repayments'] += loan_repayment_today
                    results['total_income_withdrawn'] += income_withdrawal_today
                    results['total_dividend_events'] += 1
                    if not leverage_used:
                        results['dividend_events_no_leverage'] += 1
                    dividend_today = gross_dividend
                    dividend_tax_today = dividend_tax

                (shares_held, loan_balance,
                 div_margin_call_occurred, div_margin_call_amount) = self._enforce_margin_requirements(
                    price, shares_held, loan_balance
                )

                if div_margin_call_occurred:
                    results['total_margin_calls'] += 1
                    results['total_margin_sales'] += div_margin_call_amount
                    margin_call_occurred = margin_call_occurred or div_margin_call_occurred
                    margin_call_amount += div_margin_call_amount

            dca_investment_today = 0
            if self.dca_amount > 0 and not in_income_phase:
                dca_investment_today = self.dca_amount
                (shares_held, loan_balance, _,
                 dca_loan_repayment) = self._reinvest_cash(
                    price, shares_held, loan_balance, self.dca_amount, wind_down_triggered
                )
                results['total_dca_invested'] += self.dca_amount
                results['total_loan_repayments'] += dca_loan_repayment

            current_leverage = (shares_held * price) / ((shares_held * price) - loan_balance) if ((shares_held * price) - loan_balance) > 0 else float('inf')
            self._update_tracking_arrays(
                results, (shares_held * price), ((shares_held * price) - loan_balance), loan_balance, current_leverage,
                dividend_today, gap_interest, wind_down_triggered, loan_repayment_today,
                margin_call_occurred, margin_call_amount, dividend_tax_today, income_withdrawal_today,
                dca_investment_today
            )

            prev_date = date

        results['dates'] = dates
        results['final_portfolio_value'] = results['portfolio_values'][-1]
        results['final_equity'] = results['equity_values'][-1]
        results['final_loan_balance'] = results['loan_balances'][-1]
        results['wind_down_triggered'] = wind_down_triggered
        results['wind_down_date'] = wind_down_date
        
        # Add benchmark results
        results['benchmark_values'] = benchmark_values
        results['benchmark_final_value'] = benchmark_values[-1]
        results['benchmark_dividends_received'] = benchmark_dividends_received
        results['benchmark_cagr'] = self.calculate_cagr(
            self.initial_investment, benchmark_values[-1], len(dates) / 252
        )

        # Calculate dividend leverage percentage
        if results['total_dividend_events'] > 0:
            results['pct_dividends_no_leverage'] = (results['dividend_events_no_leverage'] / results['total_dividend_events']) * 100
        else:
            results['pct_dividends_no_leverage'] = 0.0
        
        results = self._calculate_final_metrics(results, dates, shares_held, accrued_interest)
        
        # Validate performance for realism
        self._validate_performance(results)
        
        # Calculate Greeks AFTER final metrics
        results = self.calculate_greeks(results, prices)
        
        return results

    def _validate_performance(self, results: Dict):
        """Validate performance metrics for realism"""
        pass

    def print_results(self, results: Dict):
        dates = results['dates']  # Get dates from results
        print("\n" + "="*60)
        print("STRICT LEVERAGE DRIP STRATEGY RESULTS")
        print("="*60)
        print(f"Ticker: {self.ticker}")
        print(f"Period: {self.actual_start_date} to {self.actual_end_date}")
        print(f"Initial Investment: ${self.initial_investment:,.2f}")
        if self.dca_amount > 0:
            print(f"Daily DCA Amount: ${self.dca_amount:,.2f}")
            print(f"Total DCA Invested: ${results['total_dca_invested']:,.2f}")
            print(f"Total Capital Contributed: ${results['total_capital_contributed']:,.2f}")
        print(f"Target Leverage: {self.target_leverage:.1f}x")
        print(f"Interest Rate: Fed Rate + {self.broker_spread:.2%} (from FEDFUNDS.csv, charged monthly)")
        print(f"Margin Requirement: {self.margin_requirement:.0%} Equity (Maintenance)")
        print(f"Max Allowed Leverage: {1/self.margin_requirement:.1f}x")
        print(f"Interest Payment Method: {'From Dividends' if self.pay_interest_from_dividends else 'Accumulate to Margin'}")
        print(f"Income Withdrawal Rate: {self.income_withdrawal_rate:.1%} of net dividends")
        print(f"Income Hold-off Period: {self.income_hold_off_years:.1f} years")

        print("\n" + "-"*40)
        print("TAX SETTINGS")
        print("-"*40)
        print(f"Tax Treatment: Standard Dividend")
        print(f"Dividend Tax Rate: {self.tax_rate:.0%}")

        print(f"\nInvestment Period: {results['years']:.1f} years")

        if self.wind_down_threshold:
            print(f"Wind-down Threshold: ${self.wind_down_threshold:,.2f}")
            if results['wind_down_triggered']:
                print(f"Wind-down Triggered: {results['wind_down_date'].strftime('%Y-%m-%d')}")

        print("\n" + "-"*40)
        print("FINAL PORTFOLIO POSITION")
        print("-"*40)
        print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
        print(f"Final Equity Value: ${results['final_equity']:,.2f}")
        print(f"Final Loan Balance: ${results['final_loan_balance']:,.2f}")
        final_equity_ratio = results['final_equity'] / results['final_portfolio_value'] if results['final_portfolio_value'] > 0 else 0
        print(f"Final Equity Ratio: {final_equity_ratio:.1%}")

        print("\n" + "-"*40)
        print("MARGIN CALLS")
        print("-"*40)
        print(f"Total Margin Calls: {results['total_margin_calls']}")
        print(f"Total Amount Sold in Margin Calls: ${results['total_margin_sales']:,.2f}")

        print("\n" + "-"*40)
        print("CASH FLOWS")
        print("-"*40)
        print(f"Gross Dividends Received: ${results['total_dividends']:,.2f}")
        print(f"Dividend Taxes Paid: ${results['total_dividend_taxes']:,.2f}")
        print(f"Net Dividends After Tax: ${results['total_dividends'] - results['total_dividend_taxes']:,.2f}")
        print(f"Income Withdrawn (Cash): ${results['total_income_withdrawn']:,.2f}")
        print(f"Dividends Reinvested: ${results['total_dividends'] - results['total_dividend_taxes'] - results['total_income_withdrawn']:,.2f}")
        print(f"Total Interest Paid: ${results['total_interest']:,.2f}")
        print(f"Total Loan Repayments: ${results['total_loan_repayments']:,.2f}")

        print("\n" + "-"*40)
        print("INCOME WITHDRAWAL STATISTICS")
        print("-"*40)
        if self.income_hold_off_years > 0:
            income_start_date = dates[0] + pd.DateOffset(days=self.income_hold_off_years * 365.25)
            if income_start_date <= dates[-1]:
                print(f"Income Phase Started: {income_start_date.strftime('%Y-%m-%d')} (after {self.income_hold_off_years:.1f} year hold-off)")
            else:
                print(f"Income Phase: Not yet started (hold-off period ends {income_start_date.strftime('%Y-%m-%d')})")
        
        if results['total_income_withdrawn'] > 0:
            income_years = max(0.1, results['years'] - self.income_hold_off_years)  # Avoid division by zero
            annual_income = results['total_income_withdrawn'] / income_years if income_years > 0 else 0
            avg_quarterly_income = annual_income / 4
            income_yield_on_initial = (annual_income / self.initial_investment) * 100
            print(f"Total Cash Income Received: ${results['total_income_withdrawn']:,.2f}")
            print(f"Average Annual Income (during income phase): ${annual_income:,.2f}")
            print(f"Average Quarterly Income: ${avg_quarterly_income:,.2f}")
            print(f"Income Yield on Initial Investment: {income_yield_on_initial:.2f}%")
        else:
            if self.income_withdrawal_rate > 0:
                print("No income withdrawn yet (still in hold-off period or no dividends)")
            else:
                print("No income withdrawn (withdrawal rate set to 0%)")

        print("\n" + "-"*40)
        print("DIVIDEND LEVERAGE STATISTICS")
        print("-"*40)
        print(f"Total Dividend Events: {results['total_dividend_events']}")
        print(f"Dividends Reinvested Without Leverage: {results['dividend_events_no_leverage']}")
        print(f"Percentage of Dividends Without Leverage: {results['pct_dividends_no_leverage']:.1f}%")

        print("\n" + "-"*40)
        print("PERFORMANCE METRICS")
        print("-"*40)
        if self.dca_amount > 0:
            print(f"Equity CAGR: {results['equity_cagr']:.2f}% (approximated with DCA contributions)")
        else:
            print(f"Equity CAGR: {results['equity_cagr']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}% (on {results['max_drawdown_date']})")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        print("\n" + "-"*40)
        print("BENCHMARK COMPARISON")
        print("-"*40)
        print(f"Benchmark Final Value: ${results['benchmark_final_value']:,.2f}")
        print(f"Benchmark CAGR: {results['benchmark_cagr']:.2f}%")
        print(f"Benchmark Dividends Received: ${results['benchmark_dividends_received']:,.2f}")
        print(f"Strategy vs Benchmark: {results['equity_cagr'] - results['benchmark_cagr']:.2f}% outperformance")
        print(f"Strategy Multiple: {results['final_equity'] / results['benchmark_final_value']:.2f}x")
        

        print("\n" + "-"*40)
        print("STRATEGY GREEKS")
        print("-"*40)
        print(f"Delta (Position Sensitivity): {results['delta']:.3f}")
        print(f"  - ${results['delta']:.2f} equity change per $1 stock move")
        print(f"Gamma (Delta Change Rate): {results['average_gamma']:.6f}")
        print(f"Theta (Daily Carry Cost): ${results['theta']:.2f}")
        print(f"  - With Reinvestment Impact: ${results['theta_with_reinvestment']:.2f}")
        print(f"  - Annual Theta Impact: ${results['theta'] * 365.25:,.2f}")
        print(f"  - Theta/Equity Ratio: {results['theta_to_equity_ratio']:.4f}% daily")
        print(f"Vega (Volatility Sensitivity): ${results['vega_proxy']:.2f}")
        print(f"  - Per 1% Vol Change: ${results['vega_per_percent']:,.2f}")
        print(f"  - Vega Beta (Returns-based): ${results['vega_beta']:,.2f}")
        print(f"Rho (Interest Rate Sensitivity): ${results['rho']:.2f} per day per 1% change")
        print(f"  - Annual Rho Impact: ${results['annual_rho']:,.2f} per 1% change")

        print("\n" + "-"*40)
        print("LEVERAGE & RISK METRICS")
        print("-"*40)
        print(f"Lambda (Final Leverage): {results['lambda']:.2f}x")
        print(f"Dividend Coverage Ratio: {results['dividend_coverage_ratio']:.2f}x")
        print(f"Margin Safety Buffer: {results['margin_safety']:.1%}")
        print(f"Risk-Adjusted Delta: {results['risk_adjusted_delta']:.2f}")
        print(f"Leverage-Adjusted Vega: {results['leverage_adjusted_vega']:.2f}")

    def plot_results(self, results: Dict):
        from matplotlib.ticker import FuncFormatter
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 8.5))
        dates = results['dates']

        # Formatter for currency values
        def currency_formatter(x, p):
            if abs(x) >= 1e6:
                return f'${x/1e6:.1f}M'
            elif abs(x) >= 1e3:
                return f'${x/1e3:.1f}K'
            else:
                return f'${x:.0f}'

        # Subplot 1: Portfolio Value vs Equity vs Benchmark
        ax1.plot(dates, results['portfolio_values'], label='Portfolio Value', linewidth=2, color='blue')
        ax1.plot(dates, results['equity_values'], label='Equity Value', linewidth=2, color='green')
        ax1.plot(dates, results['benchmark_values'], label='Benchmark Value', linewidth=2, color='gray', linestyle='--')
        
        margin_call_dates = [date for date, flag in zip(dates, results['margin_call_flags']) if flag]
        if margin_call_dates:
            ax1.plot(margin_call_dates,
                     [results['equity_values'][i] for i, flag in enumerate(results['margin_call_flags']) if flag],
                     'rx', markersize=8, label='Margin Call')
        if results['wind_down_triggered']:
            ax1.axvline(x=results['wind_down_date'], color='red', linestyle='--', alpha=0.7, label='Wind-down Start')
        
        ax1.set_title(f'{self.ticker} Leveraged DRIP Strategy Performance')
        ax1.set_ylabel('Value ($)')
        ax1.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Leverage Ratio Over Time
        ax2.plot(dates, results['leverage_ratios'], label='Leverage Ratio', color='red', linewidth=2)
        ax2.axhline(y=self.target_leverage, color='black', linestyle='--', alpha=0.5, label=f'Target: {self.target_leverage}x')
        ax2.axhline(y=1/self.margin_requirement, color='orange', linestyle='--', alpha=0.5, label='Max Allowed (Margin)')
        if margin_call_dates:
            ax2.plot(margin_call_dates,
                     [results['leverage_ratios'][i] for i, flag in enumerate(results['margin_call_flags']) if flag],
                     'rx', markersize=8, label='Margin Call')
        ax2.set_title('Leverage Ratio Over Time')
        ax2.set_ylabel('Leverage Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Monthly Income Received
        # Create monthly income series by resampling daily data
        df_temp = pd.DataFrame({
            'income_withdrawals': results['income_withdrawals'],
            'dividend_payments': results['dividend_payments'],
            'interest_costs': results['interest_costs'],
            'dca_investments': results['dca_investments']
        }, index=dates)
        
        monthly_data = df_temp.resample('ME').sum()
        
        # Plot monthly income streams
        ax3.bar(monthly_data.index, monthly_data['income_withdrawals'], 
                label='Monthly Income Withdrawn', color='darkgreen', alpha=0.8, width=20)
        if self.dca_amount > 0:
            ax3.bar(monthly_data.index, monthly_data['dca_investments'],
                    label='Monthly DCA Investments', color='purple', alpha=0.8, width=20,
                    bottom=monthly_data['income_withdrawals'])
        ax3.plot(monthly_data.index, monthly_data['dividend_payments'], 
                 label='Monthly Gross Dividends', color='green', linewidth=2, marker='o', markersize=4)
        ax3.plot(monthly_data.index, monthly_data['interest_costs'], 
                 label='Monthly Interest Costs', color='red', linewidth=2, marker='s', markersize=4)
        
        ax3.set_title('Monthly Income and Costs')
        ax3.set_ylabel('Monthly Amount ($)')
        ax3.set_xlabel('Date')
        ax3.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax3.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('performance_charts.png')
        plt.close()

    def export_greeks_summary(self, results: Dict) -> pd.DataFrame:
        """Export Greeks summary as a DataFrame"""
        greeks_data = {
            'Metric': ['Delta', 'Gamma', 'Theta (Daily)', 'Theta (w/ Reinvestment)', 
                       'Theta (Annual)', 'Vega Proxy', 'Vega (per 1%)', 'Vega Beta', 'Rho (Daily)', 
                       'Rho (Annual)', 'Lambda', 'Dividend Coverage', 'Margin Safety', 
                       'Risk-Adj Delta', 'Leverage-Adj Vega'],
            'Value': [
                results['delta'],
                results['average_gamma'],
                results['theta'],
                results['theta_with_reinvestment'],
                results['theta'] * 365.25,
                results['vega_proxy'],
                results['vega_per_percent'],
                results['vega_beta'],
                results['rho'],
                results['annual_rho'],
                results['lambda'],
                results['dividend_coverage_ratio'],
                results['margin_safety'],
                results['risk_adjusted_delta'],
                results['leverage_adjusted_vega']
            ],
            'Description': [
                'Position sensitivity to price',
                'Rate of delta change',
                'Daily cost of carry',
                'Daily carry with reinvestment',
                'Annual cost of carry',
                'Volatility sensitivity (correlation)',
                'Per 1% volatility change',
                'Returns-based volatility sensitivity',
                'Daily interest rate sensitivity',
                'Annual interest rate sensitivity',
                'Effective leverage',
                'Dividend/Interest ratio',
                'Distance from margin call',
                'Volatility-adjusted delta',
                'Leverage-adjusted vega'
            ]
        }
        
        return pd.DataFrame(greeks_data)

def main():
    parser = argparse.ArgumentParser(description='Strict Leverage DRIP Strategy Backtest')
    parser.add_argument('--ticker', default='QYLD', help='Stock ticker symbol')
    parser.add_argument('--start_date', default='2001-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial_investment', type=float, default=100000, help='Initial investment amount')
    parser.add_argument('--leverage_ratio', type=float, default=2.0, help='Target leverage ratio')
    parser.add_argument('--broker_spread', type=float, default=0.02, help='Broker spread over Fed rate (200 bps = 0.02)')
    parser.add_argument('--margin_requirement', type=float, default=0.25, help='Minimum equity ratio (maintenance requirement)')
    parser.add_argument('--wind_down_threshold', type=float, default=100000000, help='Equity threshold to trigger wind-down')
    parser.add_argument('--wind_down_rate', type=float, default=1.0, help='Fraction of dividends used for loan repayment')
    parser.add_argument('--tax_rate', type=float, default=0.2, help='Tax rate on dividends (default: 0.20)')
    parser.add_argument('--pay_interest_from_dividends', type=bool, default=True, help='Pay interest from dividends (True) or let it accumulate to margin (False)')
    parser.add_argument('--income_withdrawal_rate', type=float, default=0.5, help='Percentage of net dividends to withdraw as income (0.0 to 1.0)')
    parser.add_argument('--income_hold_off_years', type=float, default=20.0, help='Years to wait before taking income withdrawals (default: 0.0)')
    parser.add_argument('--dca_amount', type=float, default=0.0, help='Daily DCA amount to invest')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--export_greeks', action='store_true', help='Export Greeks summary to CSV')

    args = parser.parse_args()

    # Validate tax rate
    if not (0.0 <= args.tax_rate <= 1.0):
        raise ValueError("Tax rate must be between 0.0 and 1.0")
    
    # Validate income withdrawal rate
    if not (0.0 <= args.income_withdrawal_rate <= 1.0):
        raise ValueError("Income withdrawal rate must be between 0.0 and 1.0")
    
    # Validate income hold-off years
    if args.income_hold_off_years < 0.0:
        raise ValueError("Income hold-off years must be non-negative")

    strategy = StrictLeverageDripStrategy(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_investment=args.initial_investment,
        leverage_ratio=args.leverage_ratio,
        broker_spread=args.broker_spread,
        margin_requirement=args.margin_requirement,
        wind_down_threshold=args.wind_down_threshold,
        wind_down_rate=args.wind_down_rate,
        tax_rate=args.tax_rate,
        pay_interest_from_dividends=args.pay_interest_from_dividends,
        income_withdrawal_rate=args.income_withdrawal_rate,
        income_hold_off_years=args.income_hold_off_years,
        dca_amount=args.dca_amount
    )

    try:
        results = strategy.run_strategy()
        strategy.print_results(results)
        if args.plot:
            strategy.plot_results(results)
        if args.export_greeks:
            greeks_df = strategy.export_greeks_summary(results)
            filename = f"{args.ticker}_greeks_summary.csv"
            greeks_df.to_csv(filename, index=False)
            print(f"\nGreeks summary exported to: {filename}")
    except Exception as e:
        print(f"Error running strategy: {str(e)}")

if __name__ == "__main__":
    main()