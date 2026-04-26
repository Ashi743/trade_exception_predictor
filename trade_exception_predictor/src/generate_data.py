"""Synthetic trade data generator."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def generate_synthetic_trades(
    n_samples: int = 1000,
    exception_rate: float = 0.3,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic trading data."""
    np.random.seed(seed)

    # Time range
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(hours=np.random.randint(0, 8760)) for _ in range(n_samples)]

    # Base data
    counterparties = ['Bank_A', 'Bank_B', 'Bank_C', 'Bank_D', 'Bank_E', 'Bank_F', 'Bank_G']
    instruments = ['FX', 'EQUITY', 'BOND']

    data = {
        'trade_id': [f'T{i:05d}' for i in range(1, n_samples + 1)],
        'timestamp': dates,
        'counterparty': np.random.choice(counterparties, n_samples),
        'instrument_type': np.random.choice(instruments, n_samples),
        'notional_amount': np.random.lognormal(mean=13, sigma=1.5, size=n_samples),
        'trade_price': np.random.uniform(95, 155, n_samples),
        'settlement_date': [d + timedelta(days=2) for d in dates],
        'market_volatility': np.random.beta(2, 5, n_samples),
        'counterparty_risk_score': np.random.beta(2, 5, n_samples),
        'execution_speed_ms': np.random.exponential(scale=100, size=n_samples),
        'price_deviation_pct': np.random.exponential(scale=0.3, size=n_samples),
        'trade_size_percentile': np.random.uniform(0, 100, n_samples),
    }

    df = pd.DataFrame(data)

    # Create target with some patterns
    exception_flags = np.zeros(n_samples)

    # High volatility + high risk -> more exceptions
    high_vol_high_risk = (df['market_volatility'] > 0.6) & (df['counterparty_risk_score'] > 0.5)
    exception_flags[high_vol_high_risk] = 1

    # Large price deviations -> more exceptions
    large_deviation = df['price_deviation_pct'] > 0.7
    exception_flags[large_deviation] = 1

    # Slow execution + large trade size -> more exceptions
    slow_large = (df['execution_speed_ms'] > 300) & (df['trade_size_percentile'] > 75)
    exception_flags[slow_large] = 1

    # Random exceptions to reach target rate
    exception_count = int(n_samples * exception_rate)
    current_exceptions = int(exception_flags.sum())
    remaining = exception_count - current_exceptions

    if remaining > 0:
        non_exception_indices = np.where(exception_flags == 0)[0]
        random_exceptions = np.random.choice(non_exception_indices, remaining, replace=False)
        exception_flags[random_exceptions] = 1

    df['is_exception'] = exception_flags.astype(int)

    return df


def save_trades(df: pd.DataFrame, output_path: str = 'data/trades_synthetic.csv') -> None:
    """Save trades to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} trades to {output_path}")
    print(f"Exception rate: {df['is_exception'].mean():.1%}")


if __name__ == '__main__':
    # Generate and save
    df = generate_synthetic_trades(n_samples=1000, exception_rate=0.3)
    save_trades(df)
