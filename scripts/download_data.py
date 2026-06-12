"""Download SPY spot, the live option chain, and FRED rates into DuckDB.

Usage (from the project root):
    py -3.12 scripts/download_data.py [TICKER]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

from vol_surrogate.data.fetchers import fetch_option_chain, fetch_riskfree, fetch_spot
from vol_surrogate.data.store import init_db, upsert_options, upsert_rates, upsert_spot


def main() -> None:
    load_dotenv()
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    db_path = os.getenv("DB_PATH", "data/vol_surrogate.duckdb")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    con = init_db(db_path)
    print(f"Fetching {ticker} spot history…")
    n_spot = upsert_spot(con, fetch_spot(ticker))
    print(f"  {n_spot} spot rows")

    print(f"Fetching {ticker} option chain…")
    n_opt = upsert_options(con, fetch_option_chain(ticker))
    print(f"  {n_opt} option quotes")

    print("Fetching FRED 3m T-bill…")
    rates = fetch_riskfree()
    n_rate = upsert_rates(con, rates)
    print(f"  {n_rate} rate rows (latest {rates.iloc[-1]['rate']:.2%})")
    con.close()
    print(f"Done → {db_path}")


if __name__ == "__main__":
    main()
