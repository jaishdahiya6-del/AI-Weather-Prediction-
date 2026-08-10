"""Run the full pipeline end-to-end: train classifier + regressors (LSTM optional, slower).

Usage:
    python run_pipeline.py            # classifier + regressors
    python run_pipeline.py --lstm     # also train the LSTM
"""
import argparse

from src.training.train_classifier import train as train_classifier
from src.training.train_regressor import train as train_regressor
from src.utils.logger import get_logger

log = get_logger("pipeline")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", action="store_true", help="Also train the LSTM model (slower)")
    args = parser.parse_args()

    log.info("STEP 1/3: Training rain classifier")
    train_classifier()

    log.info("STEP 2/3: Training regressors (temperature, humidity, wind_speed, rainfall)")
    train_regressor()

    if args.lstm:
        log.info("STEP 3/3: Training LSTM")
        from src.training.train_lstm import train as train_lstm
        train_lstm()
    else:
        log.info("Skipping LSTM (pass --lstm to include it)")

    log.info("Pipeline complete. Run `streamlit run dashboard/app.py` to view results.")
