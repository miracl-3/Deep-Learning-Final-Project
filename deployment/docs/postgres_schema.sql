-- PostgreSQL schema blueprint for Task 5.3.
-- This file documents how raw stock data, transformed features, predictions,
-- and portfolio recommendations would be stored in production.

CREATE SCHEMA IF NOT EXISTS stock_ml;

CREATE TABLE IF NOT EXISTS stock_ml.raw_stock_prices (
    market TEXT NOT NULL,
    ticker TEXT NOT NULL,
    trade_date DATE NOT NULL,
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC,
    adjusted_close_price NUMERIC,
    volume NUMERIC,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS stock_ml.model_features (
    market TEXT NOT NULL,
    ticker TEXT NOT NULL,
    trade_date DATE NOT NULL,
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC,
    adjusted_close_price NUMERIC,
    volume NUMERIC,
    log_return NUMERIC,
    ma10 NUMERIC,
    ma20 NUMERIC,
    volatility10 NUMERIC,
    transformed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (market, ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS stock_ml.model_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    task_name TEXT NOT NULL,
    market TEXT NOT NULL,
    ticker TEXT NOT NULL,
    prediction_date DATE NOT NULL,
    target_name TEXT NOT NULL,
    predicted_value NUMERIC NOT NULL,
    api_endpoint TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stock_ml.portfolio_recommendations (
    recommendation_id BIGSERIAL PRIMARY KEY,
    risk_profile TEXT NOT NULL,
    ticker TEXT NOT NULL,
    prediction_date DATE,
    current_close NUMERIC,
    predicted_return NUMERIC,
    predicted_price NUMERIC,
    risk_adjusted_score NUMERIC,
    portfolio_ranking_score NUMERIC,
    weight NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stock_ml_predictions_lookup
    ON stock_ml.model_predictions (task_name, market, ticker, prediction_date);

CREATE INDEX IF NOT EXISTS idx_stock_ml_portfolio_profile
    ON stock_ml.portfolio_recommendations (risk_profile, prediction_date);
