-- sql/schema.sql
-- ──────────────────────────────────────────────────────────────────────
-- PURPOSE:  Create the PostgreSQL database schema for the BVMT platform.
--
-- WHAT IS A SCHEMA?
--   A schema is the "blueprint" of your database — it defines which
--   tables exist, what columns each table has, and the data types.
--
-- HOW TO RUN THIS FILE:
--   Option 1 (psql CLI):
--       psql -U postgres -d bvmt -f sql/schema.sql
--
--   Option 2 (pgAdmin):
--       Open pgAdmin → select the "bvmt" database → Query Tool →
--       paste this file → Execute (F5).
--
-- TABLES EXPLAINED:
--
--   stocks
--     Master list of BVMT-listed companies.
--     One row per company (e.g. PGH, BNA, SFBT).
--
--   prices
--     Daily OHLCV (Open, High, Low, Close, Volume) data.
--     One row per trading day per stock.
--     The foreign key (stock_id) links back to the stocks table.
--
--   predictions
--     Model output — one row per forecast.
--     Stores which model produced it, the predicted price, and the
--     date range it covers.
--
--   news
--     News headlines with sentiment scores.
--     Used by the NewsAgent for sentiment analysis.
-- ──────────────────────────────────────────────────────────────────────

-- Create the database (run this separately if needed)
-- CREATE DATABASE bvmt;

-- ── Table: stocks ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stocks (
    id          SERIAL PRIMARY KEY,       -- auto-incrementing ID
    ticker      VARCHAR(10) NOT NULL UNIQUE,  -- e.g. 'PGH'
    name        VARCHAR(200) NOT NULL,    -- e.g. 'Poulina Group Holding'
    sector      VARCHAR(100),             -- e.g. 'Agriculture & Food'
    created_at  TIMESTAMP DEFAULT NOW()
);

-- ── Table: prices ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prices (
    id          SERIAL PRIMARY KEY,
    stock_id    INTEGER REFERENCES stocks(id) ON DELETE CASCADE,
    date        DATE NOT NULL,
    open        NUMERIC(12, 4),           -- opening price
    high        NUMERIC(12, 4),           -- highest price of the day
    low         NUMERIC(12, 4),           -- lowest price of the day
    close       NUMERIC(12, 4) NOT NULL,  -- closing price (most important)
    volume      BIGINT,                   -- number of shares traded
    created_at  TIMESTAMP DEFAULT NOW(),

    -- Prevent duplicate rows for the same stock on the same day
    UNIQUE (stock_id, date)
);

-- ── Table: predictions ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    stock_id        INTEGER REFERENCES stocks(id) ON DELETE CASCADE,
    model_name      VARCHAR(50) NOT NULL,     -- e.g. 'TFT_v1'
    predicted_date  DATE NOT NULL,            -- the date being forecasted
    predicted_close NUMERIC(12, 4) NOT NULL,  -- the forecasted closing price
    confidence      NUMERIC(5, 4),            -- optional: model confidence 0-1
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ── Table: news ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS news (
    id              SERIAL PRIMARY KEY,
    stock_id        INTEGER REFERENCES stocks(id) ON DELETE CASCADE,
    headline        TEXT NOT NULL,
    source          VARCHAR(200),             -- e.g. 'TAP', 'Business News'
    published_at    TIMESTAMP,
    sentiment_score NUMERIC(5, 4),            -- -1.0 (neg) to +1.0 (pos)
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ── Indexes for faster queries ─────────────────────────────────────
-- Speed up "get all prices for stock X sorted by date"
CREATE INDEX IF NOT EXISTS idx_prices_stock_date
    ON prices (stock_id, date);

-- Speed up "get latest news for stock X"
CREATE INDEX IF NOT EXISTS idx_news_stock_published
    ON news (stock_id, published_at DESC);

-- ── Seed data: example BVMT tickers ───────────────────────────────
-- You can remove or modify these. They are just examples.
INSERT INTO stocks (ticker, name, sector) VALUES
    ('PGH',  'Poulina Group Holding',     'Agriculture & Food'),
    ('BNA',  'Banque Nationale Agricole',  'Banking'),
    ('SFBT', 'Sté Frigorifique et Brasserie de Tunis', 'Food & Beverage'),
    ('BIAT', 'Banque Internationale Arabe de Tunisie', 'Banking'),
    ('STB',  'Société Tunisienne de Banque', 'Banking')
ON CONFLICT (ticker) DO NOTHING;  -- don't fail if they already exist
