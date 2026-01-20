# System Design Document (SDD) - AI Trading Bot

## 1. Introduction
This document provides a detailed technical overview of the AI Trading Bot's architecture and design. It is intended for developers who will maintain and extend the system.

## 2. System Architecture
The system is designed as a containerized application orchestrated by Docker Compose. It consists of two main services: the `bot` application and a `redis` database.

![System Architecture Diagram](https://i.imgur.com/placeholder.png "A real diagram should be created here showing the components")
*(Placeholder for a real architecture diagram)*

### 2.1. Components
- **`run.py` (Scheduler / Entrypoint):**
    - The main entry point for the application.
    - Uses `APScheduler` to create a blocking scheduler.
    - Schedules the `run_trading_cycle` function to run every 5 hours.
- **`app/main.py` (Core Logic):**
    - Contains the `run_trading_cycle` function, which orchestrates the main business logic for one cycle.
    - Contains the `TradeExecution` class, responsible for interacting with the exchange (placing orders, checking prices).
    - Contains the `PerformanceMetrics` class, responsible for tracking P/L and other stats.
- **`app/ai/connector.py` (AI Connector):**
    - A module responsible for communicating with the external AI service.
    - Fetches trading signals. Currently a mock implementation.
- **`app/persistence/redis_client.py` (Redis Client):**
    - A singleton client that abstracts all interactions with the Redis database.
    - Handles serialization and deserialization of data (e.g., JSON).
- **`app/config.py` (Configuration Loader):**
    - Loads configuration from `config/config.yml` and environment variables.
    - Provides a single `CONFIG` object for use throughout the application.
- **`docker-compose.yml` (Orchestrator):**
    - Defines the `bot` and `redis` services.
    - Manages networking between services and persistent volumes.
- **Redis Service:**
    - A standard Redis container used for all data persistence.
    - Configured with a volume to ensure data survives restarts.

## 3. Data Flow
1.  The `APScheduler` in `run.py` triggers the `run_trading_cycle()` function.
2.  The `TradeExecution` class is instantiated. It loads all active positions from Redis.
3.  `check_existing_positions()` is called. For each active position, it fetches the current price from the exchange and checks if any take-profit or stop-loss conditions are met. If so, it executes a "SELL" order.
4.  `get_ai_signals()` is called to fetch new trading recommendations from the AI service.
5.  The system iterates through the signals.
6.  For a "BUY" signal, a trade is executed via the `TradeExecution` class. A new position is created and stored as a hash in Redis.
7.  For a "SELL" signal, the system checks Redis for an active position for that asset. If one exists, a "SELL" order is executed and the position is deleted from Redis.
8.  The `PerformanceMetrics` class is updated with the outcome of any closed trades.
9.  The cycle ends, and the scheduler waits for the next 5-hour interval.

## 4. Data Design
- **Redis:**
    - `active_positions`: A Redis **Hash**.
        - *Key:* `active_positions`
        - *Field:* The order ID of the buy trade (e.g., `'12345'`).
        - *Value:* A JSON string representing the position dictionary (symbol, amount, entry_price, etc.).
    - `performance_metrics`: A Redis **String**.
        - *Key:* `performance_metrics`
        - *Value:* A JSON string representing the metrics dictionary.
    - `trade_history`: A Redis **List**.
        - *Key:* `trade_history`
        - *Value:* Each entry is a JSON string representing a closed trade's details (profit, exit price, etc.).

## 5. Future Improvements & Scalability
- **Error Handling:** The current error handling is basic. A more robust system for retries, especially for exchange API calls, could be implemented (e.g., using a backoff strategy).
- **AI Connector:** The mock connector needs to be replaced with a real implementation, including proper authentication and error handling for the AI service API.
- **Position Sizing:** Position sizing is currently hardcoded to a minimum value for BUYs. This should be replaced with a dynamic calculation based on risk parameters, portfolio value, and signal confidence.
- **Code Modularity:** The `app/main.py` file is large. The `TradeExecution` and `PerformanceMetrics` classes could be moved to their own modules within the `app` directory to improve organization.
- **Testing:** While unit tests for components exist, more comprehensive integration and end-to-end tests could be added.
