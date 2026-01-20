# Product Requirements Document (PRD) - AI Trading Bot

## 1. Introduction
This document details the product requirements for the AI Trading Bot. It expands on the business requirements outlined in the BRD and defines the features and functionality of the system.

## 2. User Personas
- **Bot Operator / Owner:** The individual who owns the bot, provides the API keys, deploys it, and monitors its performance (primarily through notifications).

## 3. Features

### 3.1. Core Trading Engine
- **FR-01: AI Signal Consumption:** The system must connect to a specified AI service endpoint to fetch trading signals.
- **FR-02: Scheduled Execution:** The trading logic must be executed automatically on a fixed schedule (every 5 hours).
- **FR-03: Dynamic Trade Execution:**
    - On a "BUY" signal, the system shall execute a market buy order for the specified asset.
    - On a "SELL" signal, the system shall check for an existing open position for that asset and, if found, execute a market sell order to close it.
    - On a "NEUTRAL" signal, the system shall take no action for that asset.
- **FR-04: Exchange Integration:** The system must integrate with the Binance spot market via their API.
- **FR-05: Dynamic Symbol Handling:** The system must not be hardcoded to a single trading pair and must be able to trade any asset provided by the AI service.

### 3.2. Persistence & State Management
- **FR-06: Redis for Active Positions:** All open trade positions must be stored in a Redis database to ensure they are not lost if the application restarts.
- **FR-07: Redis for Performance Metrics:** Key performance indicators (KPIs) such as total trades, profit/loss, and win rate must be stored in Redis.
- **FR-08: Redis Data Persistence:** The Redis service must be configured to persist its data to disk, ensuring that data survives a full system (e.g., Docker Compose) restart.

### 3.3. Configuration
- **FR-09: Externalized Configuration:** All non-sensitive configuration parameters (e.g., scheduler interval, Redis host) must be stored in a `config.yml` file.
- **FR-10: Secure Credential Management:** All sensitive credentials (API keys, Telegram tokens) must be loaded from environment variables and not stored in the codebase.

### 3.4. Monitoring & Notifications
- **FR-11: Telegram Notifications:** The system must send notifications to a specified Telegram chat for critical events, including:
    - Opening a new position.
    - Closing a position (including profit/loss).
    - Critical errors (e.g., failed to connect to exchange, failed to execute trade).

### 3.5. Deployment & Maintenance
- **FR-12: Dockerization:** The application and its dependencies (like Redis) must be defined in a `docker-compose.yml` file for one-command setup.
- **FR-13: Automated Deployment Script:** A `deploy.sh` script must be provided to automate the deployment process on a standard Linux VM.
- **FR-14: Automated Testing:** The codebase must include a suite of automated tests that can be run to verify functionality and prevent regressions. The test suite must achieve at least 85% code coverage.

## 4. Assumptions and Dependencies
- A functioning external AI service that provides signals via a predictable API endpoint is required.
- The user must provide valid API keys for Binance and Telegram.
- The deployment environment will be a Linux-based Virtual Machine with Docker and Docker Compose installed.
