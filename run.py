import logging
import time
from apscheduler.schedulers.blocking import BlockingScheduler
from app.main import run_trading_cycle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    The main entry point of the application.
    Initializes and starts the scheduler.
    """
    scheduler = BlockingScheduler(timezone="UTC")

    logging.info("Initializing application...")

    # Schedule the main trading cycle to run every 5 hours
    scheduler.add_job(
        run_trading_cycle,
        'interval',
        hours=5,
        id='trading_cycle_job',
        replace_existing=True
    )

    logging.info("Scheduler started. The trading cycle will run every 5 hours.")
    logging.info(f"Next run is scheduled for: {scheduler.get_job('trading_cycle_job').next_run_time}")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logging.info("Scheduler stopped.")

if __name__ == '__main__':
    main()
