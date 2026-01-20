import logging
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from app.config import CONFIG

def get_ai_signals():
    """
    Fetches trading signals from the Google Gemini model via Vertex AI.
    """
    logging.info("Fetching trading signals from Gemini AI...")

    try:
        # Initialize Vertex AI. Assumes the environment is authenticated
        # (e.g., running on Cloud Run with a service account).
        # TODO: User needs to provide their PROJECT_ID and REGION in config or env.
        project_id = CONFIG['gcp']['project_id']
        region = CONFIG['gcp']['region']
        vertexai.init(project=project_id, location=region)

        # Load the Gemini model
        # Using gemini-1.5-flash as it's fast and cost-effective
        model = GenerativeModel("gemini-1.5-flash-001")

        # Define the list of assets to analyze. This could be made dynamic in the future.
        assets_to_analyze = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"]

        prompt = f"""
        You are an expert crypto market analyst. Your task is to analyze the current market sentiment
        and short-term price-action indicators for the following crypto assets: {', '.join(assets_to_analyze)}.

        Based on your analysis of publicly available internet data (news, social media sentiment, basic technical indicators),
        provide a trading signal for each asset.

        The output MUST be a valid JSON object. The object should contain a single key "actions",
        which is a list of objects. Each object in the list must have the following three keys:
        1. "asset": The trading pair (e.g., "BTC/USDT").
        2. "signal": Your trading recommendation, which must be one of "BUY", "SELL", or "NEUTRAL".
        3. "confidence": A float between 0.0 and 1.0 representing your confidence in the signal.

        Do not include any other text, explanations, or markdown formatting in your response.
        Your entire response must be only the raw JSON object.

        Example response:
        {{
          "actions": [
            {{
              "asset": "BTC/USDT",
              "signal": "BUY",
              "confidence": 0.85
            }},
            {{
              "asset": "ETH/USDT",
              "signal": "NEUTRAL",
              "confidence": 0.60
            }}
          ]
        }}
        """

        response = model.generate_content(prompt)

        # Clean up the response to extract only the JSON part
        raw_text = response.text.strip().replace('```json', '').replace('```', '').strip()

        logging.info(f"Raw response from AI: {raw_text}")

        # Parse the JSON response
        signals_data = json.loads(raw_text)
        actions = signals_data.get('actions', [])

        logging.info(f"Successfully received and parsed {len(actions)} signals from AI.")
        return actions

    except Exception as e:
        logging.error(f"An error occurred while fetching signals from Gemini AI: {e}")
        # Return an empty list in case of any error to prevent trading on faulty data
        return []

if __name__ == '__main__':
    # For local testing of this module
    # Note: Requires GOOGLE_APPLICATION_CREDENTIALS to be set locally
    signals = get_ai_signals()
    print(signals)
