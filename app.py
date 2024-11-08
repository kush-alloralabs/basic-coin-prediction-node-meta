import json
from flask import Flask, Response, request
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER
from typing import Dict, Any
from enum import Enum

class Category(Enum):
    CRYPTO_PRICE = "crypto_price"
    CTR = "ctr"
    VIEWABILITY = "viewability"
    BID_PRICE = "bid_price"

class TopicMeta:
    def __init__(self, category: Category, target_variable: str, topic_id: int):
        self.category = category
        self.target_variable = target_variable
        self.topic_id = topic_id
        
    @property
    def data_structure(self) -> Dict[str, Any]:
        if self.category == Category.CRYPTO_PRICE:
            return {"target_variable": self.target_variable, "price": None}
        elif self.category == Category.CTR:
            return {"target_variable": self.target_variable, "ctr": None}
        # ... existing code ...

app = Flask(__name__)


def update_data():
    """Download price data, format data and train model."""
    files = download_data(TOKEN, TRAINING_DAYS, REGION, DATA_PROVIDER)
    format_data(files, DATA_PROVIDER)
    train_model(TIMEFRAME)


@app.route("/inference", methods=['POST'])
def generate_inference():
    """Generate inference for given topic or category/target."""
    data = request.get_json()
    topic_id = data.get('topic_id')
    
    if topic_id is None:
        # Try category-based lookup
        category = data.get('category')
        target_variable = data.get('target_variable')
        
        if not category or not target_variable:
            return Response(
                json.dumps({"error": "Either topic_id or category/target_variable required"}),
                status=400,
                mimetype='application/json'
            )
            
        try:
            category_enum = Category(category)
            topic_id = topic_registry.get_topic_id(category_enum, target_variable)
            if topic_id is None:
                return Response(
                    json.dumps({"error": "Topic not found for category/target"}),
                    status=404,
                    mimetype='application/json'
                )
        except ValueError:
            return Response(
                json.dumps({"error": "Invalid category"}),
                status=400,
                mimetype='application/json'
            )

    try:
        inference = get_inference(topic_id, TIMEFRAME, REGION, DATA_PROVIDER)
        return Response(json.dumps(inference), status=200, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


@app.route("/ground-truth", methods=['POST'])
def submit_ground_truth():
    """Submit ground truth data for a topic."""
    data = request.get_json()
    topic_id = data.get('topic_id')
    value = data.get('value')
    
    if not topic_id or value is None:
        return Response(
            json.dumps({"error": "topic_id and value are required"}),
            status=400,
            mimetype='application/json'
        )
    
    try:
        topic = topic_registry.get_topic_by_id(topic_id)
        if not topic:
            return Response(
                json.dumps({"error": "Topic not found"}),
                status=404,
                mimetype='application/json'
            )
            
        # Store ground truth data
        store_ground_truth(topic, value)
        return Response(json.dumps({"status": "success"}), status=200, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return "0"
    except Exception:
        return "1"


def init_topics():
    """Initialize default topics."""
    # Example topics
    topics = [
        TopicMeta(Category.CRYPTO_PRICE, "ETH", 1),
        TopicMeta(Category.CRYPTO_PRICE, "BTC", 2),
        TopicMeta(Category.CTR, "campaign_1", 3),
        TopicMeta(Category.VIEWABILITY, "publisher_1", 4),
    ]
    
    for topic in topics:
        topic_registry.register_topic(topic)

if __name__ == "__main__":
    init_topics()
    app.run(host="0.0.0.0", port=8000)
