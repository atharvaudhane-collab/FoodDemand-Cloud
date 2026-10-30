from flask import Flask, request, render_template_string
import joblib
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

model = joblib.load("food_demand_model.pkl")
HISTORY_FILE = "history.csv"


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>FoodDemand Cloud</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f7f6;
            margin: 0;
            color: #222;
        }

        header {
            background: linear-gradient(135deg, #1b5e20, #43a047);
            color: white;
            padding: 28px;
            text-align: center;
        }

        header h1 {
            margin: 0;
            font-size: 32px;
        }

        header p {
            margin: 8px 0 0;
        }

        .container {
            max-width: 1100px;
            margin: 25px auto;
            padding: 0 20px;
        }

        .card {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 14px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }

        h2 {
            color: #1b5e20;
        }

        form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        label {
            font-weight: bold;
        }

        input, select {
            width: 100%;
            padding: 11px;
            margin-top: 6px;
            border: 1px solid #ccc;
            border-radius: 7px;
            box-sizing: border-box;
        }

        button {
            grid-column: 1 / -1;
            padding: 14px;
            background: #ff9800;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #f57c00;
        }

        .result {
            text-align: center;
            background: #fff8e1;
            border-left: 6px solid #ff9800;
        }

        .prediction {
            font-size: 42px;
            font-weight: bold;
            color: #e65100;
        }

        .demand-level {
            font-size: 22px;
            font-weight: bold;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }

        .stat {
            background: #e8f5e9;
            padding: 20px;
            text-align: center;
            border-radius: 12px;
        }

        .stat h3 {
            margin: 5px;
            color: #1b5e20;
        }

        .stat p {
            font-size: 25px;
            font-weight: bold;
            margin: 8px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 14px;
        }

        th {
            background: #1b5e20;
            color: white;
            padding: 10px;
        }

        td {
            padding: 9px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }

        tr:hover {
            background: #f1f8e9;
        }

        .chart {
            margin-top: 15px;
        }

        .bar-row {
            display: flex;
            align-items: center;
            margin: 13px 0;
        }

        .bar-label {
            width: 120px;
            font-weight: bold;
        }

        .bar-container {
            flex: 1;
            background: #e0e0e0;
            height: 28px;
            border-radius: 15px;
            overflow: hidden;
        }

        .bar {
            height: 100%;
            background: linear-gradient(90deg, #66bb6a, #2e7d32);
            border-radius: 15px;
        }

        .bar-value {
            width: 60px;
            text-align: right;
            font-weight: bold;
        }

        .architecture {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        .arch-box {
            padding: 15px 20px;
            background: #e8f5e9;
            border: 2px solid #43a047;
            border-radius: 10px;
            text-align: center;
            font-weight: bold;
        }

        .arrow {
            font-size: 25px;
            color: #ff9800;
        }

        .info {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }

        .info-box {
            padding: 20px;
            border-radius: 12px;
            background: #fff3e0;
            text-align: center;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: #666;
        }

        @media(max-width: 700px) {
            form, .stats, .info {
                grid-template-columns: 1fr;
            }

            .bar-label {
                width: 90px;
            }
        }
    </style>
</head>

<body>

<header>
    <h1>🍕 FoodDemand Cloud ☁️</h1>
    <p>Cloud-Based Food Order Demand Prediction System</p>
</header>

<div class="container">

    <div class="card">
        <h2>🔮 Predict Food Demand</h2>

        <form method="POST">

            <div>
                <label>Food Item</label>
                <select name="food_item">
                    <option>Pizza</option>
                    <option>Burger</option>
                    <option>Biryani</option>
                    <option>French Fries</option>
                    <option>Sandwich</option>
                    <option>Pasta</option>
                </select>
            </div>

            <div>
                <label>Day</label>
                <select name="day">
                    <option>Monday</option>
                    <option>Tuesday</option>
                    <option>Wednesday</option>
                    <option>Thursday</option>
                    <option>Friday</option>
                    <option>Saturday</option>
                    <option>Sunday</option>
                </select>
            </div>

            <div>
                <label>Price (₹)</label>
                <input type="number" name="price" required>
            </div>

            <div>
                <label>Previous Demand</label>
                <input type="number" name="previous_demand" required>
            </div>

            <div>
                <label>Promotion</label>
                <select name="promotion">
                    <option value="0">No Promotion</option>
                    <option value="1">Promotion Active</option>
                </select>
            </div>

            <div>
                <label>Temperature (°C)</label>
                <input type="number" name="temperature" required>
            </div>

            <button type="submit">🚀 Predict Demand</button>

        </form>
    </div>

    {% if prediction is not none %}
    <div class="card result">
        <h2>Prediction Result</h2>

        <div class="prediction">
            {{ prediction }} Orders
        </div>

        <div class="demand-level">
            {{ demand_level }}
        </div>
    </div>
    {% endif %}


    <div class="card">
        <h2>📊 Analytics Dashboard</h2>

        <div class="stats">

            <div class="stat">
                <h3>Total Predictions</h3>
                <p>{{ analytics.total }}</p>
            </div>

            <div class="stat">
                <h3>Average Demand</h3>
                <p>{{ analytics.average }}</p>
            </div>

            <div class="stat">
                <h3>Highest Demand</h3>
                <p>{{ analytics.highest }}</p>
            </div>

            <div class="stat">
                <h3>Top Food</h3>
                <p>{{ analytics.food }}</p>
            </div>

        </div>
    </div>


    <div class="card">
        <h2>📈 Food Demand Analytics Chart</h2>
        <p>Average predicted demand for each food item.</p>

        <div class="chart">

        {% for item in food_chart %}

            <div class="bar-row">

                <div class="bar-label">
                    {{ item.food }}
                </div>

                <div class="bar-container">
                    <div class="bar"
                         style="width: {{ item.width }}%;">
                    </div>
                </div>

                <div class="bar-value">
                    {{ item.demand }}
                </div>

            </div>

        {% endfor %}

        </div>
    </div>


    <div class="card">
        <h2>🕒 Recent Prediction History</h2>

        {% if history %}

        <table>
            <tr>
                <th>Food</th>
                <th>Day</th>
                <th>Price</th>
                <th>Previous Demand</th>
                <th>Promotion</th>
                <th>Temperature</th>
                <th>Predicted Demand</th>
                <th>Time</th>
            </tr>

            {% for row in history %}

            <tr>
                <td>{{ row.food_item }}</td>
                <td>{{ row.day }}</td>
                <td>{{ row.price }}</td>
                <td>{{ row.previous_demand }}</td>
                <td>{{ row.promotion }}</td>
                <td>{{ row.temperature }}</td>
                <td><b>{{ row.predicted_demand }}</b></td>
                <td>{{ row.time }}</td>
            </tr>

            {% endfor %}

        </table>

        {% else %}
            <p>No prediction history available.</p>
        {% endif %}

    </div>


    <div class="card">
        <h2>☁️ Cloud System Architecture</h2>

        <div class="architecture">

            <div class="arch-box">
                👤 User
            </div>

            <div class="arrow">→</div>

            <div class="arch-box">
                🌐 Internet
            </div>

            <div class="arrow">→</div>

            <div class="arch-box">
                ☁️ AWS EC2
            </div>

            <div class="arrow">→</div>

            <div class="arch-box">
                🐍 Flask App
            </div>

            <div class="arrow">→</div>

            <div class="arch-box">
                🤖 Random Forest
            </div>

            <div class="arrow">→</div>

            <div class="arch-box">
                📊 Prediction
            </div>

        </div>

    </div>


    <div class="card">
        <h2>💡 Project Information</h2>

        <div class="info">

            <div class="info-box">
                <h3>☁️ AWS Cloud</h3>
                <p>Hosted on Amazon EC2</p>
            </div>

            <div class="info-box">
                <h3>🤖 Machine Learning</h3>
                <p>Random Forest Model</p>
            </div>

            <div class="info-box">
                <h3>📊 Dataset</h3>
                <p>1000 Food Order Records</p>
            </div>

        </div>
    </div>

</div>

<footer>
    FoodDemand Cloud | Cloud-Based Food Order Demand Prediction System
</footer>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    demand_level = ""

    if request.method == "POST":

        food_item = request.form["food_item"]
        day = request.form["day"]
        price = float(request.form["price"])
        previous_demand = float(request.form["previous_demand"])
        promotion = int(request.form["promotion"])
        temperature = float(request.form["temperature"])

        input_data = pd.DataFrame([{
            "food_item": food_item,
            "day": day,
            "price": price,
            "previous_demand": previous_demand,
            "promotion": promotion,
            "temperature": temperature
        }])

        prediction = model.predict(input_data)[0]
        prediction = round(prediction)

        if prediction < 80:
            demand_level = "🟢 Low Demand"
        elif prediction < 150:
            demand_level = "🟡 Medium Demand"
        else:
            demand_level = "🔴 High Demand"

        new_record = pd.DataFrame([{
            "food_item": food_item,
            "day": day,
            "price": price,
            "previous_demand": previous_demand,
            "promotion": promotion,
            "temperature": temperature,
            "predicted_demand": prediction,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
        }])

        if os.path.exists(HISTORY_FILE):
            new_record.to_csv(
                HISTORY_FILE,
                mode="a",
                header=False,
                index=False
            )
        else:
            new_record.to_csv(
                HISTORY_FILE,
                index=False
            )


    analytics = {
        "total": 0,
        "average": 0,
        "highest": 0,
        "food": "-"
    }

    history = []
    food_chart = []

    if os.path.exists(HISTORY_FILE):

        history_df = pd.read_csv(HISTORY_FILE)

        history_df = history_df.dropna(
            subset=["predicted_demand"]
        )

        if len(history_df) > 0:

            analytics = {
                "total": len(history_df),
                "average": round(
                    history_df["predicted_demand"].mean()
                ),
                "highest": int(
                    history_df["predicted_demand"].max()
                ),
                "food": history_df.loc[
                    history_df["predicted_demand"].idxmax(),
                    "food_item"
                ]
            }

            chart_data = (
                history_df
                .groupby("food_item")["predicted_demand"]
                .mean()
                .round()
                .sort_values(ascending=False)
            )

            max_demand = chart_data.max()

            for food, demand in chart_data.items():

                width = round(
                    (demand / max_demand) * 100
                ) if max_demand > 0 else 0

                food_chart.append({
                    "food": food,
                    "demand": int(demand),
                    "width": width
                })

            history_df = history_df.tail(10)

            history = history_df.to_dict(
                orient="records"
            )


    return render_template_string(
        HTML,
        prediction=prediction,
        demand_level=demand_level,
        history=history,
        analytics=analytics,
        food_chart=food_chart
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
