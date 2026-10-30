import pandas as pd
import random

food_items = ["Pizza", "Burger", "Biryani", "French Fries", "Sandwich", "Pasta"]

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

data = []

for i in range(1000):
    food = random.choice(food_items)
    day = random.choice(days)
    price = random.randint(80, 300)
    previous_demand = random.randint(20, 200)
    promotion = random.choice([0, 1])
    temperature = random.randint(20, 40)

    demand = (
        previous_demand
        + (30 if promotion == 1 else 0)
        + (10 if day in ["Saturday", "Sunday"] else 0)
        + random.randint(-15, 15)
    )

    data.append([
        food,
        day,
        price,
        previous_demand,
        promotion,
        temperature,
        max(demand, 1)
    ])

df = pd.DataFrame(data, columns=[
    "food_item",
    "day",
    "price",
    "previous_demand",
    "promotion",
    "temperature",
    "demand"
])

df.to_csv("food_demand_dataset.csv", index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print(df.head())
