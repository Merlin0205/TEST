def generate_behavioral_data(num_customers=2000):
    """
    Generates a behavioral dataset with customer purchase information, including deliberate errors.

    Args:
        num_customers (int, optional): The number of customers to generate data for. Defaults to 2000.

    Returns:
        pandas.DataFrame: The generated behavioral dataset as a Pandas DataFrame.
    """

    # Importing required libraries
    import pandas as pd
    import random
    from faker import Faker

    # Initialize Faker
    fake = Faker()

    # Setting the number of customers
    num_customers = 2000

    #behavioral_data = {
        "customer_id": [i for i in range(1, num_customers + 1)],  # Unique customer IDs
        "email": [fake.email() for _ in range(num_customers)],  # Realistic email addresses
        "total_spent": [round(random.uniform(50, 5000), 2) for _ in range(num_customers)],  # Random total spending
        "total_orders": [random.randint(1, 50) for _ in range(num_customers)],  # Number of orders placed
        "avg_order_value": [],  # To be calculated based on total_spent and total_orders
        "last_purchase_days_ago": [random.randint(0, 365) for _ in range(num_customers)],  # Days since last purchase
        "categories_bought": [random.randint(1, 6) for _ in range(num_customers)],  # Number of unique categories
        "brands_bought": [random.randint(1, 6) for _ in range(num_customers)],  # Number of unique brands
    }

    # Calculate avg_order_value and add errors deliberately
    for i in range(num_customers):
        if i % 50 == 0:  # Every 50th row will have a missing avg_order_value
            behavioral_data["avg_order_value"].append(None)
        else:
            total_orders = behavioral_data["total_orders"][i]
            total_spent = behavioral_data["total_spent"][i]
            avg_value = total_spent / total_orders if total_orders > 0 else 0
            behavioral_data["avg_order_value"].append(round(avg_value, 2))

    # Introduce specific errors into the dataset
    for i in range(20):  # Add invalid email addresses for the first 20 customers
        behavioral_data["email"][i] = "invalid_email.com" if i % 2 == 0 else "user@@example.com"

    for i in range(2):  # Add negative total_spent for 2 customers
        behavioral_data["total_spent"][i] = -random.uniform(100, 500)

    for i in range(num_customers - 70, num_customers):  # Customers with no purchase data
        behavioral_data["total_spent"][i] = None
        behavioral_data["total_orders"][i] = 0
        behavioral_data["avg_order_value"][i] = None
        behavioral_data["categories_bought"][i] = None
        behavioral_data["brands_bought"][i] = None
        behavioral_data["last_purchase_days_ago"][i] = None

    # Convert to DataFrame
    behavioral_df = pd.DataFrame(behavioral_data)

    return behavioral_df Generate Behavioral Dataset
    