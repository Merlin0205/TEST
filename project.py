
# Customer Segmentation for a Clothing Company

# Objective

# The objective of this project is to build an automated customer segmentation system for a fashion business that operates both online and through physical stores with a loyalty program.

# The system aims to:
# - Segment customers based on their purchasing behavior, demographics, and preferences.
# - Connect this data with inventory information to generate personalized product recommendations.

# Key Goals
# 1. **Customer Segmentation**
#    Group customers into meaningful segments using clustering techniques based on relevant attributes, such as:
#    - Spending habits
#    - Purchase frequency
#    - Product preferences
#    - Engagement metrics

# 2. **Personalized Product Offers**
#    Recommend products tailored to each customer segment by analyzing:
#    - Customer preferences
#    - Inventory data (stock availability and profit margins)

# 3. **Inventory Optimization**
#    Prioritize promoting products that:
#    - Have sufficient stock levels
#    - Provide higher profit margins
#    This ensures maximum revenue while maintaining relevance to the target segments.

# 4. **Scalable Design**
#    Create a solution that:
#    - Can scale to handle larger datasets
#    - Maintains efficiency and adaptability for various marketing **strategies**

# Generate, check and prepare data

# *   K-menas -- dataset name **clustering_dataset_scaled**
# *   Inventory -- dataset name **inventory_dataset**

# Install required libraries
!pip install faker
!pip install gradio
!pip install google-generativeai --upgrade

# Generate dataset 1 "behavioral_dataset" and "behavioral_dataset_noEMAIL"

# The Behavioral Dataset provides information about customer purchasing behavior, serving as the foundation for segmentation. It combines transactional data from the e-shop and loyalty program.
# 
# The following code generates the Behavioral Dataset:

# Importing required libraries
import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Setting the number of customers
num_customers = 2000

# Generate Behavioral Dataset
behavioral_data = {
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

# Display the first few rows of the dataset
behavioral_df.head()

# Save the dataset to a variable for further use
behavioral_dataset = behavioral_df
behavioral_dataset.sample(20)

# Analyze dataset for inconsistencies

# Import required library
import numpy as np

# Generate basic descriptive statistics for numeric columns
print("Descriptive Statistics for Numeric Columns:")
print(behavioral_dataset.describe())

# Check for unique values in categorical columns
print("\nUnique values in 'email':")
print(behavioral_dataset['email'].nunique(), "unique email addresses out of", len(behavioral_dataset))

# Identify invalid email formats
invalid_emails = behavioral_dataset[~behavioral_dataset['email'].str.contains(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', na=False)]
print("\nInvalid email addresses:")
print(invalid_emails)

# Check for negative or zero values in total_spent
negative_spent = behavioral_dataset[behavioral_dataset['total_spent'] < 0]
print("\nRows with negative 'total_spent':")
print(negative_spent)

# Check for customers with zero total_orders but non-zero total_spent
inconsistent_data = behavioral_dataset[(behavioral_dataset['total_orders'] == 0) & (behavioral_dataset['total_spent'] > 0)]
print("\nRows where 'total_orders' == 0 but 'total_spent' > 0:")
print(inconsistent_data)

# Analyze 'categories_bought' and 'brands_bought' for unrealistic values
print("\nAnalysis of 'categories_bought' and 'brands_bought':")
print("Unique values in 'categories_bought':", behavioral_dataset['categories_bought'].unique())
print("Unique values in 'brands_bought':", behavioral_dataset['brands_bought'].unique())

# What to Remove from the Dataset
# Based on the analysis of the dataset, the following data should be removed to ensure consistency and accuracy:

# Rows with invalid email addresses:
# Rows with emails that do not follow a valid email format (e.g., invalid_email.com, user@@example.com) should be removed, as these cannot be used for communication or further analysis.
# Rows with negative total_spent:
# Rows where total_spent is negative should be removed, as negative spending is illogical and likely indicates data entry errors.
# Rows with missing values:
# Rows where any critical fields (total_spent, categories_bought, brands_bought) are missing (NaN) should be removed to maintain dataset integrity. These include customers who registered but had no transactions.

# Remove rows with invalid email addresses
behavioral_dataset = behavioral_dataset[
    behavioral_dataset['email'].str.contains(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', na=False)
]

# Remove rows with negative total_spent
behavioral_dataset = behavioral_dataset[
    behavioral_dataset['total_spent'] >= 0
]

# Remove rows with missing critical values
behavioral_dataset = behavioral_dataset.dropna(
    subset=['total_spent', 'categories_bought', 'brands_bought']
)

# Display summary of cleaned dataset
print("Summary of cleaned dataset:")
print(behavioral_dataset.info())

# Display the first few rows of the cleaned dataset
behavioral_dataset.head()

# Analyze AGAIN dataset for inconsistencies if they were removed

# Import required library
import numpy as np

# Function to check and report issues in the dataset
def check_data_issues(dataset):
    issues_found = False  # Flag to track if any issues are found

    print("=== Dataset Integrity Check ===")

    # Descriptive statistics
    print("\n[INFO] Basic Descriptive Statistics:")
    print(dataset.describe())

    # Check for unique emails
    unique_emails = dataset['email'].nunique()
    total_records = len(dataset)
    print(f"\n[INFO] Unique email addresses: {unique_emails} out of {total_records}")

    # Invalid email formats
    invalid_emails = dataset[~dataset['email'].str.contains(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', na=False)]
    if not invalid_emails.empty:
        issues_found = True
        print("\n[WARNING] Invalid email addresses found:")
        print(invalid_emails)
    else:
        print("\n[INFO] All email addresses are valid.")

    # Negative total_spent
    negative_spent = dataset[dataset['total_spent'] < 0]
    if not negative_spent.empty:
        issues_found = True
        print("\n[WARNING] Rows with negative 'total_spent':")
        print(negative_spent)
    else:
        print("\n[INFO] No negative values in 'total_spent'.")

    # Inconsistencies: zero total_orders with non-zero total_spent
    inconsistent_data = dataset[(dataset['total_orders'] == 0) & (dataset['total_spent'] > 0)]
    if not inconsistent_data.empty:
        issues_found = True
        print("\n[WARNING] Rows where 'total_orders' == 0 but 'total_spent' > 0:")
        print(inconsistent_data)
    else:
        print("\n[INFO] No inconsistencies in 'total_orders' and 'total_spent'.")

    # Check categories_bought and brands_bought for missing values
    if dataset['categories_bought'].isnull().any() or dataset['brands_bought'].isnull().any():
        issues_found = True
        print("\n[WARNING] Missing values found in 'categories_bought' or 'brands_bought'.")
    else:
        print("\n[INFO] No missing values in 'categories_bought' or 'brands_bought'.")

    # Final report
    if issues_found:
        print("\n[RESULT] Issues detected in the dataset. Please review the warnings above.")
    else:
        print("\n[RESULT] All data checks passed. Dataset is clean and ready for analysis.")

# Run the function to check the dataset
check_data_issues(behavioral_dataset)

# In this step, we remove the email column from the dataset to ensure customer privacy and focus on anonymized data analysis. The resulting dataset, named behavioral_dataset_noEMAIL, retains all other information.

# Create a new dataset without the email column
behavioral_dataset_noEMAIL = behavioral_dataset.drop(columns=['email'])

# Display the first few rows of the new dataset
behavioral_dataset_noEMAIL.head(20)

# Generate dataset 2 "Preference_dataset"

# The Preference Dataset focuses on customer purchasing preferences, capturing insights such as favorite product categories, brands, and price ranges. This dataset will provide valuable information for personalized product recommendations.

# Generate dataset

# Import required libraries
import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Categories and Brands for the clothing and accessories e-shop
CATEGORIES = ["Tops", "Bottoms", "Dresses", "Outerwear", "Shoes", "Accessories", "Sportswear"]
BRANDS = [
    "Nike", "Adidas", "Puma", "Zara", "H&M", "Gucci", "Prada", "Levi's", "Ralph Lauren", "Under Armour",
    "Calvin Klein", "New Balance", "Tommy Hilfiger", "Versace", "Burberry"
]

# Number of customers (same as in Behavioral Dataset)
num_customers = 2000

# Generate Preference Dataset
preference_data = {
    "customer_id": [i for i in range(1, num_customers + 1)],  # Unique customer IDs
    "top_category": [random.choice(CATEGORIES) for _ in range(num_customers)],  # Most frequent category
    "top_brand": [random.choice(BRANDS) for _ in range(num_customers)],  # Most frequent brand
    "price_preference_range": [random.randint(1, 3) for _ in range(num_customers)],  # 1 = Low, 2 = Mid, 3 = High
    "discount_sensitivity": [round(random.uniform(0.0, 1.0), 2) for _ in range(num_customers)],  # Sensitivity to discounts
    "luxury_preference_score": [random.randint(1, 5) for _ in range(num_customers)]  # Preference for luxury (1-5)
}

# Convert to DataFrame
preference_df = pd.DataFrame(preference_data)

# Save the dataset to a variable for further use
preference_dataset_names = preference_df

# Display the first few rows of the dataset
preference_dataset_names.sample(20)

# This code converts categorical features (top category and brand) in the preference dataset to numerical IDs for use in the K-means clustering algorithm, which requires numerical input.

from IPython.display import display

# Create mapping tables for 'top_category' and 'top_brand'
category_mapping = {category: idx for idx, category in enumerate(CATEGORIES)}
brand_mapping = {brand: idx for idx, brand in enumerate(BRANDS)}

# Save the mapping tables to DataFrames for future use
category_mapping_df = pd.DataFrame(list(category_mapping.items()), columns=["category_name", "category_id"])
brand_mapping_df = pd.DataFrame(list(brand_mapping.items()), columns=["brand_name", "brand_id"])

# Display the mapping tables as tables
print("Category Mapping Table:")
display(category_mapping_df)

print("\nBrand Mapping Table:")
display(brand_mapping_df)

# Convert 'top_category' and 'top_brand' to numeric values in preference_dataset_names
preference_dataset = preference_dataset_names.copy()  # Create a copy to store the results
preference_dataset["top_category"] = preference_dataset_names["top_category"].map(category_mapping)
preference_dataset["top_brand"] = preference_dataset_names["top_brand"].map(brand_mapping)

# Display 20 random rows of the updated dataset as a table
print("\nUpdated Preference Dataset with Numeric Values (Sample of 20 rows):")
display(preference_dataset.sample(20))

# Generate dataset 3 "Inventory_dataset"

# The Inventory Dataset contains detailed information about the products available in stock for an e-commerce clothing and accessories store. It is logically connected to the categories and brands used in customer datasets.

# Import required libraries
import pandas as pd
import random

# Categories, Brands, Colors, and Adjectives
CATEGORIES = ["Tops", "Bottoms", "Dresses", "Outerwear", "Shoes", "Accessories", "Sportswear"]
BRANDS = [
    "Nike", "Adidas", "Puma", "Zara", "H&M", "Gucci", "Prada", "Levi's", "Ralph Lauren", "Under Armour",
    "Calvin Klein", "New Balance", "Tommy Hilfiger", "Versace", "Burberry"
]
ADJECTIVES = ["Classic", "Modern", "Stylish", "Luxury", "Casual", "Comfortable", "Premium"]
COLORS = ["Red", "Blue", "Black", "White", "Green", "Beige", "Pink", "Grey"]

# Number of products
num_products = 1000

# Generate unique product names
unique_product_names = set()

def generate_unique_product_name():
    """Generates a unique product name."""
    while True:
        brand = random.choice(BRANDS)
        category = random.choice(CATEGORIES)
        adjective = random.choice(ADJECTIVES)
        color = random.choice(COLORS)
        product_name = f"{brand} {color} {adjective} {category}"
        if product_name not in unique_product_names:
            unique_product_names.add(product_name)
            return product_name

# Generate Inventory Dataset
inventory_data = {
    "product_id": [i for i in range(1, num_products + 1)],
    "product_name": [generate_unique_product_name() for _ in range(num_products)],
    "category": [],
    "brand": [],
    "stock_quantity": [random.randint(0, 100) for _ in range(num_products)],
    "retail_price": [round(random.uniform(300, 5000), 2) for _ in range(num_products)],
    "cost_price": [],
    "profit_margin": []
}

# Populate category and brand based on product_name
for product_name in inventory_data["product_name"]:
    split_name = product_name.split(" ")
    inventory_data["brand"].append(split_name[0])
    inventory_data["category"].append(split_name[-1])

# Calculate cost_price and profit_margin
for i in range(num_products):
    retail_price = inventory_data["retail_price"][i]
    profit_margin = round(random.uniform(50, 100), 2) / 100
    cost_price = round(retail_price * (1 - profit_margin), 2)
    inventory_data["cost_price"].append(cost_price)
    inventory_data["profit_margin"].append(round(profit_margin * 100, 2))

# Convert to DataFrame
inventory_df = pd.DataFrame(inventory_data)

# Save the dataset to a variable for further use
inventory_dataset = inventory_df

# Display the first few rows of the dataset
inventory_dataset.sample(20)

# Merging Preference and Behavioral Datasets

# combined_dataset_scaled is the final dataset for segmentation
# Data scaling is essential before applying K-means clustering because the algorithm relies on Euclidean distance to measure the similarity between data points. Features with larger numerical ranges can dominate the clustering process, so scaling ensures all features contribute equally.

# Merge preference_dataset and behavioral_dataset_noEMAIL on 'customer_id'
combined_dataset = pd.merge(preference_dataset, behavioral_dataset_noEMAIL, on="customer_id", how="inner")

# Display 20 random rows from the merged dataset
print("\nCombined Dataset (Sample of 20 rows):")
combined_dataset.sample(20)

# This code prepares a dataset for clustering by performing several key preprocessing steps. It begins by checking for missing values and removing incomplete rows to ensure data integrity. Next, it standardizes the numeric columns to ensure all features contribute equally to the clustering process. Finally, it combines the standardized data with customer IDs for further analysis.

# Import required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Check the dataset structure
#print("Combined Dataset Overview:")
#print(combined_dataset.info())

# Step 2: Check for missing values in each
