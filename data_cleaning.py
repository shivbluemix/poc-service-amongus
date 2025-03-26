import pandas as pd
import random
import string

df = pd.read_csv("data/data_with_duplicates.csv")

# Define possible values for roles, permissions, emails, company names, and phone numbers
# roles = ["Admin", "User", "Manager", "Guest"]
# permissions = ["Read", "Write", "Execute", "Delete"]
# company_names = [f"Company {''.join(random.choices(string.ascii_uppercase, k=1))}{i}" for i in range(1, 26)]

# # Generate 25 random email addresses
# domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com"]
# emails = [f"user{i}@{random.choice(domains)}" for i in range(1, 26)]

# # Create a new DataFrame to store the original and duplicate rows
# new_rows = []

# # Generate duplicate rows
# for _, row in df.iterrows():
#     # Add the original row
#     new_rows.append(row.to_dict())

#     # Generate a random number of duplicates (e.g., 1 to 3 duplicates)
#     num_duplicates = random.randint(0, 7)
#     for _ in range(num_duplicates):
#         duplicate_row = row.to_dict()
#         # Keep the first and last name the same, but randomize other columns
#         duplicate_row["role"] = random.choice(roles)
#         duplicate_row["permission"] = random.choice(permissions)
#         duplicate_row["email"] = random.choice(emails)
#         duplicate_row["company_name"] = random.choice(company_names)
#         new_rows.append(duplicate_row)

# # Create a new DataFrame with the original and duplicate rows
# new_df = pd.DataFrame(new_rows)

# # Save the updated DataFrame to a new CSV file
# new_df.to_csv("data/data_with_duplicates.csv", index=False)

# Print the first few rows of the new DataFrame
# Load the CSV file
# df = pd.read_csv("data/data_with_duplicates.csv")

# Concatenate first_name and last_name into a new column full_name
df["full_name"] = df["first_name"] + " " + df["last_name"]

# Drop the original first_name and last_name columns
df.drop(columns=["first_name", "last_name"], inplace=True)

# Print the updated DataFrame
print(df.head(10))

# Save the updated DataFrame to a new CSV file
df.to_csv("data/data_with_duplicates.csv", index=False)

# Check for NaN values
nan_count = df.isna().sum()
print(nan_count)
