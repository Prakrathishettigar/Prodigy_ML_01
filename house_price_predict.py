import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X_train = train_data[features].fillna(train_data[features].median())
y_train = train_data[target].fillna(train_data[target].median())

model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
train_pred = model.predict(X_train)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

rmse = np.sqrt(mean_squared_error(y_train, train_pred))
mae = mean_absolute_error(y_train, train_pred)
r2 = r2_score(y_train, train_pred)
mean_price = y_train.mean()
median_price = y_train.median()

print("Model trained successfully!")
print(f"\nModel Performance Metrics:")
print(f"RMSE: ${rmse:,.0f}")
print(f"MAE: ${mae:,.0f}")
print(f"R² Score: {r2:.3f}")
print(f"Mean Price: ${mean_price:,.0f}")
print(f"Median Price: ${median_price:,.0f}")
print(f"Training samples: {X_train.shape[0]}")

# Feature statistics
print(f"\nFeature Statistics:")
for feature in features:
    print(f"{feature}: Mean={X_train[feature].mean():.1f}, Median={X_train[feature].median():.1f}, Min={X_train[feature].min()}, Max={X_train[feature].max()}")

# User input function
def predict_house_price():
    print("\nEnter house details:")
    
    living_area = float(input("Ground Living Area (sq ft): "))
    bedrooms = int(input("Number of bedrooms: "))
    bathrooms = int(input("Number of full bathrooms: "))
    
    # Make prediction
    user_data = pd.DataFrame({
        'GrLivArea': [living_area],
        'BedroomAbvGr': [bedrooms], 
        'FullBath': [bathrooms]
    })
    predicted_price = model.predict(user_data)[0]
    
    print(f"\nPredicted House Price: ${predicted_price:,.0f}")
    return predicted_price

# Generate graphs
def show_graphs():
    # Model predictions for graph
    train_pred = model.predict(X_train)
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_train, train_pred, alpha=0.6)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted')
    
    # 2. Residuals
    plt.subplot(2, 2, 2)
    residuals = y_train - train_pred
    plt.scatter(train_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    
    # 3. Feature correlations
    plt.subplot(2, 2, 3)
    corr_data = train_data[features + [target]].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    
    # 4. Price distribution
    plt.subplot(2, 2, 4)
    plt.hist(y_train, bins=30, alpha=0.7)
    plt.xlabel('Sale Price')
    plt.ylabel('Frequency')
    plt.title('Price Distribution')
    
    plt.tight_layout()
    plt.show()

# Generate test predictions and save
def create_submission():
    X_test = test_data[features].fillna(test_data[features].median())
    test_predictions = model.predict(X_test)
    
    submission = sample_submission.copy()
    submission.iloc[:, 1] = test_predictions
    submission.to_csv('predictions.csv', index=False)
    
    print(f"Test predictions saved to 'predictions.csv'")
    print(f"Average predicted price: ${test_predictions.mean():,.0f}")
    print(f"Test samples: {len(test_predictions)}")

# Main loop
while True:
    print("\n" + "="*40)
    print("HOUSE PRICE PREDICTOR")
    print("="*40)
    print("1. Predict price")
    print("2. Show graphs")
    print("3. Create submission file")
    print("4. Exit")
    
    choice = input("\nChoose option (1-4): ")
    
    if choice == '1':
        predict_house_price()
    elif choice == '2':
        show_graphs()
    elif choice == '3':
        create_submission()
    elif choice == '4':
        print("Goodbye!")
        break
    else:
        print("Invalid choice. Please enter 1-4.")
