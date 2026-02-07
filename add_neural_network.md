# Adding Neural Networks to California Housing Model

This guide shows how to add neural networks to your model comparison.

## Option 1: MLPRegressor (Simplest - Already Have Dependencies)

### Step 1: Add to imports (Section 5)

```python
from sklearn.neural_network import MLPRegressor
```

### Step 2: Add to models dictionary

```python
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Support Vector Regression': SVR(kernel='rbf', C=1.0),
    'Neural Network (MLP)': MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),  # 3 hidden layers: 100 -> 50 -> 25 neurons
        activation='relu',                  # ReLU activation function
        solver='adam',                      # Adam optimizer
        alpha=0.0001,                       # L2 regularization
        batch_size='auto',                  # Automatic batch sizing
        learning_rate='adaptive',           # Adaptive learning rate
        learning_rate_init=0.001,          # Initial learning rate
        max_iter=500,                       # Maximum iterations
        random_state=42,
        early_stopping=True,                # Stop if validation score doesn't improve
        validation_fraction=0.1,            # 10% for validation
        n_iter_no_change=20,               # Stop after 20 epochs without improvement
        verbose=False
    )
}
```

### Step 3: Add to hyperparameter grid (Section 6)

```python
param_grids = {
    # ... existing grids ...
    'Neural Network (MLP)': {
        'hidden_layer_sizes': [
            (100,),           # 1 hidden layer
            (100, 50),        # 2 hidden layers
            (100, 50, 25),    # 3 hidden layers
            (150, 75),        # 2 layers, more neurons
            (200, 100, 50)    # 3 layers, more neurons
        ],
        'activation': ['relu', 'tanh'],
        'alpha': uniform(0.0001, 0.01),          # L2 penalty
        'learning_rate_init': uniform(0.0001, 0.01),
        'batch_size': [32, 64, 128],
        'max_iter': [500, 1000]
    }
}
```

---

## Option 2: Keras/TensorFlow (More Flexible)

### Step 1: Install TensorFlow

```bash
pip install tensorflow>=2.15.0
```

### Step 2: Create Keras wrapper for scikit-learn compatibility

Add this NEW CELL after the imports in Section 5:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.base import BaseEstimator, RegressorMixin

class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper to make Keras models compatible with scikit-learn API
    """
    def __init__(self, epochs=100, batch_size=32, learning_rate=0.001, 
                 hidden_layers=(100, 50, 25), dropout_rate=0.2):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
    
    def _build_model(self, input_dim):
        """Build the neural network architecture"""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(self.hidden_layers[0], activation='relu', 
                              input_dim=input_dim))
        model.add(layers.Dropout(self.dropout_rate))
        
        # Hidden layers
        for neurons in self.hidden_layers[1:]:
            model.add(layers.Dense(neurons, activation='relu'))
            model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def fit(self, X, y):
        """Train the model"""
        # Build model
        self.model = self._build_model(X.shape[1])
        
        # Early stopping callback
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0).flatten()
    
    def score(self, X, y):
        """Return R² score"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
```

### Step 3: Add to models dictionary

```python
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Support Vector Regression': SVR(kernel='rbf', C=1.0),
    'Deep Neural Network (Keras)': KerasRegressorWrapper(
        epochs=200,
        batch_size=64,
        learning_rate=0.001,
        hidden_layers=(128, 64, 32),
        dropout_rate=0.2
    )
}
```

### Step 4: Add to hyperparameter grid (Section 6)

```python
param_grids = {
    # ... existing grids ...
    'Deep Neural Network (Keras)': {
        'epochs': [100, 200, 300],
        'batch_size': [32, 64, 128],
        'learning_rate': uniform(0.0001, 0.01),
        'hidden_layers': [
            (100, 50),
            (128, 64, 32),
            (150, 100, 50),
            (200, 100, 50, 25)
        ],
        'dropout_rate': uniform(0.1, 0.3)
    }
}
```

### Step 5: (Optional) Visualize training history

Add this NEW CELL after training the Keras model:

```python
# Visualize neural network training history
if 'Deep Neural Network (Keras)' in tuned_models:
    keras_model = tuned_models['Deep Neural Network (Keras)']
    history = keras_model.history
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Neural Network Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Neural Network Training History - MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

---

## Comparison of Options

| Feature | MLPRegressor | Keras/TensorFlow |
|---------|--------------|------------------|
| **Setup Complexity** | Very Easy | Moderate |
| **Dependencies** | Already included | Requires TensorFlow |
| **Flexibility** | Limited | Very High |
| **Architecture Options** | Basic MLP | Any architecture (CNN, RNN, custom) |
| **Training Visualization** | No | Yes (training curves) |
| **Performance** | Good | Excellent |
| **GPU Support** | No | Yes |
| **Early Stopping** | Yes | Yes |
| **Best For** | Quick experiments | Production/research |

## Recommendation

- **Start with MLPRegressor**: It's already available, easy to use, and integrates seamlessly
- **Upgrade to Keras** if you need:
  - More complex architectures
  - Better control over training
  - GPU acceleration
  - Training visualization
  - Production deployment

## Expected Performance

Neural networks typically achieve:
- **RMSE**: 0.42-0.48 (similar to tree-based models)
- **R²**: 0.82-0.86
- **Training Time**: Longer than tree models, but can be accelerated with GPU

## Tips for Better Neural Network Performance

1. **Feature Scaling**: Already done! Neural networks require scaled features
2. **Architecture**: Start simple (2-3 layers), then add complexity
3. **Regularization**: Use dropout (0.1-0.3) to prevent overfitting
4. **Learning Rate**: Start with 0.001, tune if needed
5. **Early Stopping**: Prevents overfitting, saves training time
6. **Batch Size**: Larger (64-128) for faster training, smaller (16-32) for better generalization
7. **Epochs**: Start with 100-200, use early stopping to find optimal point

## Common Issues & Solutions

### Issue: Training is slow
**Solution**: Reduce max_iter/epochs, increase batch_size, or use GPU

### Issue: Model is overfitting (train score >> test score)
**Solution**: Increase alpha/dropout_rate, reduce model complexity, or use early stopping

### Issue: Model is underfitting (both scores are poor)
**Solution**: Increase hidden_layer_sizes, train longer, or reduce regularization

### Issue: Results are inconsistent
**Solution**: Set random_state=42 everywhere, use more epochs, or average multiple runs
