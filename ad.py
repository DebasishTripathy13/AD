import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Optional imports - the algorithm will adapt if these aren't available
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.models import Model
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("TensorFlow not available. Autoencoder functionality will be disabled.")

try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_BOOSTING = True
except ImportError:
    HAS_BOOSTING = False
    print("XGBoost and/or LightGBM not available. Using RandomForest as substitute.")

try:
    from sklearn.mixture import GaussianMixture
    HAS_GMM = True
except ImportError:
    HAS_GMM = False
    print("GaussianMixture not available. GMM functionality will be disabled.")

class AdaptiveAnomalyDetector:
    """
    A comprehensive anomaly detection system that combines multiple techniques:

    Unsupervised methods:
    - Isolation Forest: Effective for high-dimensional data and various anomaly types
    - One-Class SVM: Good for non-linear boundaries
    - Local Outlier Factor: Excels at detecting density-based anomalies
    - DBSCAN: Identifies clusters and noise points
    - Autoencoder: Captures complex patterns through neural network reconstruction
    - GMM: Models normal data as a mixture of Gaussian distributions

    Semi-supervised methods (when labels are available):
    - XGBoost/LightGBM: Strong gradient boosting classifiers
    - Random Forest: Robust ensemble method as fallback

    Features:
    - Adaptive preprocessing pipeline with multiple scaling options
    - Optional dimensionality reduction
    - Automatic hyperparameter adaptation
    - Model explainability
    - Ensemble weighting based on performance
    """

    def __init__(self, contamination=0.1, use_dim_reduction=True, n_components=0.95,
                 adaptive_scaling=True, ensemble_weight=0.6, random_state=42,
                 enable_all_models=True):
        """
        Initialize the adaptive anomaly detector

        Parameters:
        -----------
        contamination : float, default=0.1
            Expected proportion of outliers in the data
        use_dim_reduction : bool, default=True
            Whether to use dimensionality reduction
        n_components : float or int, default=0.95
            Number of components to keep if using dimensionality reduction
        adaptive_scaling : bool, default=True
            Whether to automatically select the best scaling method
        ensemble_weight : float, default=0.6
            Weight for supervised models vs. unsupervised models
        random_state : int, default=42
            Random seed for reproducibility
        enable_all_models : bool, default=True
            Whether to enable all models or use a selective subset
        """
        self.contamination = contamination
        self.use_dim_reduction = use_dim_reduction
        self.n_components = n_components
        self.adaptive_scaling = adaptive_scaling
        self.ensemble_weight = ensemble_weight
        self.random_state = random_state
        self.enable_all_models = enable_all_models

        # Initialize preprocessing components
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        self.best_scaler = None
        self.pca = None

        # Initialize dictionary to store models
        self.models = {}

        # Initialize unsupervised models
        self._init_unsupervised_models()

        # Supervised models will be initialized during fit if labels are available
        self.supervised_models = {}

        # Model results and metadata
        self.model_scores = {}
        self.model_weights = {}
        self.is_fitted = False
        self.best_preprocessing = None
        self.selected_features = None

    def _init_unsupervised_models(self):
        """Initialize all unsupervised models with default parameters"""

        # Core models - always enabled
        self.models['iforest'] = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto'
        )

        self.models['ocsvm'] = OneClassSVM(
            nu=self.contamination,
            kernel='rbf',
            gamma='scale'
        )

        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=True
        )

        # Optional models - enabled based on flags
        if self.enable_all_models:
            self.models['dbscan'] = DBSCAN(
                eps=0.5,  # Will be tuned during fitting
                min_samples=5,
                n_jobs=-1
            )

            self.models['kmeans'] = KMeans(
                n_clusters=3,  # Will be tuned during fitting
                random_state=self.random_state
            )

            # Gaussian Mixture Model
            if HAS_GMM:
                self.models['gmm'] = GaussianMixture(
                    n_components=5,  # Will be tuned during fitting
                    covariance_type='full',
                    random_state=self.random_state
                )

        # Deep learning model (autoencoder)
        self.autoencoder = None
        self.encoder = None

    def _build_autoencoder(self, input_dim):
        """Build autoencoder architecture based on input dimensions"""
        if not HAS_TENSORFLOW:
            return None, None

        # Determine architecture based on input size
        if input_dim < 10:
            hidden_dims = [max(int(input_dim * 2), 8), max(int(input_dim), 4)]
        else:
            hidden_dims = [input_dim * 2, input_dim, input_dim // 2]

        # Bottleneck dimension
        bottleneck_dim = max(2, input_dim // 4)

        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = input_layer

        # Encoding layers
        for dim in hidden_dims:
            encoded = Dense(dim, activation='relu')(encoded)
            encoded = Dropout(0.2)(encoded)

        # Bottleneck layer
        bottleneck = Dense(bottleneck_dim, activation='relu')(encoded)

        # Decoding layers
        decoded = bottleneck
        for dim in reversed(hidden_dims):
            decoded = Dense(dim, activation='relu')(decoded)
            decoded = Dropout(0.2)(decoded)

        # Output layer
        output_layer = Dense(input_dim, activation='sigmoid')(decoded)

        # Create models
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        encoder = Model(inputs=input_layer, outputs=bottleneck)

        # Compile
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        return autoencoder, encoder

    def _select_best_scaler(self, X):
        """Select the best scaler based on data characteristics"""
        if not self.adaptive_scaling:
            return 'standard'

        # Check for outliers
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1

        # Count features with outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = 0
        for i in range(X.shape[1]):
            outliers = np.sum((X[:, i] < lower_bound[i]) | (X[:, i] > upper_bound[i]))
            if outliers / X.shape[0] > 0.05:  # More than 5% outliers
                outlier_count += 1

        # If many outliers, prefer robust scaling
        if outlier_count > X.shape[1] * 0.3:
            return 'robust'
        # If data has different ranges, prefer min-max scaling
        elif np.std(np.std(X, axis=0)) > 1.0:
            return 'minmax'
        # Otherwise use standard scaler
        else:
            return 'standard'

    def _tune_model_parameters(self, X):
        """Tune parameters for models based on data characteristics"""
        n_samples = X.shape[0]

        # Update DBSCAN parameters if available
        if 'dbscan' in self.models:
            # Estimate epsilon using nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(10, max(5, int(np.log(n_samples)))),
                                 algorithm='auto').fit(X)
            distances, _ = nn.kneighbors(X)

            # Use average of k-distances as epsilon
            eps = np.mean(distances[:, -1]) * 1.5
            self.models['dbscan'].eps = eps

            # Update min_samples based on dataset size
            self.models['dbscan'].min_samples = max(5, int(np.log(n_samples)))

        # Update KMeans parameters
        if 'kmeans' in self.models:
            # Estimate number of clusters using silhouette score
            from sklearn.metrics import silhouette_score

            best_score = -1
            best_k = 3

            # Try a range of cluster numbers
            max_clusters = min(10, n_samples // 100 + 2)
            for k in range(2, max_clusters):
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X)

                if len(np.unique(labels)) > 1:  # Ensure we have at least 2 clusters
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k

            self.models['kmeans'].n_clusters = best_k

        # Update GMM parameters
        if HAS_GMM and 'gmm' in self.models:
            # Similar approach for GMM component selection
            best_bic = np.inf
            best_n = 2

            # Try a range of component numbers
            max_components = min(10, n_samples // 100 + 2)
            for n in range(2, max_components):
                gmm = GaussianMixture(n_components=n, random_state=self.random_state)
                gmm.fit(X)
                bic = gmm.bic(X)

                if bic < best_bic:
                    best_bic = bic
                    best_n = n

            self.models['gmm'].n_components = best_n

    def fit(self, X, y=None, feature_names=None):
        """
        Fit the anomaly detector to the data

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Target values (1 for normal, 0 for anomaly)
        feature_names : list, optional
            Names of features for explanation purposes

        Returns:
        --------
        self : object
        """
        # Store feature names if provided
        self.feature_names = feature_names

        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                self.feature_names = X.columns.tolist()
            X = X.values

        # Select best scaler
        scaler_name = self._select_best_scaler(X)
        self.best_scaler = scaler_name

        # Scale the data
        X_scaled = self.scalers[scaler_name].fit_transform(X)

        # Apply dimensionality reduction if needed
        if self.use_dim_reduction and X.shape[1] > 2:
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
            X_processed = self.pca.fit_transform(X_scaled)

            # Store explained variance for later
            self.explained_variance = self.pca.explained_variance_ratio_
        else:
            X_processed = X_scaled
            self.pca = None

        # Tune model parameters based on data characteristics
        self._tune_model_parameters(X_processed)

        # Build and train autoencoder if TensorFlow is available
        if HAS_TENSORFLOW:
            n_features = X_processed.shape[1]
            self.autoencoder, self.encoder = self._build_autoencoder(n_features)

            # Train autoencoder
            if self.autoencoder is not None:
                self.autoencoder.fit(
                    X_processed, X_processed,
                    epochs=min(100, max(20, 5000 // X.shape[0])),  # Adapt epochs to dataset size
                    batch_size=min(64, X.shape[0] // 10 + 1),
                    shuffle=True,
                    validation_split=0.1,
                    verbose=0
                )

        # Fit unsupervised models
        for name, model in self.models.items():
            if name != 'kmeans':  # KMeans will be handled differently
                model.fit(X_processed)

        # Special handling for KMeans
        if 'kmeans' in self.models:
            self.models['kmeans'].fit(X_processed)
            # Calculate distance to closest centroid
            self.kmeans_centroids = self.models['kmeans'].cluster_centers_

        # For supervised learning if labels are available
        if y is not None and np.sum(y == 0) > 0 and np.sum(y == 1) > 0:
            print("Label information detected. Enabling supervised models.")

            # Split data for supervised models
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y
            )

            # Initialize and train supervised models
            if HAS_BOOSTING:
                self.supervised_models['xgb'] = xgb.XGBClassifier(
                    objective='binary:logistic',
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state
                )
                self.supervised_models['xgb'].fit(X_train, y_train)

                self.supervised_models['lgb'] = lgb.LGBMClassifier(
                    objective='binary',
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state
                )
                self.supervised_models['lgb'].fit(X_train, y_train)

            self.supervised_models['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=min(10, X.shape[1]),
                random_state=self.random_state
            )
            self.supervised_models['rf'].fit(X_train, y_train)

            # Evaluate models on validation set and compute weights
            model_scores = {}
            if len(np.unique(y_val)) > 1:
                for name, model in self.supervised_models.items():
                    pred_proba = model.predict_proba(X_val)[:, 1]
                    try:
                        model_scores[name] = roc_auc_score(y_val, pred_proba)
                    except Exception:
                        model_scores[name] = 0.5  # Default score if AUC fails

                # Normalize scores to get weights
                if sum(model_scores.values()) > 0:
                    total_score = sum(model_scores.values())
                    self.model_weights = {k: v/total_score for k, v in model_scores.items()}
                else:
                    # Equal weights if scoring fails
                    self.model_weights = {k: 1.0/len(model_scores) for k in model_scores.keys()}

        self.is_fitted = True
        self.best_preprocessing = {
            'scaler': scaler_name,
            'dim_reduction': self.use_dim_reduction and self.pca is not None,
            'n_components': self.pca.n_components_ if self.pca is not None else None
        }

        print(f"Detector fitted successfully with {scaler_name} scaling.")
        if self.pca is not None:
            print(f"Dimensionality reduced from {X.shape[1]} to {X_processed.shape[1]} features.")

        return self

    def decision_function(self, X):
        """
        Average anomaly score for each sample from multiple models
        Higher scores indicate more normal examples

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        Returns:
        --------
        scores : ndarray of shape (n_samples,)
            Anomaly scores of samples (higher = more normal)
        """
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Call fit() first.")

        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Apply preprocessing
        X_scaled = self.scalers[self.best_scaler].transform(X)

        if self.pca is not None:
            X_processed = self.pca.transform(X_scaled)
        else:
            X_processed = X_scaled

        # Get scores from each model
        scores = {}

        # Core unsupervised models
        # Isolation Forest (higher = more normal)
        scores['iforest'] = self.models['iforest'].decision_function(X_processed)

        # One-Class SVM (higher = more normal)
        scores['ocsvm'] = self.models['ocsvm'].decision_function(X_processed)

        # LOF (higher = more normal, need to negate)
        scores['lof'] = -self.models['lof'].decision_function(X_processed)

        # Optional models
        if 'dbscan' in self.models:
            # DBSCAN (convert to scores)
            dbscan_labels = self.models['dbscan'].fit_predict(X_processed)
            dbscan_scores = np.ones_like(dbscan_labels, dtype=float)
            dbscan_scores[dbscan_labels == -1] = -1  # noise points
            scores['dbscan'] = dbscan_scores

        if 'kmeans' in self.models:
            # KMeans (distance to nearest centroid)
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(X_processed, self.kmeans_centroids)
            min_distances = np.min(distances, axis=1)

            # Invert distances (larger distance = more likely anomaly)
            kmeans_scores = -min_distances

            # Normalize
            if np.std(kmeans_scores) > 0:
                kmeans_scores = (kmeans_scores - np.mean(kmeans_scores)) / np.std(kmeans_scores)

            scores['kmeans'] = kmeans_scores

        if HAS_GMM and 'gmm' in self.models:
            # GMM (log-likelihood, higher = more normal)
            scores['gmm'] = self.models['gmm'].score_samples(X_processed)

        # Autoencoder reconstruction error (negate to make higher = more normal)
        if HAS_TENSORFLOW and self.autoencoder is not None:
            reconstructions = self.autoencoder.predict(X_processed, verbose=0)
            mse = np.mean(np.power(X_processed - reconstructions, 2), axis=1)
            scores['autoencoder'] = -mse

        # Add supervised model scores if available
        for name, model in self.supervised_models.items():
            # Get probabilities for normal class (higher = more normal)
            scores[name] = model.predict_proba(X_processed)[:, 1]

        # Normalize score ranges to [0, 1]
        for key in scores:
            if len(scores[key]) > 0:
                score_min = np.min(scores[key])
                score_max = np.max(scores[key])
                if score_max > score_min:  # Avoid division by zero
                    scores[key] = (scores[key] - score_min) / (score_max - score_min)

        # Compute final scores with weighting
        # Base weights for unsupervised models
        weights = {}
        base_weight = 1.0

        if len(self.supervised_models) > 0:
            # If supervised models exist, adjust weights
            num_unsupervised = len(scores) - len(self.supervised_models)
            base_weight = (1 - self.ensemble_weight) / max(1, num_unsupervised)

            # Add supervised model weights
            for model in self.model_weights.keys():
                weights[model] = self.ensemble_weight * self.model_weights.get(model, 1/len(self.model_weights))

        # Set base weights for unsupervised models
        for model in scores.keys():
            if model not in weights:
                weights[model] = base_weight

        # Weighted average of normalized scores
        final_scores = np.zeros(X_processed.shape[0])
        total_weight = 0

        for model, weight in weights.items():
            if model in scores and len(scores[model]) > 0:
                final_scores += weight * scores[model]
                total_weight += weight

        if total_weight > 0:
            final_scores /= total_weight

        # Store model scores for explanation
        self.last_scores = scores

        return final_scores

    def predict(self, X, threshold=None):
        """
        Predict if samples are normal (-1) or anomalous (1)

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        threshold : float, optional
            Custom threshold for anomaly detection

        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels (-1 for normal, 1 for anomalies)
        """
        scores = self.decision_function(X)

        # Determine threshold if not provided
        if threshold is None:
            threshold = np.percentile(scores, self.contamination * 100)

        # Convert to anomaly labels (-1 for normal, 1 for anomalies)
        predictions = np.ones_like(scores)
        predictions[scores > threshold] = -1

        return predictions

    def predict_proba(self, X):
        """
        Probability estimates for samples being anomalies

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Probability of samples being anomalies (0 to 1)
        """
        scores = self.decision_function(X)

        # Convert scores to probability range [0, 1]
        # Higher scores indicate more normal instances
        # So we need to invert the scores
        score_min = np.min(scores)
        score_max = np.max(scores)
        if score_max > score_min:  # Avoid division by zero
            probs = 1 - (scores - score_min) / (score_max - score_min)
        else:
            probs = np.ones_like(scores) * 0.5

        return probs

    def fit_predict(self, X, y=None):
        """
        Fit the model and predict anomaly labels

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Target values

        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels (-1 for normal, 1 for anomalies)
        """
        self.fit(X, y)
        return self.predict(X)

    def explain_prediction(self, X, indices=None, top_n=3):
        """
        Explain which models contributed most to the anomaly prediction

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        indices : list, optional
            Indices of samples to explain
        top_n : int, default=3
            Number of top contributing models to return

        Returns:
        --------
        explanations : list of dicts
            List of dictionaries containing model contributions
        """
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Call fit() first.")

        # Convert to numpy array if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get predictions
        y_pred = self.predict(X)
        anomaly_probs = self.predict_proba(X)

        # If no indices provided, explain all samples
        if indices is None:
            indices = range(len(y_pred))

        # Process individual models' outputs
        individual_preds = {}

        # Get predictions from each model
        for model_name in self.models.keys():
            if model_name == 'iforest':
                individual_preds[model_name] = self.models[model_name].predict(X)
            elif model_name == 'ocsvm':
                individual_preds[model_name] = self.models[model_name].predict(X)
            elif model_name == 'lof':
                individual_preds[model_name] = self.models[model_name].predict(X)
            elif model_name == 'dbscan':
                dbscan_labels = self.models[model_name].fit_predict(X)
                # Convert to -1/1 format
                preds = np.ones_like(dbscan_labels)
                preds[dbscan_labels == -1] = -1
                individual_preds[model_name] = preds

        # Add supervised models if available
        for model_name, model in self.supervised_models.items():
            # Convert 0/1 to -1/1 format
            preds = model.predict(X)
            individual_preds[model_name] = 2 * preds - 1

        # Generate explanations for each requested sample
        explanations = []
        for idx in indices:
            explanation = {
                'sample_idx': idx,
                'prediction': 'Anomaly' if y_pred[idx] == 1 else 'Normal',
                'anomaly_probability': anomaly_probs[idx],
                'contributing_models': []
            }

            # Find models that agree with the prediction
            contributions = {}
            for model_name, preds in individual_preds.items():
                contributions[model_name] = 1 if preds[idx] == y_pred[idx] else 0

            # Sort by contribution
            sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

            # Get top contributing models
            explanation['contributing_models'] = [model for model, contrib in sorted_contribs[:top_n] if contrib > 0]

            # Feature importance if available for anomalies
            if y_pred[idx] == 1 and HAS_TENSORFLOW and self.autoencoder is not None:
                # Use reconstruction error per feature as a proxy for feature importance
                X_processed = self._preprocess_data(X[idx:idx+1])
                reconstructed = self.autoencoder.predict(X_processed, verbose=0)[0]
                errors = np.abs(X_processed[0] - reconstructed)

                # Get top contributing features
                if self.feature_names is not None:
                    if self.pca is not None:
                        # For PCA, we need a different approach
                        # We'll use the magnitude of PCA components
                        feature_importance = {}
                        n_orig_features = len(self.feature_names)
                        n_components = self.pca.components_.shape[0]

                        # Calculate weighted errors using PCA components
                        weighted_errors = np.zeros(n_orig_features)
                        for i in range(n_components):
                            component_error = errors[i]
                            weighted_errors += component_error * np.abs(self.pca.components_[i])

                        # Create feature importance dictionary
                        for i in range(n_orig_features):
                            feature_importance[self.feature_names[i]] = weighted_errors[i]
                    else:
                        # Direct feature importance
                        feature_importance = {self.feature_names[i]: errors[i] for i in range(len(errors))}

                    # Sort and get top features
                    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    explanation['top_anomalous_features'] = [feature for feature, score in sorted_features[:top_n]]

            explanations.append(explanation)

        return explanations

    def _preprocess_data(self, X):
        """Helper method to preprocess data consistently"""
        X_scaled = self.scalers[self.best_scaler].transform(X)
        if self.pca is not None:
            return self.pca.transform(X_scaled)
        return X_scaled

    def get_model_info(self):
        """Get information about the models and their configuration"""
        if not self.is_fitted:
            return {"status": "Not fitted yet"}

        info = {
            "preprocessing": self.best_preprocessing,
            "active_models": list(self.models.keys()),
            "supervised_models": list(self.supervised_models.keys()) if hasattr(self, 'supervised_models') else [],
            "contamination": self.contamination
        }

        if self.pca is not None:
            info["pca_explained_variance"] = sum(self.explained_variance)

        return info

    def evaluate(self, X, y):
        """
        Evaluate the performance of the anomaly detector.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels (1 for normal, 0 for anomaly)

        Returns:
        --------
        metrics : dict
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise Exception("Model is not fitted yet. Call fit() first.")

        # Predict anomalies
        y_pred = self.predict(X)
        y_scores = self.decision_function(X)

        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = np.mean(y_pred == y)
        metrics['precision'], metrics['recall'], _ = precision_recall_curve(y, y_scores)
        metrics['auc'] = auc(metrics['recall'], metrics['precision'])
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred)

        return metrics

    def plot_anomalies(self, X, y=None, title="Anomaly Detection"):
        """
        Plot the anomalies detected by the model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,), optional
            True labels (1 for normal, 0 for anomaly)
        title : str, optional
            Title of the plot
        """
        import matplotlib.pyplot as plt

        # Predict anomalies
        y_pred = self.predict(X)

        # Plot
        plt.figure(figsize=(10, 6))
        if y is not None:
            plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', label='True Labels')
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='x', label='Predicted Labels')
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

    def save_model(self, filepath):
        """
        Save the trained model to a file.

        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        import joblib
        joblib.dump(self, filepath)

    @staticmethod
    def load_model(filepath):
        """
        Load a trained model from a file.

        Parameters:
        -----------
        filepath : str
            Path to load the model from

        Returns:
        --------
        model : AdaptiveAnomalyDetector
            Loaded model
        """
        import joblib
        return joblib.load(filepath)

    def feature_importance(self, X):
        """
        Calculate feature importance based on the autoencoder reconstruction error.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        Returns:
        --------
        importance : dict
            Dictionary containing feature importance scores
        """
        if not HAS_TENSORFLOW or self.autoencoder is None:
            raise Exception("Autoencoder is not available or not trained.")

        # Preprocess data
        X_processed = self._preprocess_data(X)

        # Get reconstruction error
        reconstructions = self.autoencoder.predict(X_processed, verbose=0)
        errors = np.mean(np.abs(X_processed - reconstructions), axis=0)

        # Normalize errors
        errors /= np.max(errors)

        # Create feature importance dictionary
        importance = {self.feature_names[i]: errors[i] for i in range(len(errors))}

        return importance
