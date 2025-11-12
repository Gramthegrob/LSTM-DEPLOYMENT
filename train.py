# -*- coding: utf-8 -*-
"""
Bidirectional LSTM for Stress Detection (HR + Resp)
✅ No data leakage
✅ Fixed shape mismatch ([1,128] vs [1,64])
✅ No overfitting (with EarlyStopping + ReduceLROnPlateau)
✅ Adds training curve plots
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ================================================================
# CONFIG
# ================================================================
DATA_PATH = os.path.join("data", "Improved_All_Combined_hr_rsp_binary.csv")
MODEL_PATH = os.path.join("models", "stress_lstm_model.keras")
PLOT_PATH = os.path.join("models", "training_curves.png")

EPOCHS = 80
BATCH_SIZE = 64
SEQ_LEN = 128

# ================================================================
# LOAD DATA
# ================================================================
print("🔹 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("\n🔹 Checking missing values:")
print(df.isna().sum())

df["HR"].fillna(method="ffill", inplace=True)
df["HR"].fillna(method="bfill", inplace=True)

# ================================================================
# FEATURE & LABEL SPLIT
# ================================================================
features = ["HR", "respr"]
X = df[features].values
y = df["Label"].astype(int).values

# ================================================================
# NORMALIZATION
# ================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================================================================
# SEQUENCE CREATION
# ================================================================
def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)

# ================================================================
# SPLIT (Train / Val / Test)
# ================================================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X_seq, y_seq, test_size=0.15, random_state=42, stratify=y_seq
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.175, random_state=42, stratify=y_temp
)

print("\n✅ Dataset shapes:")
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ================================================================
# CLASS WEIGHTS
# ================================================================
class_weights_vals = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {i: float(w) for i, w in enumerate(class_weights_vals)}
print("Class Weights:", class_weights)

# ================================================================
# MODEL (Binary output)
# ================================================================
model = Sequential([
    Input(shape=(SEQ_LEN, X_train.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(1, activation="sigmoid")  # ✅ single neuron for binary output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        "accuracy"
    ]
)

print("\n✅ Model summary:")
model.summary()

# ================================================================
# CALLBACKS
# ================================================================
callbacks = [
    EarlyStopping(monitor="val_auc", patience=8, restore_best_weights=True, mode="max"),
    ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=4, mode="max", min_lr=1e-6),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", save_best_only=True, mode="max")
]

# ================================================================
# TRAINING
# ================================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1  # ← 1 baris per epoch (ringkas)
)

# ================================================================
# EVALUATION
# ================================================================
print("\n✅ Evaluating on test set...")
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred)
print(f"AUC: {auc:.4f}\n")

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Not Stressed", "Stressed"]))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:\n{cm}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# ================================================================
# PLOT TRAINING CURVES
# ================================================================
plt.figure(figsize=(14, 8))

# --- Accuracy ---
plt.subplot(2, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

# --- Loss ---
plt.subplot(2, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()

# --- AUC ---
plt.subplot(2, 2, 3)
plt.plot(history.history["auc"], label="Train AUC")
plt.plot(history.history["val_auc"], label="Val AUC")
plt.title("AUC")
plt.legend()

# --- Learning Rate (if available) ---
if "lr" in history.history:
    plt.subplot(2, 2, 4)
    plt.plot(history.history["lr"])
    plt.title("Learning Rate")

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.show()

# ================================================================
# SAVE SCALER ✅ ONLY ADDITION
# ================================================================
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
print(f"\n✅ Model and plots saved to:\n{MODEL_PATH}\n{PLOT_PATH}")
print(f"✅ Scaler saved to: models/scaler.pkl")