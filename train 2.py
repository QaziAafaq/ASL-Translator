import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("1. Loading ASL landmark data...")
try:
    data = pd.read_csv('asl_dataset.csv')
except FileNotFoundError:
    print("Error: 'asl_dataset.csv' not found. Please run Collect_data.py first!")
    exit()

X = data.iloc[:, 1:].values.astype(np.float32)   # 63 landmark features
y = data.iloc[:, 0].values.astype(np.int32)       # label index (0=A … 25=Z)

num_classes  = 26
num_features = 63

alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Show per-class sample counts so you can spot imbalance
print("\nSamples per letter:")
unique, counts = np.unique(y, return_counts=True)
for idx, cnt in zip(unique, counts):
    print(f"  {alphabet[idx]}: {cnt}")

# ─────────────────────────────────────────────
# 2. Data Augmentation
#    Adds Gaussian noise & small random scale
#    jitter to every sample → gives the model
#    a far richer, more varied training set.
# ─────────────────────────────────────────────
def augment(X, y, copies=6, noise_std=0.015, scale_range=0.08):
    """Return original data + `copies` augmented versions."""
    augmented_X = [X]
    augmented_y = [y]
    rng = np.random.default_rng(42)
    for _ in range(copies):
        noise  = rng.normal(0, noise_std, X.shape).astype(np.float32)
        scales = rng.uniform(1 - scale_range, 1 + scale_range,
                             (X.shape[0], 1)).astype(np.float32)
        X_aug = np.clip(X * scales + noise, -1.0, 1.0)
        augmented_X.append(X_aug)
        augmented_y.append(y)
    return np.concatenate(augmented_X), np.concatenate(augmented_y)

print("\n2. Augmenting data (6x)...")
X_aug, y_aug = augment(X, y)
print(f"   Total samples after augmentation: {len(X_aug)}")

# ─────────────────────────────────────────────
# 3. Train / Validation Split
# ─────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug
)
print(f"   Train: {len(X_train)}  |  Val: {len(X_val)}")

# ─────────────────────────────────────────────
# 4. Build a Deeper Model with More Regularisation
#    The bigger network + BatchNorm + higher
#    Dropout prevents overfitting and forces the
#    model to learn robust landmark patterns.
# ─────────────────────────────────────────────
print("\n3. Building improved model...")
model = Sequential([
    tf.keras.Input(shape=(num_features,)),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─────────────────────────────────────────────
# 5. Train with Callbacks 
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=7, min_lr=1e-6, verbose=1),
]

print("\n4. Training...")
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────
# 6. Per-Letter Accuracy Report 
# ─────────────────────────────────────────────
print("\n5. Per-letter accuracy on validation set:")

# Use the ORIGINAL (non-augmented) data for a fair real-world test
X_orig_train, X_orig_val, y_orig_train, y_orig_val = train_test_split(
    X, y, test_size=0.2, random_state=99, stratify=y
)



preds = np.argmax(model.predict(X_orig_val, verbose=0), axis=1)
correct_dict = {}
total_dict   = {}
for true_label, pred_label in zip(y_orig_val, preds):
    total_dict[true_label]   = total_dict.get(true_label, 0) + 1
    correct_dict[true_label] = correct_dict.get(true_label, 0) + (1 if true_label == pred_label else 0)

problem_letters = set("RVWXZ")
all_ok = True
for i in range(26):
    if total_dict.get(i, 0) == 0:
        continue
    acc  = correct_dict.get(i, 0) / total_dict[i] * 100
    mark = " ← STILL STRUGGLING" if acc < 80 else ""
    if acc < 80:
        all_ok = False
    print(f"  {alphabet[i]}: {correct_dict.get(i,0)}/{total_dict[i]} = {acc:.1f}%{mark}"
          + (" [target]" if alphabet[i] in problem_letters else ""))

if all_ok:
    print("\n  All letters >= 80% accuracy — good to go!")
else:
    print("\n  Some letters still below 80%. Consider collecting more data for them.")

# ─────────────────────────────────────────────
# 7. Save
# ─────────────────────────────────────────────
print("\n6. Saving model to 'asl_model.h5'...")
model.save('asl_model.h5')
print("Done! 'asl_model.h5' has been updated.")
