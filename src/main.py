import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import norm
import hashlib
import warnings
import random
import matplotlib.pyplot as plt

# Suppress informational messages and warnings for a clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# --- Phase 1: Data Handling ---
def load_all_user_data(user_id_range):
    all_data = {}    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd() # Fallback for interactive environments
        
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    data_path = os.path.join(project_root, 'dataset', 'Behaviour Biometrics', 'feature_kmt_dataset', 'feature_kmt_xlsx')
    
    if not os.path.isdir(data_path):
        print(f"[ERROR] The directory '{data_path}' does not exist.")
        return {} 
    for user_id in user_id_range:
        user_file = f"feature_kmt_user_{user_id:04d}.xlsx"
        full_path = os.path.join(data_path, user_file)        
        if not os.path.exists(full_path):
            continue        
        try:
            df = pd.read_excel(full_path)
        except Exception:
            continue
        label_col_name = 'label' if 'label' in df.columns else 'class'
        if label_col_name not in df.columns:
            continue
            
        legitimate_df = df[df[label_col_name] == 1].copy()
        if not legitimate_df.empty:
            features_df = legitimate_df.drop(columns=[label_col_name])
            all_data[user_id] = features_df.values
            
    if not all_data:
        return {}
        
    scaler = MinMaxScaler()
    stacked_data = np.vstack(list(all_data.values()))
    scaler.fit(stacked_data)
    
    for user_id in all_data:
        all_data[user_id] = scaler.transform(all_data[user_id])
        
    return all_data

# --- Phase 2: Siamese Network for Template Generation ---

def make_siamese_pairs(all_data):
    """Creates positive (same user) and negative (different user) pairs for training."""
    pairs, labels = [], []
    user_ids = list(all_data.keys())
    if len(user_ids) < 2: return np.array([]), np.array([])

    for user_id in user_ids:
        user_samples = all_data[user_id]
        if len(user_samples) < 2: continue
        
        num_positive_added = 0
        for i in range(len(user_samples)):
            for j in range(i + 1, len(user_samples)):
                pairs.append([user_samples[i], user_samples[j]])
                labels.append(1.0)
                num_positive_added += 1

        for _ in range(num_positive_added):
            imposter_id = random.choice([uid for uid in user_ids if uid != user_id])
            anchor_sample = random.choice(user_samples)
            imposter_sample = random.choice(all_data[imposter_id])
            pairs.append([anchor_sample, imposter_sample])
            labels.append(0.0)
            
    return np.array(pairs), np.array(labels)

def create_siamese_network(input_dim: int, encoding_dim: int):
    """Creates the base feature extraction network (encoder) for the Siamese model."""
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(input_layer)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(encoding_dim, activation='relu')(x)
    return keras.Model(inputs=input_layer, outputs=x, name="base_network")

class DistanceLayer(layers.Layer):
    """A custom Keras layer to compute the L2 distance between two vectors."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        anchor, positive = inputs
        sum_squared = tf.reduce_sum(tf.square(anchor - positive), axis=1, keepdims=True)
        return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

def contrastive_loss(y_true, y_pred): # The loss function for a Siamese Network. It pushes positive pairs closerand negative pairs further apart.
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

# --- Phase 3: Cryptographic Component ---
class SimulatedFuzzyExtractor:
    """A class to simulate the functionality of a Fuzzy Extractor."""
    def __init__(self, key_length=16):
        self.key_length = key_length

    def generate(self, template: np.array):
        """Generates a key and helper data from an enrollment template."""
        secret_key = os.urandom(self.key_length)
        helper_data = {'original_template': template, 'secret_key': secret_key}
        return secret_key, helper_data

    def reproduce(self, new_template: np.array, helper_data: dict, tolerance: float):
        """Reproduces the original key if the new template is within the tolerance."""
        original_template = helper_data['original_template']
        secret_key = helper_data['secret_key']
        distance = norm(original_template - new_template)
        
        if distance <= tolerance:
            return secret_key, distance
        return None, distance

# --- Phase 4: System Evaluation and Demonstration ---

if __name__ == "__main__":
    # --- Configuration ---
    USER_ID_RANGE = range(1, 89)
    ENCODING_DIM = 32
    EPOCHS = 50
    KEY_LENGTH_BYTES = 16
    DEMO_USER_ID = 5
    DEMO_INTRUDER_IDS = [6, 10, 25]
    MODEL_SAVE_PATH = 'siamese_model.keras'
    print("===== STARTING SIAMESE NETWORK-BASED EVALUATION =====")
    all_user_data = load_all_user_data(USER_ID_RANGE)
    if not all_user_data:
        print("[ERROR] Critical Error: No data loaded. Exiting.")
        exit()
    print(f"[INFO] Loaded data for {len(all_user_data)} users.")

    # --- Model Training or Loading ---
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[INFO] Found existing model at '{MODEL_SAVE_PATH}'. Loading model...")
        base_network = keras.models.load_model(MODEL_SAVE_PATH)
        print("[SUCCESS] Model loaded successfully.")
    else:
        print("[INFO] No existing model found. Starting training process...")
        pairs, labels = make_siamese_pairs(all_user_data)
        if len(pairs) == 0:
            print("[ERROR] Critical Error: Could not generate training pairs. Exiting.")
            exit()
        input_dim = pairs.shape[-1]
        base_network = create_siamese_network(input_dim, ENCODING_DIM)
        
        input_a = layers.Input(shape=(input_dim,), name="input_anchor")
        input_b = layers.Input(shape=(input_dim,), name="input_positive")
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        distance = DistanceLayer()([processed_a, processed_b])
        siamese_model = keras.Model(inputs=[input_a, input_b], outputs=distance)
        siamese_model.compile(optimizer='adam', loss=contrastive_loss)
        print(f"[INFO] Training Siamese Network on {len(pairs)} pairs...")
        siamese_model.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=EPOCHS, batch_size=64, verbose=1)
        print("[SUCCESS] Training complete.")
        base_network.save(MODEL_SAVE_PATH)
        print(f"[SUCCESS] Trained model saved to '{MODEL_SAVE_PATH}'.")

    # --- Evaluation ---
    print("[INFO] Pre-computing all user templates for faster evaluation...")
    template_generator = base_network
    user_templates = {}
    for user_id, samples in all_user_data.items():
        user_templates[user_id] = template_generator.predict(samples, verbose=0)
    print("[SUCCESS] All templates generated.")

    print("[INFO] Calculating genuine and imposter distances for EER analysis...")
    genuine_distances, imposter_distances = [], []

    for user_id, templates in user_templates.items():
        if len(templates) < 2: continue
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                genuine_distances.append(norm(templates[i] - templates[j]))
        
        enrollment_template = templates[0]
        for imposter_id, imposter_templates in user_templates.items():
            if user_id == imposter_id: continue
            for imposter_template in imposter_templates:
                imposter_distances.append(norm(enrollment_template - imposter_template))
    
    print("[SUCCESS] Distance calculation complete.")
    thresholds = np.arange(0.0, 2.0, 0.01)
    frr_list, far_list = [], []
    eer_threshold, min_diff = 0, 1.0
    for t in thresholds:
        frr = np.sum(np.array(genuine_distances) > t) / len(genuine_distances)
        far = np.sum(np.array(imposter_distances) <= t) / len(imposter_distances)
        frr_list.append(frr)
        far_list.append(far)
        if abs(frr - far) < min_diff:
            min_diff = abs(frr - far)
            eer_threshold = t
            eer_value = (frr + far) / 2
    print("\n\n===== EVALUATION COMPLETE =====")
    print(f"\n--- System Performance Metrics ---")
    print(f"Equal Error Rate (EER) is approximately: {eer_value:.2%}")
    print(f"Optimal Distance Threshold for EER: {eer_threshold:.2f}")
    print("------------------------------------")
    # --- Demonstration Mode ---
    print(f"\n===== DEMONSTRATION MODE (User {DEMO_USER_ID} vs Intruders) =====")
    if DEMO_USER_ID not in user_templates or len(user_templates[DEMO_USER_ID]) < 3:
        print(f"[ERROR] Cannot run demonstration. User {DEMO_USER_ID} does not have enough samples (needs at least 3).")
    else:
        genuine_templates = user_templates[DEMO_USER_ID]
        enrollment_template = genuine_templates[0:1]
        fuzzy_extractor = SimulatedFuzzyExtractor(key_length=KEY_LENGTH_BYTES)
        genuine_key, helper_data = fuzzy_extractor.generate(enrollment_template)
        print(f"\n[1] User {DEMO_USER_ID} Enrollment:")
        print(f"    > Genuine Cryptographic Key: {genuine_key.hex()}")
        print("\n--- Legitimate Login Attempts ---")
        for i in range(1, 3):
            legitimate_login_template = genuine_templates[i:i+1]
            reproduced_key, dist = fuzzy_extractor.reproduce(legitimate_login_template, helper_data, tolerance=eer_threshold)
            print(f"\n[2.{i}] User {DEMO_USER_ID} Attempt:")
            print(f"    > Distance from enrollment template: {dist:.4f}")
            if reproduced_key:
                print(f"    > Reproduced Key: {reproduced_key.hex()}")
                print("    > Status: Access Granted")
                if genuine_key == reproduced_key:
                    print("    > Key Match: Correct key reproduced!")
            else:
                print("    > Status: Access Denied (False Reject)")
        print("\n--- Intruder Login Attempts ---")
        for i, intruder_id in enumerate(DEMO_INTRUDER_IDS):
            if intruder_id not in user_templates:
                print(f"\n[3.{i+1}] Intruder {intruder_id} not found in dataset. Skipping.")
                continue
            intruder_template = user_templates[intruder_id][0:1]
            intruder_key_attempt, dist = fuzzy_extractor.reproduce(intruder_template, helper_data, tolerance=eer_threshold)
            print(f"\n[3.{i+1}] Intruder {intruder_id} Attempt:")
            print(f"    > Distance from enrollment template: {dist:.4f}")
            if intruder_key_attempt:
                print(f"    > Key Generated by Intruder: {intruder_key_attempt.hex()}")
                print("    > Status: Access Granted (False Accept)")
            else:
                print("    > Key Generated by Intruder: None")
                print("    > Status: Access Denied")
        print("------------------------------------")

    # Plot the FAR and FRR curves
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, frr_list, label='False Rejection Rate (FRR)', color='blue')
    plt.plot(thresholds, far_list, label='False Acceptance Rate (FAR)', color='red')
    plt.axvline(x=eer_threshold, color='green', linestyle='--', label=f'EER Threshold ≈ {eer_threshold:.2f}')
    plt.title('False Acceptance Rate vs. False Rejection Rate')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('eer_curve.png')
    print("\n[INFO] EER curve plot saved to eer_curve.png")

    

