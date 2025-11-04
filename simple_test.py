import tensorflow as tf

print("="*60)
print("🖥️ GPU CHECK")
print("="*60)

# List available devices
print("\n📱 Available devices:")
devices = tf.config.list_physical_devices()
for device in devices:
    print(f"  - {device}")

# Check GPU specifically
print("\n🎮 GPU Devices:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Found {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"   - {gpu}")
else:
    print("❌ No GPU found")

# Check CPU
print("\n⚙️ CPU Devices:")
cpus = tf.config.list_physical_devices('CPU')
if cpus:
    print(f"✅ Found {len(cpus)} CPU(s)")
    for cpu in cpus:
        print(f"   - {cpu}")

print("\n" + "="*60)