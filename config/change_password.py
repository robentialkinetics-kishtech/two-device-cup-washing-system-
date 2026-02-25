import hashlib
import json

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Enter your new password
new_password = input("Enter new password: ")
password_hash = hash_password(new_password)

print(f"\nPassword: {new_password}")
print(f"Hash: {password_hash}")

# Update settings.json
with open("config/settings.json", 'r') as f:
    settings = json.load(f)

settings["user"]["password_hash"] = password_hash

with open("config/settings.json", 'w') as f:
    json.dump(settings, f, indent=2)

print("\n✓ Password updated successfully!")
print(f"New credentials - Username: {settings['user']['username']} / Password: {new_password}")
