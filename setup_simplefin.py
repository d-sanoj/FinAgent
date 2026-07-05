import os
import base64
import urllib.request
import urllib.parse
import re

def setup_simplefin():
    env_file = ".env"
    if not os.path.exists(env_file):
        return

    with open(env_file, "r") as f:
        env_data = f.read()
    
    # Check if simplefin is still the default template
    if "your-simplefin-username" not in env_data:
        return
        
    print("⚠️  SimpleFin credentials not found in .env")
    token = input("Please enter your SimpleFin setup token (or press Enter to skip): ").strip()
    if not token:
        print("No token provided, skipping SimpleFin setup.")
        return
        
    try:
        print("🔄 Exchanging setup token for access credentials...")
        # Decode the token (which is a base64 encoded URL)
        claim_url = base64.b64decode(token).decode('utf-8')
        
        # POST to the claim URL
        import requests
        response = requests.post(claim_url)
        response.raise_for_status()
        access_url = response.text.strip()
        
        # Parse the access URL
        parsed = urllib.parse.urlparse(access_url)
        username = parsed.username
        password = parsed.password
        
        if not username or not password:
            raise ValueError("Failed to extract username/password from the Access URL.")
            
        # Update .env file
        env_data = re.sub(r'SIMPLEFIN_USERNAME=.*', f'SIMPLEFIN_USERNAME={username}', env_data)
        env_data = re.sub(r'SIMPLEFIN_PASSWORD=.*', f'SIMPLEFIN_PASSWORD={password}', env_data)
        
        with open(env_file, "w") as f:
            f.write(env_data)
            
        print("✅ SimpleFin credentials successfully generated and saved to .env")
        
    except Exception as e:
        print(f"❌ Failed to setup SimpleFin: {e}")

if __name__ == "__main__":
    setup_simplefin()
