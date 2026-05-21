import os
import sys
import asyncio
import ngrok

async def main():
    try:
        # Start the ngrok tunnel forwarding traffic to local port 8001
        print("[INFO] Starting ngrok tunnel for port 8001...")
        listener = await ngrok.forward(8001, authtoken_from_env=True)
        print("\n" + "=" * 60)
        print("  🎉 Your API is now public at:")
        print(f"  {listener.url()}")
        print("  ")
        print("  Share this URL with your colleagues!")
        print("  Swagger UI is at: " + f"{listener.url()}/docs")
        print("=" * 60 + "\n")
        
        # Keep the tunnel open
        print("[INFO] Tunnel is open. Press Ctrl+C to close.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[INFO] Closing tunnel.")
    except Exception as e:
        print(f"\n[ERROR] Failed to start ngrok tunnel: {e}")
        print("Make sure you have set the NGROK_AUTHTOKEN environment variable.")
        print("Example: set NGROK_AUTHTOKEN=your_auth_token_here")

if __name__ == '__main__':
    asyncio.run(main())
