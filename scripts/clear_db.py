"""
Clear ALL collections from the vector database to start fresh.

Usage:
    python clear_db.py              # Interactive mode (asks for confirmation)
    python clear_db.py --force      # Force delete without confirmation
"""
import chromadb
from pathlib import Path
import sys

print("=" * 70)
print("ChromaDB Collection Cleaner")
print("=" * 70)

# Connect to ChromaDB
persist_dir = "./data/vector_store"
client = chromadb.PersistentClient(path=persist_dir)

# List all collections
try:
    collections = client.list_collections()
    
    if not collections:
        print("\n✓ No collections found - database is already empty!")
    else:
        print(f"\nFound {len(collections)} collection(s):")
        for col in collections:
            print(f"  - {col.name} ({col.count()} chunks)")
        
        print(f"\n{'='*70}")
        print("⚠️  WARNING: This will delete ALL collections and their data!")
        print("="*70)
        
        # Check for force flag
        force_delete = '--force' in sys.argv
        
        if force_delete:
            print("\n⚡ Force mode enabled - deleting without confirmation...")
            confirmed = True
        else:
            # Confirmation
            response = input("\nType 'yes' to confirm deletion: ").strip().lower()
            confirmed = (response == 'yes')
        
        if confirmed:
            print("\nDeleting collections...")
            deleted_count = 0
            
            # Get collection names before deleting (to avoid iterator issues)
            collection_names = [col.name for col in collections]
            collection_counts = {col.name: col.count() for col in collections}
            
            for name in collection_names:
                try:
                    client.delete_collection(name)
                    print(f"  ✓ Deleted '{name}' ({collection_counts.get(name, 0)} chunks)")
                    deleted_count += 1
                except Exception as e:
                    # Check if it's just already deleted
                    error_msg = str(e).lower()
                    if 'does not exist' in error_msg or 'not found' in error_msg:
                        print(f"  ⚠️  '{name}' already deleted or doesn't exist")
                    else:
                        print(f"  ✗ Error deleting '{name}': {e}")
            
            print(f"\n{'='*70}")
            print(f"✅ Successfully deleted {deleted_count} collection(s)!")
            print("="*70)
            
            print("\n📝 Next Steps:")
            print("  1. Update chunk size in config.py (if needed)")
            print("  2. Re-index documents:")
            print("     - Run: streamlit run app_advanced.py")
            print("     - Or: ./venv/bin/python main.py")
        else:
            print("\n❌ Deletion cancelled.")
            print("   No collections were deleted.")
            
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "="*70)