"""
Analyze documents and recommend optimal chunk size.
"""

from pathlib import Path
import statistics
from src.ingestion.document_processor import DocumentProcessor
from src.rag.config import ChunkingConfig

def analyze_with_chunk_size(collection_path, chunk_size, overlap):
    """Analyze documents with specific chunk size."""
    
    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap)
    processor = DocumentProcessor(config)
    
    pdf_files = list(collection_path.glob("*.pdf"))
    if not pdf_files:
        return None
    
    all_chunks = []
    chunk_lengths = []
    
    for pdf in pdf_files:
        try:
            chunks = processor.process_file(pdf)
            all_chunks.extend(chunks)
            chunk_lengths.extend([len(chunk.page_content) for chunk in chunks])
        except Exception as e:
            print(f"   Error: {e}")
    
    if not all_chunks:
        return None
    
    return {
        'total_chunks': len(all_chunks),
        'avg_length': statistics.mean(chunk_lengths) if chunk_lengths else 0,
        'min_length': min(chunk_lengths) if chunk_lengths else 0,
        'max_length': max(chunk_lengths) if chunk_lengths else 0,
        'per_doc': len(all_chunks) / len(pdf_files) if pdf_files else 0,
        'samples': all_chunks[:2]
    }


def main():
    print("=" * 70)
    print("Chunk Size Analysis Tool")
    print("=" * 70)
    
    documents_dir = Path("./data/documents")
    if not documents_dir.exists():
        print("\n❌ data/documents/ not found!")
        return
    
    collections = [d for d in documents_dir.iterdir() if d.is_dir()]
    if not collections:
        print("\n❌ No collections found!")
        return
    
    for collection_dir in collections:
        pdf_files = list(collection_dir.glob("*.pdf"))
        if not pdf_files:
            continue
        
        print(f"\n{'='*70}")
        print(f"📁 Collection: {collection_dir.name} ({len(pdf_files)} PDFs)")
        print(f"{'='*70}")
        
        # Test configurations
        test_configs = [
            (500, 75),
            (750, 100),
            (1000, 150),
            (1000, 200),
            (1500, 200),
        ]
        
        print(f"\n{'Config':<15} {'Chunks':<8} {'Avg Len':<10} {'Per Doc':<10}")
        print("-" * 70)
        
        results = []
        for size, overlap in test_configs:
            stats = analyze_with_chunk_size(collection_dir, size, overlap)
            if stats:
                results.append((size, overlap, stats))
                marker = "← Current" if size == 1000 and overlap == 200 else ""
                print(f"{size}/{overlap:<10} {stats['total_chunks']:<8} "
                      f"{stats['avg_length']:<10.0f} {stats['per_doc']:<10.1f} {marker}")
        
        # Find current setting
        current = None
        for size, overlap, stats in results:
            if size == 1000 and overlap == 200:
                current = stats
                break
        
        if current:
            print(f"\n💡 Analysis:")
            chunks_per_doc = current['per_doc']
            
            if chunks_per_doc < 3:
                print(f"   ⚠️  Only {chunks_per_doc:.1f} chunks/doc = LOW granularity")
                print(f"   ✅ RECOMMEND: Reduce to 500-750 for better retrieval")
                print(f"   Expected: 2-3x more chunks, better specificity")
            elif chunks_per_doc > 10:
                print(f"   ⚠️  {chunks_per_doc:.1f} chunks/doc = TOO fragmented")
                print(f"   ✅ RECOMMEND: Increase to 1500 for more context")
            else:
                print(f"   ✅ Current (1000/200) is GOOD ({chunks_per_doc:.1f} chunks/doc)")
                print(f"   Optional: Try 750/100 for slightly better granularity")
            
            # Show sample
            if current['samples']:
                print(f"\n📄 Sample Chunk:")
                sample = current['samples'][0].page_content[:200].replace('\n', ' ')
                print(f"   Length: {len(current['samples'][0].page_content)} chars")
                print(f"   Content: {sample}...")
    
    print(f"\n{'='*70}")
    print("📝 RECOMMENDATIONS BY DOCUMENT TYPE:")
    print("="*70)
    print("""
Invoices/Receipts:    chunk_size=500,  overlap=100 (20%)  ✅
Short Docs (<2 pg):   chunk_size=750,  overlap=100 (13%)
Medium Docs (2-5 pg): chunk_size=1000, overlap=200 (20%)  ← Current
Long Docs (>5 pg):    chunk_size=1500, overlap=200 (13%)

Rule: Keep overlap at 15-20% of chunk_size
""")
    
    print("\n🔧 TO CHANGE:")
    print("="*70)
    print("""
1. Edit config.py:
   chunk_size: int = 500      # Change this
   chunk_overlap: int = 100   # Change this

2. Clear DB:
   ./venv/bin/python clear_db.py

3. Re-index:
   streamlit run app_advanced.py (upload docs)
""")


if __name__ == "__main__":
    main()