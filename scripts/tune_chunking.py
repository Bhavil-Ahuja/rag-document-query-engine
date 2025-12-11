"""
Script to test different chunking configurations and see results.

Run this to understand how chunk overlap affects your documents.
"""

from pathlib import Path
from src.ingestion.document_processor import DocumentProcessor
from src.rag.config import ChunkingConfig

def test_chunking_config(chunk_size, chunk_overlap):
    """Test a specific chunking configuration."""
    
    print(f"\n{'='*70}")
    print(f"Testing: chunk_size={chunk_size}, overlap={chunk_overlap}")
    print(f"Overlap percentage: {(chunk_overlap/chunk_size)*100:.1f}%")
    print(f"{'='*70}")
    
    # Create config
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Create processor
    processor = DocumentProcessor(config)
    
    # Test on a sample document
    pdf_files = list(Path("./data/documents/pdf_files/").glob("*.pdf"))
    
    if not pdf_files:
        print("No PDFs found in data/documents/pdf_files/")
        return
    
    test_pdf = pdf_files[0]  # Use first PDF
    print(f"\nTest file: {test_pdf.name}")
    
    # Process
    chunks = processor.process_file(test_pdf)
    
    print(f"\n📊 Results:")
    print(f"   Total chunks created: {len(chunks)}")
    
    if chunks:
        # Show first 3 chunks
        print(f"\n📄 Sample chunks (first 3):")
        for i, chunk in enumerate(chunks[:3], 1):
            content_preview = chunk.page_content[:100].replace('\n', ' ')
            print(f"\n   Chunk {i}:")
            print(f"   Length: {len(chunk.page_content)} chars")
            print(f"   Preview: {content_preview}...")
        
        # Check for overlap between consecutive chunks
        if len(chunks) > 1:
            print(f"\n🔄 Overlap Analysis:")
            chunk1_end = chunks[0].page_content[-chunk_overlap:]
            chunk2_start = chunks[1].page_content[:chunk_overlap]
            
            # Check how much actually overlaps
            common_text = ""
            for i in range(min(len(chunk1_end), len(chunk2_start))):
                if chunk1_end[i:] == chunk2_start[:len(chunk1_end)-i]:
                    common_text = chunk1_end[i:]
                    break
            
            if common_text:
                actual_overlap = len(common_text)
                print(f"   Expected overlap: {chunk_overlap} chars")
                print(f"   Actual overlap: {actual_overlap} chars")
                print(f"   Overlap text: '{common_text[:50]}...'")
            else:
                print(f"   No exact overlap found (text split at separator)")
    
    return len(chunks)


def main():
    """Test different configurations."""
    
    print("="*70)
    print("Chunk Overlap Testing Tool")
    print("="*70)
    
    # Test different configurations
    configs = [
        (1000, 0),      # No overlap
        (1000, 100),    # 10% overlap
        (1000, 200),    # 20% overlap (current)
        (1000, 300),    # 30% overlap
        (1000, 500),    # 50% overlap (high)
        (1000, 800),    # 80% overlap (excessive!)
    ]
    
    results = {}
    
    for chunk_size, overlap in configs:
        chunk_count = test_chunking_config(chunk_size, overlap)
        results[(chunk_size, overlap)] = chunk_count
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Chunk counts for different overlap settings")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'Chunks':<10} {'Overlap %':<15} {'Assessment'}")
    print("-"*70)
    
    for (size, overlap), count in results.items():
        overlap_pct = (overlap/size)*100
        
        # Assessment
        if overlap_pct == 0:
            assessment = "⚠️  No context preservation"
        elif 10 <= overlap_pct <= 20:
            assessment = "✅ Optimal"
        elif 20 < overlap_pct <= 30:
            assessment = "⚠️  Acceptable"
        elif 30 < overlap_pct <= 50:
            assessment = "⚠️  High redundancy"
        else:
            assessment = "❌ Excessive redundancy!"
        
        print(f"{size}/{overlap:<15} {count:<10} {overlap_pct:>6.1f}%       {assessment}")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS:")
    print("  ✅ For general use: chunk_size=1000, overlap=200 (20%)")
    print("  ✅ For long documents: chunk_size=1500, overlap=200 (13%)")
    print("  ✅ For short documents: chunk_size=500, overlap=100 (20%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()