import sys
sys.path.insert(0, '.')
from src.ingest.parse import parse_pdf, _is_likely_heading
from statistics import median
from pathlib import Path

# Look at blocks from the middle of oga090 to see field data structure
doc = parse_pdf(Path('data/raw/manual/oga090_statewide_field_data.pdf'))
blocks = doc.all_blocks
sizes = [b.font_size for b in blocks if b.font_size > 0]
med = median(sizes) if sizes else 0.0

print(f'oga090: total blocks={len(blocks)}, median_font={med}')
print('Blocks 200-230 (field data section):')
for b in blocks[200:230]:
    flag = 'H' if _is_likely_heading(b.text, b.font_size, med) else 'B'
    print(f'  [{flag}] fs={b.font_size:.1f}  {repr(b.text[:90])}')

print()
# Also check a good PDF to verify consolidation won't break it
doc2 = parse_pdf(Path('data/raw/manual/pending_drilling_permits.pdf'))
blocks2 = doc2.all_blocks
sizes2 = [b.font_size for b in blocks2 if b.font_size > 0]
med2 = median(sizes2) if sizes2 else 0.0
print(f'pending_drilling_permits: total blocks={len(blocks2)}, median_font={med2}')
print('First 20 blocks:')
for b in blocks2[:20]:
    flag = 'H' if _is_likely_heading(b.text, b.font_size, med2) else 'B'
    print(f'  [{flag}] fs={b.font_size:.1f}  {repr(b.text[:90])}')

print()
# Check word count distribution for oga090
word_counts = [len(b.text.split()) for b in blocks]
one_word = sum(1 for w in word_counts if w == 1)
two_word = sum(1 for w in word_counts if w == 2)
three_word = sum(1 for w in word_counts if w == 3)
many = sum(1 for w in word_counts if w > 3)
print(f'oga090 block word-count distribution: 1-word={one_word}, 2-word={two_word}, 3-word={three_word}, 4+={many}')

word_counts2 = [len(b.text.split()) for b in blocks2]
one_word2 = sum(1 for w in word_counts2 if w == 1)
many2 = sum(1 for w in word_counts2 if w > 3)
print(f'pending_permits block word-count distribution: 1-word={one_word2}, 4+={many2}')
